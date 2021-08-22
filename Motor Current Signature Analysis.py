# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:54:04 2019
This code is created to find the anomaly in motor by analyzing the three phase current of motor
This code include three current analysis methods:
1. Fast Fourier transform to find anomaly signature in steady state motor condition
2. Wavelete packet decomposition to find non-steady state signature with respect to time 
3. Parks Circle to analyze three phase unbalnce of motor  

@author: Weiting Hsu
"""
#import Motor
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from os import listdir
import re
from scipy.fftpack import fft
from scipy import signal
import math
import PyWavelets as pywt


# Finding frequency signatue of motor
def findchf(faxis,Mag,chf,frange):
    df=faxis[2]-faxis[1]
    FindI=int(np.floor((chf-(frange/2))/df))
    FindF=int(np.floor((chf+(frange/2))/df+1))
    fRangeValue=Mag[FindI:FindF+1]
    lomax = np.where(fRangeValue[0:FindF-FindI+1] == np.max(fRangeValue))[0][0]
    lomin = np.where(fRangeValue[0:FindF-FindI+1] == np.min(fRangeValue))[0][0]
    lomax = FindI+lomax
    lomin = FindI+lomin
    [MaxV,Maxindex] = [np.max(fRangeValue),lomax]
    [MinV,Minindex] = [np.min(fRangeValue),lomin]
    MAX = [(Maxindex)*df,20*math.log10(MaxV)]
    Min = [(Minindex)*df,20*math.log10(MinV)]
    return [MAX, Min]

# calculating rms value
def rms(data):
    rms = np.sqrt(np.mean(data*data))
    return rms

# Motor module 
class Motor:
    
    def __init__(self,Ratepower,bearing_eq,samplerate,voltage,current,RateSpeed,RateFreq,PolePair):
        '''  First part is for motor spec setting  '''
        # motor ratefrequency setting
        self.RateFreq = RateFreq 
        # motor rate power
        self.power = Ratepower


        '''  '''
        # sensor sampling rate 
        self.samplerate = samplerate 
        # 三相電流
        self.current = np.array(current)
        # 三相電壓
        self.voltage = np.array(voltage)
        # 其中單相電流
        self.dataI = current[0]
        # 其中單相電壓
        self.dataV = voltage[0]
        # 電流資料長度
        self.pointnum = len(self.dataI)
        # FFT 轉換找主頻
        Magdata=abs(fft(self.dataV)/self.pointnum)
        self.Main_freq = np.argmax(Magdata)/(self.pointnum/self.samplerate)
        # 找負載率
        self.loadfactor = rms(self.dataV)*rms(self.dataI)/Ratepower
        # 從額定頻率 額定速度 負載率 找轉速{主頻*60/極對數-(額定頻率*60/極對數-額定轉速)*負載率}
        self.speed = self.Main_freq*60/PolePair-(self.RateFreq*60/PolePair-RateSpeed)*self.loadfactor
        # 從電機頻率 機械頻率找華差
        self.slip = (self.Main_freq*60/PolePair-self.speed)/(self.Main_freq*60/2)   
        # 求軸承損壞特徵頻率
        #求滾珠損壞頻率
        self.RollingBall =  bearing_eq[2][0]*self.speed + bearing_eq[2][1]+self.Main_freq
        # 求外環特徵頻率
        self.OuterRace = bearing_eq[1][0]*self.speed + bearing_eq[1][1]+self.Main_freq
        # 求內還特徵頻率
        self.InnerRace = bearing_eq[0][0]*self.speed + bearing_eq[0][1]+self.Main_freq
        # 求軸偏心特徵頻率
        self.misalignR = self.Main_freq+self.speed/60
        self.misalignL = self.Main_freq-self.speed/60
        # 求轉子條損壞特徵頻率
        self.BrokenR = (1+2*self.slip)*self.Main_freq
        self.BrokenL = (1-2*self.slip)*self.Main_freq
        # 求 可得最大斜坡
        self.harmonics = 200 #int(np.floor(self.samplerate/2/self.Main_freq))

    def fft_feature(self):
        Num = len(self.dataI)
        # 找頻率 x 軸
        faxis = np.arange(0,self.samplerate/2,(self.samplerate/Num)) 
        # 求得 fft 能量值
        Magdata=abs(fft(self.dataI )/Num)
        yf2 = Magdata[range(int(Num/2))]  #由于对称性，只取一半区间

        # 把所有特徵頻率放在 list 中
        BrokenFeature = [self.BrokenR,self.BrokenL,self.misalignL,self.misalignR,self.InnerRace,self.OuterRace,self.RollingBall]
        
        # 抓取特徵頻率範圍
        fspan = 1
        
        # 矩陣宣告
        Fall_l = np.zeros([2,self.harmonics+7])
        Fall_h = np.zeros([2,self.harmonics+7])
        
        # 把所找到的特徵頻率存進矩陣中
        Fall_h[:,0],Fall_l[:,0] =  findchf(faxis,yf2,self.Main_freq ,fspan)
        
        # 找主頻能量和損壞特徵頻率能量之差值
        for o in range(1,self.harmonics):
            Fall_h[:,o],Fall_l[:,o] =  findchf(faxis,yf2,self.Main_freq*(o+1) ,fspan)
            Fall_h[1,o] = Fall_h[1,0]-Fall_h[1,o]
            Fall_l[1,o] = Fall_l[1,0]-Fall_l[1,o]
        
        for k in range(0,len(BrokenFeature)):
            # 因轉子條損壞頻率離主頻近，所以把特徵頻率附近搜尋的範圍拉近
            if k<=1:
                fspanc = 0.4
            else:
                fspanc = fspan
            Fall_h[:,self.harmonics+k],Fall_l[:,self.harmonics+k]  = findchf(faxis,Magdata,BrokenFeature[k],fspanc)    
            Fall_h[1,self.harmonics+k] = Fall_h[1,0]-Fall_h[1,self.harmonics+k]
            Fall_l[1,self.harmonics+k] = Fall_l[1,0]-Fall_h[1,self.harmonics+k]
    
        return Fall_h,Magdata,faxis
    
    # WPD 特徵擷取
    def WPD_coefficient(self,resamplerate,level):
        # 包含 偏心特徵頻率和軸承損壞頻率
        feature_num = 5
        # 找頻率解析度
        timespan = len(self.dataI)/self.samplerate
        # 把 WPD 頻譜分為幾份
        fspandiff = pow(2,level)
        # 把原始資料 Resample 後的 資料
        Iresam = signal.resample(self.dataI,int(resamplerate*timespan) )    
        # 宣告其 node 位置
        f = np.zeros([feature_num,1])    
        # 宣告 WPD coefficient 
        Fault_wpd_coefficient = np.zeros([feature_num,110])
        # WPD 頻譜每份所佔的頻率
        fspan = resamplerate/fspandiff/2
        # 小波選擇
        wavelet = 'dmey'
        # 
        order = "freq"  
        # 小波分解 選用小波 dmey extension mode:periodization(避免 Aliasing) ; level = 8
        wp = pywt.WaveletPacket(Iresam, wavelet, 'zero', level)
        # order 選擇頻域的 order
        nodes = wp.get_level(level, order)
        # 提取每個 node 的值
        values = np.array([n.data for n in nodes], 'd')
        # 找取特徵頻帶所在的node
        f[0] = np.floor(self.misalignL/fspan)
        f[1] = np.floor(self.misalignR/fspan)
        f[2] = np.floor(self.OuterRace/fspan)
        f[3] = np.floor(self.InnerRace/fspan)
        f[4] = np.floor(self.RollingBall/fspan)

        #找出損壞對應頻帶
        MisL = [f[0]*fspan, (f[0]+1)*fspan]
        MisR = [f[1]*fspan, (f[1]+1)*fspan]   
        fo = [f[2]*fspan, (f[2]+1)*fspan]
        fin = [f[3]*fspan, (f[3]+1)*fspan]
        fre = [f[4]*fspan, (f[4]+1)*fspan]
        
        
        freqband = [fo,fin,fre,MisL,MisR]
        
        MeanWpd,RmsWpd,StdWpd = ([] for _ in range(3))
        # # 提取特徵頻率之node 並 取 rms 值 和 mean 值
        for j in range(0,feature_num,1):
            Fault_wpd_coefficient[j,:] = values[int(f[j]),:]
            RmsWpd.append(rms(Fault_wpd_coefficient[j,:]))
            MeanWpd.append(np.mean(abs(Fault_wpd_coefficient[j,:])))
            StdWpd.append(np.std(Fault_wpd_coefficient[j,:]))
        return RmsWpd,MeanWpd,StdWpd,freqband
    # Parks vector function
    def Parks_vector(self):
        # Park's 座標轉換公式
        K = np.matrix([[1,-1/2,-1/2],[0,np.sqrt(3)/2,-np.sqrt(3)/2],[1/2,1/2,1/2]])*2/3
        # 矩陣宣告
        Cab = np.zeros([3,self.pointnum])
        
        # 將三相電壓 轉換
        for i in range(0,self.pointnum):
            Cab[:,i] = np.dot(K,self.current[:,i])
        # 正則畫
        data_C = np.array([Cab[0,:]/np.max([Cab[0,:],Cab[1,:]]),Cab[1,:]/np.max([Cab[0,:],Cab[1,:]])])
        # 求 covarian 求其主軸長度
        covmat = np.cov(data_C)
        # 看其主軸長度之差值 可知其形狀之真圓度
        cov_PV = abs(covmat[0,0]-covmat[1,1])
        return cov_PV,data_C
    # 求電壓的 THD
    def THD(self):
        
        frange = 2.7
        Num = len(self.dataV);
        faxis = np.arange(0,self.samplerate/2,(self.samplerate/Num)) 
        Magdata=abs(fft(self.dataV)/Num)
        
        df = self.samplerate/Num
        Main_freq_index = self.Main_freq*(1/df)
        harmonic_num  = list(range(2,self.harmonics+1))
        harmonic_index = np.multiply(Main_freq_index , harmonic_num)
        i=0 
        maxF = np.zeros([len(harmonic_index),2])
        minF = np.zeros([len(harmonic_index),2])
        for chf in harmonic_index: 
            maxF[i,:],minF[i,:] = findchf(faxis,Magdata,chf*df,frange)
            i=i+1
        # 算出倍頻對應頻率    
        maxF[:,0] = np.multiply(maxF[:,0],1/df)
        # 整數化
        A = maxF[:,0].astype(int)
        # THD 公式
        thd = np.sqrt(np.dot(Magdata[A],Magdata[A]))/Magdata[int(Main_freq_index)]
        return thd     
    # 求 三相 Imbalance  
    def imbalance(self):
        # A,B,C 三相
        Ia = rms(self.current[0,:])
        Ib = rms(self.current[1,:])
        Ic = rms(self.current[2,:])
        # 三相不平衡公式
        Im = (Ia+Ib+Ic)/3
        ImbalancedI = [abs(Ia-Im)/Im,abs(Ib-Im)/Im,abs(Ic-Im)/Im]
        return ImbalancedI

''' Main '''

plt.close('all')

motorname = '1'

samplerate = 25000;
resamplerate = 1280;
level = 8

startT = 0;
finalT = 10;


# 抓取 DATA 的總長
T = 1/samplerate;
time = finalT-startT;
t =np.arange(startT+T,finalT+T,T) ;
# 抓取 DATA 時間
date = ['8.1','8.15','9.5','10.9','11.13','12.11','1.15'];

folder, loaddata= list(),list()  

period = len(date);

point_num = len(t)
bearing_size = []
RateSpeed1 = 1785
RateFreq1 = 60   
PolePair1 = 2
Ratepower_Hp = 250
Ratepower_Watt1 = Ratepower_Hp*745.7
# 求 BEARING 特徵頻率的線性方程
bearing6313_eq = [[0.08215,1/200],[0.05118,1/1000],[0.06787,1/1000]]
bearing6318_eq = [[0.08178,7/1000],[0.05155,3/1000],[0.06971,1/1000]]

# 宣告 list
[Total_StdWpd,Date_StdWpd,Date_thd,Total_thd,Date_Falt_fft,Total_Falt_fft,Date_RmsWpd,Total_RmsWpd,Date_MeanWpd,Total_MeanWpd,Date_covPV,
Total_covPV,Date_Imbalance,Total_Imbalance] = ([] for i in range(14))  


for i in range(0,period):    
    
    # Data 匯入
    inputfolder = ['D:/Data/',motorname,'/',date[i]]
    inputfolder = ' '.join(inputfolder)
    inputfolder = re.sub('\s','',inputfolder)
    folder.append(inputfolder)
    files = listdir(inputfolder)
    amount = len(files)
    
    for j in range(0,amount): 
        
        # 資料提取
        address = inputfolder+'/'+files[j]
        loaddata = (pd.read_csv(address, sep="\t", header=0,skiprows=list(range(10))))
        data = np.array(loaddata);
        
        if i<=2:
        ## 資料抓取
        ##7.8.9 三相電壓  10.11.12 三相電流 華邦
            cloumnumv = [7,8,9] ;
            cloumnumc = [10,11,12] ;
            
            dataI1 = data[:,cloumnumc[0]-1];
            dataI2 = data[:,cloumnumc[1]-1];
            dataI3 = data[:,cloumnumc[2]-1];
        
            dataVA = data[:,cloumnumv[0]-1];
            dataVB = data[:,cloumnumv[1]-1];
            dataVC = data[:,cloumnumv[2]-1];
        
            I1 = dataI1[list(range(0,point_num))];
            I2 = dataI2[list(range(0,point_num))];
            I3 = dataI3[list(range(0,point_num))];
        
            VA = dataVA[list(range(0,point_num))];
            VB = dataVB[list(range(0,point_num))];
            VC = dataVC[list(range(0,point_num))];
        
            V1 = VA-VB;
            V2 = VB-VC; 
            V3 = VC-VA;
        else:
            cloumnumv = [8,9,10] ;
            cloumnumc = [12,13,14] ;
                     
            dataI1 = data[:,cloumnumc[0]-1];
            dataI2 = data[:,cloumnumc[1]-1];
            dataI3 = data[:,cloumnumc[2]-1];
        
            I1 = dataI1[list(range(0,point_num))];
            I2 = dataI2[list(range(0,point_num))];
            I3 = dataI3[list(range(0,point_num))];
        
            dataVA = data[:,cloumnumv[0]-1];
            dataVB = data[:,cloumnumv[1]-1];
            dataVC = data[:,cloumnumv[2]-1];
        
            V1 = dataVA[list(range(0,point_num))];
            V2 = dataVB[list(range(0,point_num))];
            V3 = dataVC[list(range(0,point_num))];
            
        voltage = [V1,V2,V3]
        
        current = [I1,I2,I3]
        
        # 馬達種類設定
        Motor1 =  Motor(Ratepower_Watt1,bearing6313_eq,samplerate,voltage,current,RateSpeed1,RateFreq1,PolePair1) 
        # 找 fft 特徵指標
        Fall_h, Magdata, faxis = Motor1.fft_feature()
        # 找 WPD 特徵指標
        RmsWpd,MeanWpd,StdWpd,freqband = Motor1.WPD_coefficient(resamplerate,level)
        # 找 Park's vector 值 和 定子繞阻損壞特徵
        cov_PV,data_C = Motor1.Parks_vector()
        # 找 THD
        thd = Motor1.THD()
        # 找三相不平衡指標看定子側繞組和轉子條損壞是否明顯
        Imbalance = Motor1.imbalance()
        
        # 把 特徵整合在一起
        Date_Falt_fft.append(Fall_h)
        Date_thd.append(thd) 
        Date_MeanWpd.append(MeanWpd)
        Date_RmsWpd.append(RmsWpd)
        Date_covPV.append(cov_PV)
        Date_Imbalance.append(np.max(Imbalance,0))
        Date_StdWpd.append(StdWpd)
    
    Total_thd.append(np.mean(Date_thd))
    Total_Falt_fft.append(np.mean(abs(np.array(Date_Falt_fft)),0))   
    Total_RmsWpd.append(np.mean(abs(np.array(Date_RmsWpd)),0))
    Total_MeanWpd.append(np.mean(abs(np.array(Date_MeanWpd)),0))
    Total_covPV.append(np.mean(abs(np.array(Date_covPV)),0))
    Total_Imbalance.append(np.mean(abs(np.array(Date_Imbalance)),0))
    Total_StdWpd.append(np.mean(Date_StdWpd,0))
    
    Date_StdWpd,Date_thd,Date_Falt_fft,Date_RmsWpd,Date_MeanWpd,Date_covPV,Date_Imbalance = ([] for _ in range(7))

Total_MeanWpd = np.array(Total_MeanWpd)
Total_Falt_fft = np.array(Total_Falt_fft)
Total_RmsWpd = np.array(Total_RmsWpd)
Total_Imbalance = np.array(Total_Imbalance)
Total_StdWpd = np.array(Total_StdWpd)


[WPD_Std_bearing_fault,WPD_Std_Mis_fault,bearing_fault, Mis_fault, BrokenBar_fault, WPD_RMS_Mis_fault,
 WPD_RMS_bearing_fault,WPD_Mean_Mis_fault ,WPD_Mean_bearing_fault ]= ([] for _ in range(9))

# 把特徵整合在一起
for i in range(0,period):
    bearing_fault.append(sum(Total_Falt_fft[i,1,Motor1.harmonics+4:Motor1.harmonics+7]))
    Mis_fault.append(sum(Total_Falt_fft[i,1,Motor1.harmonics+2:Motor1.harmonics+4]))
    BrokenBar_fault.append(sum(Total_Falt_fft[i,1,Motor1.harmonics:Motor1.harmonics+2]))
    WPD_RMS_bearing_fault.append(sum(Total_RmsWpd[i,1:5]))
    WPD_RMS_Mis_fault.append(sum(Total_RmsWpd[i,0:2]))
    WPD_Mean_bearing_fault.append(sum(Total_MeanWpd[i,1:5]))
    WPD_Mean_Mis_fault.append(sum(Total_MeanWpd[i,0:2]))
    WPD_Std_bearing_fault.append(sum(Total_StdWpd[i,1:5]))
    WPD_Std_Mis_fault.append(sum(Total_StdWpd[i,0:2]))
    
# Feature dictionary
Feature = {"FFT Bearing Fault":bearing_fault, "FFT Misalignment Fault":Mis_fault, "FFT BrokenBar Fault": BrokenBar_fault,
           "WPD RMS Bearing Fault":WPD_RMS_bearing_fault,"WPD RMS Mis Fault":WPD_RMS_Mis_fault,"WPD Mean Bearing Fault":WPD_Mean_bearing_fault,
           "WPD Mean Mis fault":WPD_Mean_Mis_fault,"THD":np.multiply(Total_thd,100),"Covariance of Parks Vector":Total_covPV,"Imbalance of current":np.multiply(Total_Imbalance,100),
           "WPD Std Mis fault":WPD_Std_Mis_fault,"WPD Std bearing fault":WPD_Std_bearing_fault}
# 畫圖的 range
yaxisrange = [(0,300),(0,300),(0,300),(0,100),(0,100),(0,100),(0,100),(0,50),(0,0.05),(0,6),(0,50),(0,100)]
# 圖 的 ylabel
ylablename = ['Amplitude(A) db scale','Amplitude(A) db scale','Amplitude(A) db scale','Amplitude(A)','Amplitude(A)','Amplitude(A)','Amplitude(A)'
              ,'Percentage(%)','Differemce(A)','Percentage(%)','Amplitude(A)','Amplitude(A)']
# 畫圖
for i ,j in zip(Feature.keys(),range(len(Feature))):
    plt.figure(figsize=(12,8))
    bar_plot = plt.bar(date,Feature[i])
    plt.title('Motor '+motorname+' '+i)
    for rect,num in zip(bar_plot,range(len(bar_plot))):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, round(Feature[i][num],4), ha='center', va='bottom')
        plt.ylim(yaxisrange[j])
        plt.grid(True)
        plt.ylabel(ylablename[j])
        plt.xlabel('date')   
        
# 畫 parks vector       
Crad = rms(data_C[0,:])+rms(data_C[1,:])
angle = np.arange(-np.pi,np.pi,0.01)   

Cx,Cy = list(),list()
for i in angle:
    Cx.append(Crad/np.sqrt(2)*np.cos(i))
    Cy.append(Crad/np.sqrt(2)*np.sin(i))
plt.figure(13)
plt.plot(data_C[0,:],data_C[1,:],'b',Cx,Cy,'r')  

    
    


