import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import angle


def generatesignal(t,freq1,freq2,freq3,Amp1,Amp2,Amp3):
    return Amp1*np.cos(2*np.pi*freq1*t)+Amp2*np.sin(2*np.pi*freq2*t)+Amp3*np.cos(2*np.pi*freq3*t+4/3*np.pi)



def DFT(signal,t):
    N = len(t)
    n = np.arange(0,N,1)
    X = np.empty(N,dtype=complex)
    for k in range(N-1):
        E = np.exp(-1j*2*np.pi*k*n/N)
        X[k+1] = 1/N*np.dot(signal[0:N-1],E[0:N-1])
    return X


samplerate = 1000
dt = 1/samplerate
ini_time = 0
end_time = 10
t = np.arange(ini_time,end_time,dt)

signal = generatesignal(t,freq1 = 200,freq2 = 300,freq3 = 100,Amp1 = 2,Amp2 =3,Amp3 =4)

plt.figure(1)
plt.plot(t,signal)
plt.xlim(ini_time,end_time)
plt.title("Signal")
plt.xlabel("Time")
plt.ylabel("Current(A)")
plt.plot(t,signal)

F = DFT(signal,t)
N = len(F)
MagF = abs(F)
AngleF = angle(F)
faxis = samplerate/2*np.arange(0,1,1/(N/2))
plt.figure(2)
print(np.size(faxis))
print(np.size(MagF[0:int(N/2)]))
plt.plot(np.transpose(faxis),MagF[0:int(N/2)])
plt.title("Signal DFT spectrum")
plt.xlabel("Frequency")
plt.ylabel("Current(A)")
plt.show()




