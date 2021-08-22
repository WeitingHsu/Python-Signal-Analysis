import numpy as np
import matplotlib.pyplot as plt

# Original function(signal) is applied on and kernel(system) would be output of kernel(system)
# In this sense, totoal data point would be n(signal data number)+m(kernel data number)-1
# 
# 


def generatesignal(t,freq1,freq2,freq3,Amp1,Amp2,Amp3):
    return Amp1*np.cos(freq1*t)+Amp2*np.sin(freq2*t)+Amp3*np.cos(freq3*t+4/3*np.pi)


def gaussiankernel(kernelspan,amp,mean,std):
    return amp*np.exp(-(kernelspan-mean)**2/2*std**2)

def convolution(signal,kernel):
    n = len(signal)
    m = len(kernel)
    y = np.empty(n+m-1)
    zerospaddingsignal = np.insert(signal,0,np.zeros(m))
    zerospaddingsignal = np.append(zerospaddingsignal,np.zeros(m),0)
    print(m)
    print(len(zerospaddingsignal[0:m]))
    b = np.dot(kernel,zerospaddingsignal[0:m])

    for i in range(n+m-1):
        y[i] = np.dot(kernel,zerospaddingsignal[i:m+i])
    return y
        





# This part is simulation signal setting
samplerate = 200
dt = 1/samplerate
Timestart = 0
Timend = 20
Timespan = Timend - Timestart
t = np.linspace(Timestart,Timend,samplerate*Timespan)
# t = np.arange(Timestart,Timend,dt)

# After setting, 
signal = generatesignal(t,freq1 = 10,freq2 = 20,freq3 = 30,Amp1 = 2,Amp2 =3,Amp3 =4)



plt.figure(1)
plt.xlim(Timestart,Timend)
plt.title("Signal")
plt.xlabel("Time")
plt.ylabel("Current(A)")
plt.plot(t,signal)



kernel = gaussiankernel(t,amp = 1,mean = 5,std = 6)

# Setting up threshold to remove value which is close to 0
Threshold = 0.001
kernel = kernel[(kernel>=Threshold)]


plt.figure(2)
plt.title("Kernel")
plt.xlabel("Time")
plt.ylabel("Current(A)")
plt.plot(kernel)

# Doing convolution
systemoutput = convolution(signal,kernel)

plt.figure(3)
plt.title("Syttem output")
plt.xlabel("Time")
plt.ylabel("Current(A)")
plt.plot(systemoutput)
plt.show()


