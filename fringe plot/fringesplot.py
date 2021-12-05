# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:37:32 2021

@author: Xinyu
"""

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
#%% load in data
N = 2048
T = 1.0 / 204800000
#%%
signal_dc = np.loadtxt("DC.txt", delimiter = '\t', usecols = 1)
signal_interferometry = np.loadtxt("common path fringes.txt", delimiter = '\t', usecols = 1)
x = np.arange(0, N * T, T)
yf = fft(signal_interferometry)
xf = fftfreq(N, T)[:N//2]
max_value = np.max(yf[1:])
max_angle = np.angle(max_value)

plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.title("amplitude extract")
plt.show()
phase = np.angle(yf, deg = False)
plt.plot(xf, phase[:N//2])
plt.xlabel("frequency")
plt.ylabel("phase")
plt.ylim(-np.pi,np.pi)
plt.title("phase extract")
plt.show()
#%% fft and extract the greatest magnitude
signal_interferometry_load = np.loadtxt("1.txt", delimiter = '\t')
signal_interferometry_256 = []
fft_signal_interferometry_256 = []
fft_max_complex_value = []
fft_max_phase = []
for i in range(256):
    signal_interferometry_256.append(signal_interferometry_load[i,:])
    fft_signal_interferometry_256.append(fft(signal_interferometry_load[i,:]))
    fft_max_complex_value.append(fft_signal_interferometry_256[i][np.abs(fft_signal_interferometry_256[i][1:]).argmax() + 1])
    #fft_max_phase.append(math.atan2(fft_max_complex_value[i].imag, fft_max_complex_value[i].real))
    fft_max_phase.append(np.angle(fft_max_complex_value[i], deg = True))
#print('The real part of the complex number is {}'.format(fft_signal_interferometry_256[1].real))
#print(np.abs(fft_signal_interferometry_256[1][1])/2048)
#%% plot signal
i = 2
fig,axes = plt.subplots(2,1)
x = np.arange(0, N, 1)
axes[0].plot(x, 2 / N * np.abs(fft_signal_interferometry_256[i]), '.')
axes[0].set_xlim(0,100)
axes[0].set_title("FFT")
axes[0].set_xlabel("frequency")
axes[0].set_ylabel("amplitude")
axes[1].plot(x, signal_interferometry_256[i])
axes[1].set_title("Singal of interferometry")
axes[1].set_xlabel("# of sample points")
axes[1].set_ylabel("amplitude")
fig.set_size_inches(30,10)
plt.show()
#%% plot 256 interferometry fringes vs phase information
#max_fft_index = []
#for i in range(256):
    #max_fft_index.append(np.abs(fft_signal_interferometry_256[i][1:]).argmax())
x_phase = np.arange(0, 256, 1)
fig = plt.gcf()
plt.plot(x_phase, fft_max_phase)
plt.ylim(-180, 180)
plt.ylabel("radius")
plt.xlabel("# of the fringes")
plt.title("phase information of fringes")
fig.set_size_inches(30,10)
plt.show()

#%%
x = np.arange(0, 256, 1)
fig,axes = plt.subplots(2,1)
axes[0].plot(x, np.array(fft_max_complex_value).real, '.')
axes[0].set_title("Real part")
axes[0].set_xlabel("Fringes #")
axes[0].set_ylabel("Value")
#axes[0].set_ylim(15000, 20000)
axes[1].plot(x, np.array(fft_max_complex_value).imag, '.')
axes[1].set_title("Imag part")
axes[1].set_xlabel("Fringes #")
axes[1].set_ylabel("Value")
#axes[1].set_ylim(-6000,6000)
fig.set_size_inches(30,10)
plt.show()
#%% plot signal
x = np.arange(0, 2047, 1)
fig,axes = plt.subplots(4,1)
for i in range(4):
    axes[i].plot(x, np.abs(fft_signal_interferometry_256[i][1:]), '.')
    axes[i].set_xlim(20, 40)
fig.set_size_inches(20,10)
plt.show()
#%% Sample
from scipy.fft import fft, fftfreq
# Number of sample points
N = 800
# sample spacing
T = 1.0 / 600.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()
#%%
mean_value = np.mean(fft_max_phase)
std_value = np.std(fft_max_phase)

