# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:38:47 2021

@author: xinyu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import math

N = 2048 # set points number
T = 1 / 204800000 # set time interval 
path = r"C:\Users\ReddyLabAdmin\Dropbox\OCT\Photothermal\OCTphototehrmal fringes\20120909\\"
path1 = r"C:\Users\ReddyLabAdmin\Dropbox\OCT\Photothermal\OCTphototehrmal fringes\20120917\\"
path2 = r"D:\Dropbox\OCT\Photothermal\OCTphototehrmal fringes\20120909\\"
path3 = r"D:\Dropbox\OCT\Photothermal\OCTphototehrmal fringes\20120917\\"
samples = np.arange(0, N * T, T)
number_points = np.arange(2048)

bandwidth = 136 * 10**(-3) # wavelenth tuning range micrometer
centerwavelenth = 1291 * 10**(-3) # center wavelenth micrometer
k_min = 2 * math.pi / (centerwavelenth + bandwidth/2)
k_max = 2 * math.pi / (centerwavelenth - bandwidth/2)
k_step = (k_max - k_min) / (2048 - 1)
k_interpolation = np.linspace(k_min, k_max, 2048)
#%%
filename_2140 = path1 + "Data1.txt"
signal_fringe_raw_2140 = np.loadtxt(filename_2140, delimiter = '\t') # load txt file with 256 signals

signal_one_raw = signal_fringe_raw_2140[45]
signal_dc_part = np.mean(signal_one_raw)
signal_to_extract_phase_dc_subtracted = signal_one_raw - signal_dc_part
signal_to_extract_phase = signal_one_raw

signal_fringe_hilbert_2140 = hilbert(signal_to_extract_phase_dc_subtracted)




phase_radius = np.angle(signal_fringe_hilbert_2140)
phase_information = phase_radius 
phase_information_unwrap = np.unwrap(phase_radius)

def signal_plot():
    fig, axes = plt.subplots(5,1)
    axes[0].plot(number_points, signal_to_extract_phase_dc_subtracted)
    axes[0].set_xlabel("# of points (k space)")
    axes[0].set_title("Raw signal")
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[1].plot(number_points, np.array(signal_fringe_hilbert_2140).real)
    axes[1].set_xlabel("# of points (k space)")
    axes[1].set_title("Real part of signal after Hilbert Tran")
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    axes[2].plot(number_points, np.array(signal_fringe_hilbert_2140).imag)
    axes[2].set_xlabel("# of points (k space)")
    axes[2].set_title("Imaginary part of signal after Hilbert Tran (Hilbert Trans of signal)")
    axes[2].tick_params(axis='both', which='major', labelsize=20)
    axes[3].plot(k_interpolation, phase_information,'.')
    axes[3].set_xlabel("k/${μm^{-1}}$")
    axes[3].set_title("phase extracted through Hilbert Tran wrapped radius")
    axes[3].tick_params(axis='both', which='major', labelsize=20)
    axes[4].plot(k_interpolation, phase_information_unwrap,'.')
    axes[4].set_xlabel("k/${μm^{-1}}$", fontsize = 20)
    #axes[4].set_xticklabels(k_interpolation.tolist())
    axes[4].set_title("phase extracted through Hilbert Tran unwrapped radius")
    axes[4].tick_params(axis='both', which='major', labelsize=20)
    fig.set_size_inches(30, 30)
    plt.show()
signal_plot()


#%% raw signal import
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import math
import sys
sys.path.append(r"C:\Users\ReddyLabAdmin\Dropbox\OCT\Photothermal\OCTphototehrmal fringes\Data\20210922\txt")

filename ="1.txt"
signal_fringe_raw = np.loadtxt(filename, delimiter = '\t') # load txt file with 256 signals
signal_dc_part = np.mean(signal_fringe_raw, axis = 1) # average over column
signal_to_extract_phase_dc_subtracted = np.array(signal_fringe_raw) - np.reshape(signal_dc_part, (len(signal_dc_part),1))


signal_fringe_hilbert = hilbert(signal_to_extract_phase_dc_subtracted) # Hilbert Transform

phase_radius = np.angle(signal_fringe_hilbert)
phase_information = phase_radius
phase_information_unwrap = np.unwrap(phase_radius)

#%% optical path lenth extraction
'''
this is the function that's used to extract Optical Path Lenth(OPL) from 
unwrapped phase function of the wavenumber k and refractive index n
OPL = 1/2 / n * d(Φ)/d(k)
k = 2Π/λ
'''
def OPL_cal(unwrapped_phase_radius,start_point, end_point):
    n = 1.5 # refractive index of glass
    bandwidth = 136 * 10**(-3)# wavelenth tuning range\
    centerwavelenth = 1291 * 10**(-3) # center wavelenth
    k_min = 2 * math.pi / (centerwavelenth + bandwidth/2)
    k_max = 2 * math.pi / (centerwavelenth - bandwidth/2)
    k_step_local = (k_max - k_min) / (2048 - 1)
    phase_function = unwrapped_phase_radius[start_point:(end_point+1)]
    k_range = (end_point - start_point) * k_step_local
    phase_range = phase_function[-1] - phase_function[0]
    OPL = phase_range / k_range / 2 / n
    return OPL
optical_path_lenth = []
for i in range(256):
    optical_path_lenth.append(OPL_cal(phase_information_unwrap[i], 500, 1500))

x = np.arange(256)
fig = plt.gcf()
plt.plot(x, optical_path_lenth)
#plt.ylim([154,155])
plt.xlabel("measurements")
plt.ylabel("μm")
fig.set_size_inches(20,15)
plt.show()
std_OPL = np.std(optical_path_lenth)
#%% using OPL to calculate an accurate unambigouous surface profile
'''
z' = 1/(2 * k) * [Φ - 2*Π*int（（Φ - 2*k*z）/2Π）]
selected_phase_signal -> one of the unwrapped phase information
k_interpolation -> k space
phase_information_unwrap -> 256 raw unwrapped phase information 
z = m/2/n
m -> slop of fitting curve for single unwrapped phase information
n -> refractive index of glass
'''
index = 50
selected_phase_signal = phase_information_unwrap[index]
n = 1.5 # refrective index of glass
m,b = np.polyfit(k_interpolation[:1700], selected_phase_signal[:1700], 1)
z = m / 2 / n
def Z_dot_cal(i):
    phai = selected_phase_signal[i]
    k = k_interpolation[i]
    z_dot = 1 / (2 * k * n) * ( phai- 2*math.pi*(int)((phai - 2*k*z*n)/(2*math.pi)))
    return z_dot
z_dot = []
for j in range(100):
    selected_phase_signal = phase_information_unwrap[j]
    z_dot_row = []
    for i in range(1700):
        z_dot_row.append(Z_dot_cal(i))
    z_dot.append(z_dot_row)
#z_error = z_dot - np.mean(z_dot)
#%%
x = np.linspace(1,1700,1700)
fig = plt.gcf()
for j in range(100):
    selected_phase_signal = phase_information_unwrap[j]
    z_dot = []
    for i in range(1700):
        z_dot.append(Z_dot_cal(i))
    z_error = z_dot - np.mean(z_dot)
    plt.plot(x[:1700], np.array(z_dot)[:1700] * 1000, '.')
plt.xlabel("measurements", fontsize=50)
plt.xticks(fontsize=50)
plt.ylabel("nm", fontsize=50)
plt.yticks(fontsize=50)
plt.ylim([165500, 165600])
fig.set_size_inches(20,15)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs

signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_phase = np.angle(analytic_signal, deg = True)
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t, instantaneous_phase)
ax1.set_xlabel("time in seconds")
#ax1.set_ylim(0.0, 120.0)
ax2.plot(t, np.array(analytic_signal).imag)
ax2.set_xlabel("time in seconds")
fig.tight_layout()
#%%
raw_signal = signal_fringe_raw_2140[10]
np.savetxt("raw_signal.txt", raw_signal, fmt='%.4s')