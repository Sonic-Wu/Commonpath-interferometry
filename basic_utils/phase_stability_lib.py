# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:04:10 2021

@author: Xinyu Wu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import math

# Set up constant number
N = 2048
T = 1 / 204800000

path2 = r"D:\Dropbox\OCT\Photothermal\OCTphototehrmal fringes\20120909\\"
path3 = r"D:\Dropbox\OCT\Photothermal\OCTphototehrmal fringes\20120917\\"

number_points = np.arange(2048)

bandwidth = 136 * 10**(-3) # wavelenth tuning range micrometer
centerwavelenth = 1291 * 10**(-3) # center wavelenth micrometer
k_min = 2 * math.pi / (centerwavelenth + bandwidth/2)
k_max = 2 * math.pi / (centerwavelenth - bandwidth/2)
k_step = (k_max - k_min) / (2048 - 1)
k_interpolation = np.linspace(k_min, k_max, 2048)

def unwrapped_phase(filename):
    
    # load txt file with 256 signals
    signal_fringe_raw = np.loadtxt(filename, delimiter = '\t')
    
    # average over column
    signal_dc_part = np.mean(signal_fringe_raw, axis = 1) 
    
    # subtract the dc part
    signal_to_extract_phase_dc_subtracted = np.array(signal_fringe_raw) -\
                         np.reshape(signal_dc_part, (len(signal_dc_part),1))
    
    # perform hilbert transform to get analytic signal
    Analytic_signal_fringe = hilbert(signal_to_extract_phase_dc_subtracted)
    
    # extract the phase information
    phase_radius = np.angle(Analytic_signal_fringe) 
    phase_information = phase_radius                         # wrapped phase
    phase_information_unwrap = np.unwrap(phase_radius)       # unwrapped phase
    return signal_fringe_raw, signal_to_extract_phase_dc_subtracted, Analytic_signal_fringe,\
            phase_information, phase_information_unwrap

def signal_plot(signal_tuple):
    title = ["Raw signal", "Real part of signal after Hilbert Tran",\
             "Imaginary part of Analytic signal after Hilbert Tran",\
             "wrapped phase of Analytic signal (radius)",\
             "wrapped phase of Analytic signal (radius)"]
    x_label = ["# of points (k space)", "# of points (k space)",\
               "# of points (k space)", "k/${μm^{-1}}$",\
               "k/${μm^{-1}}$"]
    x = [number_points, number_points, number_points, k_interpolation,\
         k_interpolation]
    fig,axes = plt.subplots(5,1)
    for i in range(5):
        axes[i].plot(x[i], signal_tuple[i])
        axes[i].set_xlabel(x_label[i])
        axes[i].title(title[i])
        axes[i].tick_params(axis='both', which='major', labelsize=20)
    fig.set_size_inches(30,30)
    plt.show()

def OPL_cal(signal, k_space, refractive_index):
    '''    
    this is the function that's used to extract Optical Path Lenth(OPL) from 
    unwrapped phase function of the wavenumber k and refractive index n
    OPL = 1/2 / n * d(Φ)/d(k)
    k = 2Π/λ
    
    z' = 1/(2 * k) * [Φ - 2*Π*int（（Φ - 2*k*z）/2Π）]
    selected_phase_signal -> one of the unwrapped phase information
    k_interpolation -> k space
    phase_information_unwrap -> 256 raw unwrapped phase information 
    z = m/2/n
    m -> slop of fitting curve for single unwrapped phase information
    n -> refractive index of glass
    '''
    if len(signal) != len(k_space):
        print("ERROR: signal and k should have same dimension")
        return
    if refractive_index <= 0:
        print("ERROR: refractive index should be positive")
        return
    n = refractive_index # refractive index of glass
    bandwidth = 136 * 10**(-3)# wavelenth tuning range\
    centerwavelenth = 1291 * 10**(-3) # center wavelenth
    #k_min = 2 * math.pi / (centerwavelenth + bandwidth/2)
    #k_max = 2 * math.pi / (centerwavelenth - bandwidth/2)
    
    m,b = np.polyfit(k_space[250:1000], signal[250:1000], 1)
    z = m / 2 / n

    z_prime = []
    for i in range(len(signal)):
        phai = signal[i]
        k = k_space[i]
        z_prime.append( 1 / (2 * k * n) * ( phai - 2*math.pi*(int)((phai - 2*k*z*n)/(2*math.pi))))
    return z, z_prime

def OPL_extraction(filename):
    pass
    