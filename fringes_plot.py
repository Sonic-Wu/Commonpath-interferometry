# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:42:29 2021

@author: xinyu
"""

#%% in one
import numpy as np
from scipy import interpolate
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
#%%
signal_fringe_raw = []
for i in range(10):
    filename = str(i+1) + ".txt"
    signal_fringe_raw.append(np.loadtxt(filename, delimiter = '\t'))

signal_interferometry_256 = []
fft_signal_interferometry_256 = []
fft_max_complex_value = []
fft_max_phase = []
fft_signal_interferometry_256.append(fft(signal_fringe_raw[0]))

#%% data pocessing
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

N = 2048
T = 1 / 204800000

horizontal_axis = np.arange(0, N * T, T)

filename_2140 = "2140/1.txt"
filename_3000 = "closing apenture/1.txt"
signal_fringe_raw_2140 = np.loadtxt(filename_2140, delimiter = '\t')
signal_fringe_raw_3000 = np.loadtxt(filename_3000, delimiter = '\t')
fft_signal_fringe_2140 = fft(signal_fringe_raw_2140)
fft_signal_fringe_3000 = fft(signal_fringe_raw_3000)
fft_max_complex_value_2140 = []
fft_max_complex_value_3000 = []
for i in range(256):
    fft_max_complex_value_2140.append(fft_signal_fringe_2140[i][np.abs(fft_signal_fringe_2140[i][1:]).argmax() + 1])
    fft_max_complex_value_3000.append(fft_signal_fringe_3000[i][np.abs(fft_signal_fringe_3000[i][1:]).argmax() + 1])
fft_max_phase_2140 = np.angle(fft_max_complex_value_2140, deg = True)
fft_max_phase_3000 = np.angle(fft_max_complex_value_3000, deg = True)

i = 1
#%%draw original signal
fig,axes = plt.subplots(2,2)
x = np.arange(0, N, 1)
xf = fftfreq(N, T)
axes[0,0].plot(xf, 1 / N * np.abs(fft_signal_fringe_2140[i]))
axes[0,0].set_title("FFT of signal in 2140 amplitude")
axes[0,0].set_xlabel("frequency")
axes[0,0].set_ylabel("amplitude")
axes[1,0].plot(x, signal_fringe_raw_2140[i])
axes[1,0].set_title("Singal of interferometry in 2140 amplitude")
axes[1,0].set_xlabel("# of sample points")
axes[1,0].set_ylabel("amplitude")
axes[0,1].plot(xf, 1 / N * np.abs(fft_signal_fringe_3000[i]))
axes[0,1].set_title("FFT of signal in 3000 amplitude")
axes[0,1].set_xlabel("frequency")
axes[0,1].set_ylabel("amplitude")
axes[1,1].plot(x, signal_fringe_raw_3000[i])
axes[1,1].set_title("Singal of interferometry in 3000 amplitude")
axes[1,1].set_xlabel("# of sample points")
axes[1,1].set_ylabel("amplitude")
fig.set_size_inches(30,10)
plt.show()
plt.clf()

# draw phase information
def plot_phase_information(y_phase, y_complex_value, name):
    
    col = ["#FC5A50", "#00FF00", "#FE420F", "#4B0082",
           "#0343DF", "#A9561E", "#C20078", "#FFD700",
           '#AAFF7B', "#13EAC9", '#000000', '#006400',
           '#ED0DD9', '#000080', '#FF6347', "#C0C0C0"]
    mrks = ["H", "*", "o", "v", 
            "^", "<", ">", "+",
            "d", "X", "h", "P",
            "x", "d", "p", "4"]
    fig,axes = plt.subplots(3,1)
    x = np.arange(0, 256, 1)    
    axes[0].scatter(x, y_phase, s = 800, c = col * 16, marker = '*' )
    axes[0].set_title("phase information in " + name + " amplitude", fontsize = 30)
    axes[0].set_xlabel("# of fringes", fontsize = 30)
    axes[0].set_ylabel("degree",  fontsize = 30)
    axes[0].tick_params(axis='both', labelsize = 30)
    axes[1].scatter(x, np.array(y_complex_value).real, s = 800, c = col * 16, marker = '*')
    axes[1].set_title("Real part of signal in " + name + " amplitude",  fontsize = 30)
    axes[1].set_xlabel("# of sample points",  fontsize = 30)
    axes[1].set_ylabel("amplitude",  fontsize = 30)
    axes[1].tick_params(axis='both', labelsize = 30)
    axes[2].scatter(x, np.array(y_complex_value).imag, s = 800, c = col * 16, marker = '*')
    axes[2].set_title("Imaginary part of signal in " + name + " amplitude",  fontsize = 30)
    axes[2].set_xlabel("# of sample points",  fontsize = 30)
    axes[2].set_ylabel("amplitude",  fontsize = 30)
    axes[2].tick_params(axis='both', labelsize = 30)
    fig.set_size_inches(110,50)
    plt.show()

plot_phase_information(fft_max_phase_2140, fft_max_complex_value_2140, '2140')
plot_phase_information(fft_max_phase_3000, fft_max_complex_value_3000, '3000')
mean_value = []
std_value = []
mean_value.append(np.mean(fft_max_phase_2140))
mean_value.append(np.mean(fft_max_phase_3000))
std_value.append(np.std(fft_max_phase_2140))
std_value.append(np.std(fft_max_phase_3000))
#%% interpolation
import pickle # for loading pickled test data
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import differential_evolution

# Double Lorentzian peak function
# bounds on parameters are set in generate_Initial_Parameters() below
def double_Lorentz(x, a, b, A, w, x_0, A1, w1, x_01):
    return a*x+b+(2*A/np.pi)*(w/(4*(x-x_0)**2 + w**2))+(2*A1/np.pi)*(w1/(4*(x-x_01)**2 + w1**2))

# Lorentzian peak function
def Lorentz(x, A, w, x_0):
    return (2*A/np.pi)*(w/(4*(x-x_0)**2 + w**2))

def derivative_Lorentz(x, A, w, x_0):
    return  16*A*w*(x_0-x)/np.pi/(4*(x-x_0)**2 + w**2)**2

# function for genetic algorithm to minimize (sum of squared error)
# bounds on parameters are set in generate_Initial_Parameters() below
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((yynew - derivative_Lorentz(xxnew, *parameterTuple)) ** 2)

def generate_Initial_Parameters(xData, yData):
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)
    
    parameterBounds = []
    #parameterBounds.append([-1.0, 1.0]) # parameter bounds for a
    #parameterBounds.append([maxY/-2.0, maxY/2.0]) # parameter bounds for b
    parameterBounds.append([0.0, maxY*100.0]) # parameter bounds for A
    parameterBounds.append([0.0, maxY/2.0]) # parameter bounds for w
    parameterBounds.append([minX, maxX]) # parameter bounds for x_0
    #parameterBounds.append([0.0, maxY*100.0]) # parameter bounds for A1
    #parameterBounds.append([0.0, maxY/2.0]) # parameter bounds for w1
    #parameterBounds.append([minX, maxX]) # parameter bounds for x_01

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    print(result.fun)
    return result.x

i = 1
xx = np.arange(0, 20, 1)
yy = np.array(fft_signal_fringe_2140[i])[np.array(fft_signal_fringe_2140[i])[1:].real.argmax() - 10 : np.array(fft_signal_fringe_2140[i])[1:].real.argmax() + 10].real
f = interpolate.interp1d(xx,yy)
xxnew = np.arange(0,19, 0.1)
yynew = f(xxnew)

# generate initial parameter values
initialParameters = generate_Initial_Parameters(xxnew,yynew)

# curve fit the test data
fittedParameters, pcov = curve_fit(derivative_Lorentz, xxnew, yynew, initialParameters, maxfev = 2000)

A, w, x_0 = fittedParameters
y_fit = derivative_Lorentz(xxnew, A, w, x_0)

plt.plot(xxnew,yynew,'o', xxnew, y_fit, '.')
plt.show()
#%%
fig,axes = plt.subplots(3,1)
x = np.arange(0, 99, 1)
start = 1
end = 100
axes[0].scatter(x, 1 / N * np.abs(fft_signal_fringe_2140[i][start:end]))
axes[1].scatter(x, np.array(fft_signal_fringe_2140[i])[start:end].real)
axes[2].scatter(x, np.array(fft_signal_fringe_2140[i])[start:end].imag)
fig.set_size_inches(20,10)
plt.show()

