#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:54:55 2023

@author: charmibhatt
"""

import numpy as np
import pandas as pd
import astropy.constants as const
import matplotlib.pyplot as plt
import timeit
# import scipy
# import scipy.stats as ss
from lmfit import Model
import csv
# import lmfit
import numba as nb
from pathlib import Path
from lmfit import Parameters

sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165' , 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']

T = [84.45, 95.03, 98.26, 115.26, 97.56, 84.71, 98.41, 88.62, 95.90, 87.57, 102.36, 86.49]
T_err = [6.82, 8.72, 9.76, 22.16, 10.30, 9.688, 9.39, 6.69, 9.31, 7.14, 10.59, 9.59]
#less uncertainities 
T = [85.1980671, 95.9170194, 97.7526553, 117.357633, 99.4837311, 86.7969455, 98.8252733, 89.3671108, 97.6353221, 88.2069174, 104.179082, 86.5815310]
T_err = [3.53219285, 4.47563910, 4.81673463, 11.8238856, 5.46520419, 5.18071752, 4.73214286, 3.39864225, 4.90728158, 3.65064980, 5.58826545, 4.90358469]



sigma = [0.185, 0.202, 0.1905, 0.1941, 0.2193, 0.1630, 0.1796, 0.1972, 0.2078, 0.220, 0.2489, 0.1900]
sigma_err= [0.008, 0.0076, 0.0085, 0.0181, 0.0095, 0.0123, 0.008, 0.006, 0.0084, 0.0069, 0.0071, 0.0123]
#less uncertainities 
sigma = [0.18528460, 0.20311258, 0.18971569, 0.19368843, 0.22016402, 0.16365104, 0.17955888, 0.19714333, 0.20891494, 0.22057438, 0.24994232, 0.18989476]
sigma_err = [0.00394647, 0.00372946, 0.00410970, 0.00890177, 0.00468315, 0.00599772, 0.00390652, 0.00294027, 0.00412508, 0.00335768, 0.00351795, 0.00600206]

#with 150 datapoints
T = [84.8194157, 95.5401280, 97.2287829, 116.540106, 99.2097303, 86.6632524, 98.4034623, 89.0236501, 97.3496743, 87.8768841, 103.793851, 86.2283890]
T_err = [2.80977672, 3.58171433, 3.84510665, 9.47334493, 4.39868936, 4.20614639, 3.78239428, 2.71146630, 3.94014863, 2.92062984, 4.46897005, 3.95544354]
sigma = [0.18496609, 0.20292189, 0.18936606, 0.19319891, 0.22004922, 0.16349391, 0.17929137, 0.19690720, 0.20881159, 0.22038457, 0.24982774, 0.18963846]
sigma_err = [0.00313293, 0.00300996, 0.00331638, 0.00722704, 0.00379580, 0.00486778, 0.00315146, 0.00236154, 0.00333481, 0.00270967, 0.00283842, 0.00486903]



# sigmas = np.array([sigma, sigma_err]).transpose()
# # print(sigmas)
# sigmas = sigmas[sigmas[:, 0].argsort()]
# # print(sigmas)

# Ts = np.array([T, T_err, sightlines]).transpose()
# print(Ts)
# # print(sigmas)
# Ts = Ts[Ts[:, 0].argsort()]
# print(Ts)

data = list(zip(sightlines, T, T_err, sigma, sigma_err))
print(data)
# sort the data by increasing T value
data.sort(key=lambda x: x[1])

# unzip the sorted data
sightlines, T, T_err, sigma, sigma_err = zip(*data)


x = np.linspace(1,12, 12)
plt.errorbar(x = x, y = T, yerr = T_err, fmt='o', color='black',
              ecolor='lightgray', elinewidth=3, capsize=0 )


plt.xticks(x, sightlines, rotation = 55, size = 9)
#plt.xticks([])  
plt.ylabel(' Rotational Temperature (K)', labelpad =15 , size = 14)
plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

plt.show()

plt.errorbar(x = x, y = sigma, yerr = sigma_err, fmt='o', color='black',
              ecolor='lightgray', elinewidth=3, capsize=0 )


plt.xticks(x, sightlines, rotation = 55, size = 9)
#plt.xticks([])  
plt.ylabel('Intrinsic linewidth (cm${-1}$)', labelpad =15 , size = 14)
plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

plt.show()
#plt.scatter(Ts[:,0], sigmas[:,0])
plt.show()

'''As subplots'''
# import matplotlib.pyplot as plt

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),  facecolor='lightyellow')

# # Top subplot - Temperature
# ax1.errorbar(x=x, y=Ts[:, 0], yerr=Ts[:, 1], fmt='o', color='black',
#              ecolor='lightgray', elinewidth=3, capsize=0)
# ax1.set_ylabel('Rotational Temperature (K)', labelpad=15, size=14)
# #ax1.set_xlabel('12 single-cloud sightlines', labelpad=25, size=14)
# ax1.set_xticks([])
# ax1.set_xticklabels([])

# # Bottom subplot - Line width
# ax2.errorbar(x=x, y=sigmas[:, 0], yerr=sigmas[:, 1], fmt='o', color='black',
#              ecolor='lightgray', elinewidth=3, capsize=0)
# ax2.set_ylabel('Intrinsic linewidth (cm$^{-1}$)', labelpad=15, size=14)
# ax2.set_xlabel('12 single-cloud sightlines', labelpad=25, size=14)
# ax2.set_xticks(x)
# ax2.set_xticklabels(sightlines, rotation=55, size=9)

# plt.tight_layout()
# plt.show()

'''Cami 2004'''

# T = [95.67, 93.59, 99.89, 105.50, 96.33, 82.07, 91.02]
# T_err = [13.09, 10.72, 13.49, 15.36, 13.46, 8.135, 11.32]
# sightlines = ['144217', '144470', '145502', '147165', '149757', '179406', '184915']



# x = np.linspace(1,7,7)
# plt.errorbar(x = x, y = T, yerr = T_err, fmt='o', color='black',
#              ecolor='lightgray', elinewidth=3, capsize=0 )


# plt.xticks(x, sightlines, rotation = 75)
# plt.ylabel('Temperature')
# plt.title('Cami et al 2004 data')


