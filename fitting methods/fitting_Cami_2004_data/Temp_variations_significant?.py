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


T = [84.45, 95.03, 98.26, 115.26, 97.56, 84.71, 98.41, 88.62, 95.90, 87.57, 102.36, 86.49]
T_err = [6.82, 8.72, 9.76, 22.16, 10.30, 9.688, 9.39, 6.69, 9.31, 7.14, 10.59, 9.59]
sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165' , 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']

sigma = [0.185, 0.202, 0.1905, 0.1941, 0.2193, 0.1630, 0.1796, 0.1972, 0.2078, 0.220, 0.2489, 0.1900]
sigma_err= [0.008, 0.0076, 0.0085, 0.0181, 0.0095, 0.0123, 0.008, 0.006, 0.0084, 0.0069, 0.0071, 0.0123]

sigmas = np.array([sigma, sigma_err]).transpose()
print(sigmas)
sigmas = sigmas[sigmas[:, 0].argsort()]
print(sigmas)

Ts = np.array([T, T_err]).transpose()
print(sigmas)
Ts = Ts[Ts[:, 0].argsort()]
print(sigmas)


x = np.linspace(1,12, 12)
# plt.errorbar(x = x, y = Ts[:,0], yerr = Ts[:,1], fmt='o', color='black',
#               ecolor='lightgray', elinewidth=3, capsize=0 )


# plt.xticks(x, sightlines, rotation = 55, size = 9)
# plt.xticks([])  
# plt.ylabel(' Rotational Temperature (K)', labelpad =15 , size = 14)
# plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

# plt.show()

# plt.errorbar(x = x, y = sigmas[:,0], yerr = sigmas[:,1], fmt='o', color='black',
#              ecolor='lightgray', elinewidth=3, capsize=0 )


# #plt.xticks(x, sightlines, rotation = 55, size = 9)
# plt.xticks([])  
# plt.ylabel('Intrinsic linewidth (cm${-1}$)', labelpad =15 , size = 14)
# plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

# plt.show()
plt.scatter(Ts[:,0], sigmas[:,0])
plt.show()
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),  facecolor='lightyellow')

# Top subplot - Temperature
ax1.errorbar(x=x, y=Ts[:, 0], yerr=Ts[:, 1], fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
ax1.set_ylabel('Rotational Temperature (K)', labelpad=15, size=14)
#ax1.set_xlabel('12 single-cloud sightlines', labelpad=25, size=14)
ax1.set_xticks([])
ax1.set_xticklabels([])

# Bottom subplot - Line width
ax2.errorbar(x=x, y=sigmas[:, 0], yerr=sigmas[:, 1], fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
ax2.set_ylabel('Intrinsic linewidth (cm$^{-1}$)', labelpad=15, size=14)
ax2.set_xlabel('12 single-cloud sightlines', labelpad=25, size=14)
ax2.set_xticks(x)
ax2.set_xticklabels(sightlines, rotation=55, size=9)

plt.tight_layout()
plt.show()

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


