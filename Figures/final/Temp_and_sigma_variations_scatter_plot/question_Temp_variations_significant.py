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


#EDIBLES data maxJ 1000 correct resolution

T = [81.18658, 87.365539, 89.9437666, 92.4637866, 87.0832607, 88.3588874, 90.2125208, 84.7285498, 85.168429, 83.6203956, 87.5451781, 83.7653075]
T_err = [6.95157686, 7.27859264, 7.67871069, 10.531173, 7.35351354, 8.77693456, 7.59965330, 6.86202221, 7.04330108, 6.76466320, 7.14583117, 7.81175188]
sigma = [0.16286939, 0.17714702, 0.16487818, 0.16119119, 0.19473806, 0.14253173, 0.1532089, 0.1730355, 0.17778417, 0.19593083, 0.21270658, 0.16688067]
sigma_err = [0.00713124, 0.00531845, 0.00658410, 0.01696309, 0.00817924, 0.01227928, 0.00606854, 0.00328920, 0.00627899, 0.00473545, 0.00423906, 0.01185782]

PR_sep = [1.27, 1.34, 1.39, 1.38, 1.3, 1.36, 1.46, 1.33, 1.37, 1.27, 1.27, 1.29]
PR_sep_unc = [0.09, 0.05, 0.06, 0.06, 0.09, 0.05, 0.07, 0.03, 0.05, 0.03, 0.05, 0.07]



data = list(zip(sightlines, T, T_err, sigma, sigma_err, PR_sep, PR_sep_unc))
print(data)
# sort the data by increasing T value
data.sort(key=lambda x: x[1])

# unzip the sorted data
sightlines, T, T_err, sigma, sigma_err, PR_sep, PR_sep_unc = zip(*data)


x = np.linspace(1,12, 12)
plt.errorbar(x = x, y = T, yerr = T_err, fmt='o', color='black',
              ecolor='lightgray', elinewidth=3, capsize=0 )


plt.xticks(x, sightlines, rotation = 55, size = 9)
#plt.xticks([])  
plt.ylabel(' Rotational Temperature (K)', labelpad =15 , size = 14)
plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

#plt.show()
save_here = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Figures/final/Temp_and_sigma_variations_scatter_plot/'
plt.savefig(save_here + "Temp_scatter_maxJ_1000.pdf", format="pdf", bbox_inches="tight")
plt.show()
'''Sigma'''   
plt.errorbar(x = x, y = sigma, yerr = sigma_err, fmt='o', color='black',
              ecolor='lightgray', elinewidth=3, capsize=0 )


plt.xticks(x, sightlines, rotation = 55, size = 9)
#plt.xticks([])  
plt.ylabel('Intrinsic linewidth (cm$^{-1}$)', labelpad =15 , size = 14)
plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

#plt.show()
plt.savefig(save_here + "linewidth_scatter_maxJ_1000.pdf", format="pdf", bbox_inches="tight")

plt.show()
plt.errorbar(x = T, y = PR_sep, xerr = T_err, yerr = PR_sep_unc, fmt='o', color='black',
              ecolor='lightgray', elinewidth=3, capsize=0 )


#plt.xticks(x, sightlines, rotation = 55, size = 9)
#plt.xticks([])  
plt.ylabel('PR peak separation (cm$^{-1}$)', labelpad =15 , size = 14)
plt.xlabel('Rotational Temperature (K)', labelpad= 25, size = 14)

#plt.show()
plt.savefig(save_here + "T_vs_PR_sep_maxJ_1000.pdf", format="pdf", bbox_inches="tight")


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


