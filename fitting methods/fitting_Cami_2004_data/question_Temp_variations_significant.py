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


#correct resolution
T = [79.1337551, 86.3583384, 88.8431642, 95.5643204, 86.0534748, 80.1042636, 89.7854061, 81.9806263, 89.0445017, 81.2361554, 89.3558085, 80.9561796]
T_err = [4.03812055, 4.56756582, 5.19026988, 11.4404767, 5.16960670, 6.11313758, 5.08189642, 3.65708860, 5.29523978, 3.89533336, 4.85655344, 5.71090004]
sigma = [0.18055114, 0.19760700, 0.18423023, 0.18244127, 0.21244470, 0.15821120, 0.17367824, 0.19180354, 0.20377079, 0.21578868, 0.24029108, 0.18563300]
sigma_err = [0.00652811, 0.00578140, 0.00663215, 0.01480600, 0.00769938, 0.01058803, 0.00625162, 0.00443305, 0.00678099, 0.00530496, 0.00524340, 0.01050214]

PR_sep = [1.27, 1.34, 1.39, 1.38, 1.3, 1.36, 1.46, 1.33, 1.37, 1.27, 1.27, 1.29]
PR_sep_unc = [0.09, 0.05, 0.06, 0.06, 0.09, 0.05, 0.07, 0.03, 0.05, 0.03, 0.05, 0.07]



data = list(zip(sightlines, T, T_err, sigma, sigma_err, PR_sep, PR_sep_unc))
print(data)
# sort the data by increasing T value
data.sort(key=lambda x: x[1])

# unzip the sorted data
sightlines, T, T_err, sigma, sigma_err, PR_sep, PR_sep_unc = zip(*data)


x = np.linspace(1,12, 12)
# plt.errorbar(x = x, y = T, yerr = T_err, fmt='o', color='black',
#               ecolor='lightgray', elinewidth=3, capsize=0 )


# plt.xticks(x, sightlines, rotation = 55, size = 9)
# #plt.xticks([])  
# plt.ylabel(' Rotational Temperature (K)', labelpad =15 , size = 14)
# plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

# #plt.show()
# plt.savefig("Temp_scatter.pdf", format="pdf", bbox_inches="tight")
 
'''Sigma'''   
# plt.errorbar(x = x, y = sigma, yerr = sigma_err, fmt='o', color='black',
#               ecolor='lightgray', elinewidth=3, capsize=0 )


# plt.xticks(x, sightlines, rotation = 55, size = 9)
# #plt.xticks([])  
# plt.ylabel('Intrinsic linewidth (cm${-1}$)', labelpad =15 , size = 14)
# plt.xlabel('12 single-cloud sightlines', labelpad= 25, size = 14)

# #plt.show()
# plt.savefig("linewidth_scatter.pdf", format="pdf", bbox_inches="tight")


plt.errorbar(x = T, y = PR_sep, xerr = T_err, yerr = PR_sep_unc, fmt='o', color='black',
              ecolor='lightgray', elinewidth=3, capsize=0 )


#plt.xticks(x, sightlines, rotation = 55, size = 9)
#plt.xticks([])  
plt.ylabel('PR peak separation (cm${-1}$)', labelpad =15 , size = 14)
plt.xlabel('Rotational Temperature (K)', labelpad= 25, size = 14)

#plt.show()
plt.savefig("T_vs_PR_sep.pdf", format="pdf", bbox_inches="tight")


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


