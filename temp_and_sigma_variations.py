#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:18:35 2023

@author: charmibhatt
"""

import pandas as pd
import numpy as np
import pandas as pd
import astropy.constants as const
from matplotlib import pyplot as plt

import timeit
from pathlib import Path
from lmfit import Parameters      

data = pd.read_excel("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting methods/master_fitting_results_in_ a_table copy.xlsx", header = 0)

x = np.linspace(1,12, 12)



B =       0.00248308 #+/- 8.7988e-05 (3.54%) (init = 0.0023)		
delta_B =  -0.06843322 #+/- 0.00301885 (4.41%) (init = -0.0353)		
zeta =  -0.31260631 #+/- 0.00953055 (3.05%) (init = -0.4197)		

T = list(data['Temp'])
T_err = list(data['T_error'])
sigma = list(data['sigma'])
sigma_err = list(data['sigma_error'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),  facecolor='lightyellow')

# Top subplot - Temperature
ax1.errorbar(x=x, y=T, yerr=T_err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
ax1.set_ylabel('Rotational Temperature (K)', labelpad=15, size=14)
#ax1.set_xlabel('12 single-cloud sightlines', labelpad=25, size=14)
ax1.set_xticks([])
ax1.set_xticklabels([])

# Bottom subplot - Line width
ax2.errorbar(x=x, y=sigma, yerr=sigma_err, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
ax2.set_ylabel('Intrinsic linewidth (cm$^{-1}$)', labelpad=15, size=14)
ax2.set_xlabel('12 single-cloud sightlines', labelpad=25, size=14)
ax2.set_xticks(x)
ax2.set_xticklabels(data['Sightline'], rotation=55, size=9)

plt.tight_layout()
plt.show()



