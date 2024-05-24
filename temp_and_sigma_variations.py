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



# B =       0.00248308 #+/- 8.7988e-05 (3.54%) (init = 0.0023)		
# delta_B =  -0.06843322 #+/- 0.00301885 (4.41%) (init = -0.0353)		
# zeta =  -0.31260631 #+/- 0.00953055 (3.05%) (init = -0.4197)		

# T = list(data['Temp'])
# T_err = list(data['T_error'])
# sigma = list(data['sigma'])
# sigma_err = list(data['sigma_error'])



#EDIBLES data maxJ 1000 correct resolution

T = [81.18658, 87.365539, 89.9437666, 92.4637866, 87.0832607, 88.3588874, 90.2125208, 84.7285498, 85.168429, 83.6203956, 87.5451781, 83.7653075]
T_err = [6.95157686, 7.27859264, 7.67871069, 10.531173, 7.35351354, 8.77693456, 7.59965330, 6.86202221, 7.04330108, 6.76466320, 7.14583117, 7.81175188]
sigma = [0.16286939, 0.17714702, 0.16487818, 0.16119119, 0.19473806, 0.14253173, 0.1532089, 0.1730355, 0.17778417, 0.19593083, 0.21270658, 0.16688067]
sigma_err = [0.00713124, 0.00531845, 0.00658410, 0.01696309, 0.00817924, 0.01227928, 0.00606854, 0.00328920, 0.00627899, 0.00473545, 0.00423906, 0.01185782]



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



