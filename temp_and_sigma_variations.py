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



T = [57.6011016, 62.1023003, 63.8959281, 65.9367659, 61.9449444, 62.9214533, 64.0519755, 60.1955110, 60.5262702, 59.4114021, 62.0635621, 59.5961519]
sigma = [0.16338840, 0.17788725, 0.16570998, 0.16102132, 0.19401663, 0.14327215, 0.15361382, 0.17375430, 0.17862829, 0.19656620, 0.21245939, 0.16766571]

# Error values for T and sigma
T_err = [2.13258713, 2.14482434, 2.34077035, 4.11513493, 2.27864098, 3.25013480, 2.28768547, 1.94774866, 2.10871753, 1.95411968, 2.05591176, 2.66153533]
sigma_err = [0.00473269, 0.00362777, 0.00443001, 0.01152762, 0.00557462, 0.00864359, 0.00400098, 0.00209948, 0.00428447, 0.00322673, 0.00289166, 0.00804948]

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



