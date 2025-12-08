#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:25:50 2024

@author: charmibhatt
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson


module_path = os.path.abspath('/home/charmibhatt/Science/Spectools/')
if module_path not in sys.path:
    sys.path.insert(0, module_path)

import general_spectools


spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
    
filename = '6614_HD{}.txt'
sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']


model_path = '/home/charmibhatt/Science/DIBs/Figures/final/Fitting_at_Jmax_1000/Individual_at-maxJ_1000/'


# for sightline in sightlines: 

#     file = filename.format(sightline)
   
    
#     Obs_data = pd.read_csv(spec_dir / file,
#                             sep = ',')
    
#     # Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
#     min_index = np.argmin(Obs_data['Flux'])
#     Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index]
    
#     plt.plot( Obs_data['Wavelength'], Obs_data['Flux'])
    
      
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.cm as cm

# Set the style and background color
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#101010')  # Set the figure background color
ax.set_facecolor('#101010')         # Set the axes background color

# Load the Pastel1 color map and generate colors
pastel_colors = cm.get_cmap('Pastel1', 12)


#######################################
'''Plotting models'''
#######################################

model_path = '/home/charmibhatt/Science/DIBs/Figures/final/Fitting_at_Jmax_1000/Individual_at-maxJ_1000/'

plt.figure(figsize=(10,8))
for sightline in sightlines:
    
    model_filename = f'{model_path}Individual_best_fit_model_{sightline}.csv'
    model = pd.read_csv(model_filename)
    
    model_wavelength = model.iloc[:,0] 
    model_flux = model.iloc[:,1] 

    
    
    min_index = np.argmin(model_flux)
    model_wavelength = model_wavelength - model_wavelength[min_index] + 15120
    
    
    #model_wavelength = model.iloc[:,0] + 15120
    model_wavelength = (1/model_wavelength)*1e8 #wavenumber to wavelength
    
    
    model_wavelength = model_wavelength[::-1]
    model_flux = model.iloc[:,1] [::-1]
    
    plt.plot(model_wavelength, model_flux)
    
# # Add labels and legend
plt.title('$\lambda$6614 DIB', fontsize = 22)
plt.xlabel('Wavelength ($\AA$)', fontsize = 20)
plt.ylabel('Flux', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

# Customize tick labels and reduce the number of ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))  # Fewer ticks on x-axis
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis


# Add minor ticks
plt.minorticks_on()
plt.tick_params(which='both', width=1, direction = 'in')
# plt.tick_params(which='major', length=7)
# plt.tick_params(which='minor', length=4, color='gray')
   
plt.xlim(6612.7, 6614.8)
plt.show()





#######################################
'''Plotting observed profiles'''
#######################################

# # Iterate over your sightlines and plot each spectrum with a color from Pastel1
# for idx, sightline in enumerate(sightlines):
#     file = filename.format(sightline)
#     Obs_data = pd.read_csv(spec_dir / file, sep=',')

#     # Adjust Wavelength
#     min_index = np.argmin(Obs_data['Flux'])
#     Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] +6613.5

#     # Normalize Flux to range from 1 to 0.9
#     min_flux = Obs_data['Flux'].min()
#     max_flux = Obs_data['Flux'].max()
#     Obs_data['Flux'] = 0.9 + 0.1 * (Obs_data['Flux'] - min_flux) / (max_flux - min_flux)

#     # Plot with colors from Pastel1 colormap
#     plt.plot(Obs_data['Wavelength'], Obs_data['Flux'], color=pastel_colors(idx / 12), label=f'Sightline {sightline}')

# # Add labels and legend
# plt.xlabel('Wavelength')
# plt.ylabel('Flux')


# plt.xlim(6612, 6615)
# plt.show()














