#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:32:19 2024

@author: charmibhatt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import griddata
from astropy import constants as cst

data_folderpath  = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/Butterfly_Nebula/H2_rotation_diagram/'

feature  = 'H2_6p908'

features = ['H2_6p108'] #, 'H2_6p908', 'H2_8p024', 'H2_9p664', 'H2_12p276', 'H2_17p035' ]
parameter = ' amps'

for feature in features: 
    filename = f'{data_folderpath}finding_integerated_intensity/intensity_integrals_{feature}.csv'
    data = pd.read_csv(filename, delimiter=',')
    
    RA = data['RA'] 
    Dec = data[' Dec']
    X = data [f'{parameter}']  # Replace 'scalefactor' with the correct column name
    
    # Define the 2D grid (adjust these ranges and resolutions as needed)
    ra_grid = np.linspace(min(RA), max(RA), 200) # num= int(max(RA) - min(RA)))
    dec_grid = np.linspace(min(Dec), max(Dec), 200) #, num= int(max(Dec) - min(Dec)))
    
    # Create a meshgrid for 2D interpolation
    ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid) #, indexing='ij')
    
    # Interpolate flux values onto the 2D grid
    all3_grid = griddata((RA, Dec), X, (ra_mesh, dec_mesh))#, method='linear')
    
    print(min(ra_grid), max(ra_grid))
    print(min(dec_grid), max(dec_grid))
    median = np.median(all3_grid)
    q1 = np.percentile(all3_grid, 25)
    q3 = np.percentile(all3_grid, 75)
    iqr = q3 - q1
    
    # Set vmin and vmax around the median adjusted by some fraction of the IQR
    vmin = median - 0.5 * iqr  # Adjust the multiplier as needed
    vmax = median + 0.5 * iqr  # Adjust the multiplier as needed

    
    plt.figure(figsize=(7,6))
    
    
    
    plt.imshow(all3_grid, cmap='viridis', origin='lower', 
                extent=(min(ra_grid), max(ra_grid), min(dec_grid), max(dec_grid)),
                aspect='auto', vmin=vmin, vmax=vmax)  # Adjust vmin and vmax as needed
    
    plt.colorbar(label=f'{parameter}')  # Add color bar with label
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.title(f'{feature} {parameter} map')
    
    
    
    
    print("Min :", np.nanmin(all3_grid))
    print("Max :", np.nanmax(all3_grid))
    print("Are there NaNs?:", np.isnan(all3_grid).any())


    # Define levels based on percentile to focus contour lines
    level_min = np.percentile(all3_grid[~np.isnan(all3_grid)], 5)  # 10th percentile
    level_max = np.percentile(all3_grid[~np.isnan(all3_grid)], 99)  # 90th percentile
    
    level_min = 0
    level_max = 3500
    levels = np.linspace(level_min, level_max, 50)  # Create 20 levels between the min and max


    
    plt.figure(figsize=(7,6))
    contour = plt.contourf(ra_mesh, dec_mesh, all3_grid, levels=levels, cmap='Reds')
    plt.colorbar(contour)
    plt.title('Contour Plot')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.show()
    
    
    
    



# plt.show()

# def wavelength_to_velocity(plotwave, rest_wave):

#     z = (plotwave - rest_wave)/rest_wave
#     velocity = z*cst.c.to("km/s").value

#     return velocity
