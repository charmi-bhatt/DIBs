#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:07:02 2023

@author: charmibhatt
"""

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def make_grid(lambda_start, lambda_end, resolution=None, oversample=None):

    # check keywords
    if oversample is None:
        oversample = 40.0
    if resolution is None:
        resolution = 1500.0

    lambda_start = np.float64(lambda_start)
    lambda_end = np.float64(lambda_end)

    # produce grid
    R = resolution * oversample
    
    #print('R = ' , R)
    n_points = (
        round(
            (np.log(lambda_end / lambda_start)) / (np.log(-(1 + 2 * R) / (1 - 2 * R)))
        )
        + 1
    )
    #print('n_points = ' , n_points)
    f = -(1 + 2 * R) / (1 - 2 * R)
    
    #print('f = ', f)
    factor = f ** np.arange(n_points)
    #print('factor = ' , factor)
    wave = np.full(int(n_points), lambda_start, dtype=np.float)
    #print('wave = ' , wave)
    grid = wave * factor
    #print('grid = ', grid)
    return grid

lambda_start = 6613.5453435174495 #-1.134 #
lambda_end =  6614.490245405555 #1.039609311008462 #

spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Data/Heather's_data")
filename = '6614_HD{}.txt'
sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']

#sightlines = ['166937', '185418', '184915']

# grid  = make_grid(lambda_start, lambda_end, resolution=107000, oversample=2)
# print(grid.shape)

common_grid_for_all = make_grid(lambda_start, lambda_end, resolution=107000, oversample=2)

common_grid_for_all = (1 / common_grid_for_all) * 1e8
common_grid_for_all = common_grid_for_all - 15119.4
common_grid_for_all = common_grid_for_all[::-1]

print(common_grid_for_all)



def obs_curve_to_fit(sightline): 
    
        '''This function reads in data, removes wings and provides just 
        the triple peak for fitting and calculates std dev for each sightline '''
    
        file = filename.format(sightline)
        # Obs_data = pd.read_csv(spec_dir / file,
        #                         delim_whitespace=(True))
        
        Obs_data = pd.read_csv(spec_dir / file,
                                sep = ',')

        '''interpolating over common grid'''
        
        Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
        Obs_data = Obs_data.iloc[::-1].reset_index(
            drop=True)
        # shifting to 6614 and scaling flux between 0.9 and 1
        min_index = np.argmin(Obs_data['Flux'])
        Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] #+ 6614
        
        Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
        
        plt.plot(Obs_data['Wavelength'], Obs_data['Flux'])
        
        Obs_y_data_to_fit  = np.interp(common_grid_for_all, Obs_data['Wavelength'], Obs_data['Flux'])
        
        print(Obs_y_data_to_fit)
        
       

        Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >= -9) & (Obs_data['Wavelength']<= -4)]
        std_dev = np.std(Obs_data_continuum['Flux'])
        
       
        return Obs_data, Obs_y_data_to_fit, std_dev
    


for sightline in sightlines: 
    Obs_data, Obs_y_data_to_fit, std_dev = obs_curve_to_fit(sightline)
    plt.plot(common_grid_for_all, Obs_y_data_to_fit)
    # print(std_dev)
    #plt.legend()
    plt.show()
    