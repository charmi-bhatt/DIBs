#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:22:39 2023

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
spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/Cami_2004_data/heliocentric/6614/")


sightlines = ['144217', '184915' , '144470', '145502', '147165', '149757', '179406', '184915']


def curve_to_fit_wavenos(sightline): 
        
        file = 'hd{}_dib6614.txt'.format(sightline)
        Obs_data = pd.read_csv(spec_dir / file,
                               delim_whitespace=(True))
        
        Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
        Obs_data = Obs_data.iloc[::-1].reset_index(
            drop=True)  # making it ascending order as we transformed wavelength into wavenumbers

        # shifting to zero and scaling flux between 0.9 and 1
        min_index = np.argmin(Obs_data['Flux'])
        Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] 
        Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
        #plt.plot(Obs_data['Wavelength'], Obs_data['Flux'] )
        
        # removing red wing
        # Obs_data_trp = Obs_data [(Obs_data['Wavelength'] >= -1) & (Obs_data['Wavelength']<= 1.2)]
        Obs_data_trp = Obs_data[(Obs_data['Flux'] <= 0.95)]  # trp = triple peak structure

        # making data evenly spaced
        x_equal_spacing = np.linspace(min(Obs_data_trp['Wavelength']), max(Obs_data_trp['Wavelength']), 100)
        plt.xlim(-2, 2)
        return Obs_data, x_equal_spacing
 
plt.figure(figsize = (15,8))

# for i, s in enumerate(sightlines): 
#     Obs_data, x_equal_spacing = curve_to_fit_wavenos(s)
#     plt.plot(Obs_data['Wavelength'], Obs_data['Flux'] - i*0.01, label = s)

    
Obs_data, x_equal_spacing = curve_to_fit_wavenos('144217')
plt.plot(Obs_data['Wavelength'], Obs_data['Flux'], label = '144217')


Obs_data, x_equal_spacing = curve_to_fit_wavenos('184915')
plt.plot(Obs_data['Wavelength']+0.06, Obs_data['Flux'] , label = '184915' )

Obs_data, x_equal_spacing = curve_to_fit_wavenos('144470')
plt.plot(Obs_data['Wavelength']+0.04, Obs_data['Flux'] , label = '144470' )

Obs_data, x_equal_spacing = curve_to_fit_wavenos('145502')
plt.plot(Obs_data['Wavelength']+0.1, Obs_data['Flux'] , label = '145502')

Obs_data, x_equal_spacing = curve_to_fit_wavenos('179406')
plt.plot(Obs_data['Wavelength'], Obs_data['Flux']  , label = '179406')

Obs_data, x_equal_spacing = curve_to_fit_wavenos('147165')
plt.plot(Obs_data['Wavelength']+0.08, Obs_data['Flux']  , label = '147165')

Obs_data, x_equal_spacing = curve_to_fit_wavenos('179406')
plt.plot(Obs_data['Wavelength']+0.03, Obs_data['Flux']  , label = '179406')

plt.title('Does width of Q branch peak changes in data from Cami et al. (2004)?')
plt.legend()