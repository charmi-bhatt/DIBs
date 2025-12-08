#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:34:55 2024

@author: charmibhatt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:14:12 2024

@author: charmibhatt
"""




import pandas as pd
import numpy as np
import pandas as pd
import astropy.constants as const
from matplotlib import pyplot as plt

import timeit
import scipy.stats as ss
from scipy.signal import argrelextrema
from matplotlib import cm
import numpy as np
import pandas as pd
import astropy.constants as const

import timeit
import scipy as sp
import scipy.stats as ss
from lmfit import Model
import csv
import lmfit
from lmfit import minimize, Parameters, report_fit 
import uncertainties as unc
import uncertainties.umath as umath 
from pathlib import Path
from lmfit import Parameters     
from matplotlib.ticker import MaxNLocator
import seaborn as sns




sightlines = ['144217', '144470',  '145502', '147165', '149757', '179406', '184915'] 

B = 0.0024
delta_B = -0.043
zeta = -0.302
Ts = [86.7510061, 83.4888836, 86.7591371, 87.2017017, 89.1137989, 77.9659866, 82.9993103]
sigmas = [0.16072784, 0.14735417, 0.15423695, 0.15793055, 0.16483438, 0.16934683, 0.17905383]
origins = [0.02422319, -0.02411924, -0.0824351, -0.05932105, 0.05731874, 0.05975262, 0.00277156]


PR_sep = [1.527, 1.518, 1.507, 1.481, 1.386, 1.356, 1.330]

Alto_fits_results = np.array([PR_sep, Ts, sigmas, origins, sightlines]).T
sorted_indices = ( Alto_fits_results[:, 0].argsort())
Alto_fits_results = Alto_fits_results[sorted_indices].astype(float)

#Alto_fits_results = Alto_fits_results[:2]
plt.figure(figsize = (3.5, 10))

start = 0
spacing = 0.07
count = 7

print(Alto_fits_results[:,4])

offset = np.arange(start, start + spacing * count, spacing)


sorted_sightlines = sightlines [::-1]


spec_dir = Path("/home/charmibhatt/Science/Research/Research/Cami_2004_data/heliocentric/6614")
filename = 'hd{}_dib6614.txt'



for i, offset in enumerate(offset):  
    
    print(i)

    #Data --------------------------------------------------
    sightline  = sorted_sightlines[i]
    
    file = filename.format(sightline)
    Obs_data = pd.read_csv(spec_dir / file,
                            sep = '\s+')
    
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index]   
    
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    
    #print(Obs_data['Wavelength'] , Obs_data['Flux'])

    plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset, color = 'black', linewidth = 1) #, label = 'HD ' + str(sightline) , color = 'black')

    #---------------------------------------------------------


    #Individual model --------------------------------------------------
   
    filepath ='/home/charmibhatt/Science/DIBs/Figures/final/Fitting_at_Jmax_1000/Cami_2004_indiviual_fits/'
    individual_model_filename = f'{filepath}Individual_best_fit_model_Cami2004_{sorted_sightlines[i]}.0.csv'
   
    
    indi_model_data = pd.read_csv(individual_model_filename, sep = ',')
   
   
    plt.plot(indi_model_data.iloc[:,0], indi_model_data.iloc[:,1] - offset, color = 'blue', linewidth = 1) #, label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))
    
    # #---------------------------------------------------------




    #Altogether model --------------------------------------------------

    filepath ='/home/charmibhatt/Science/DIBs/Figures/final/Fitting_at_Jmax_1000/Cami_2004_data/'
    alto_model_filename = f'{filepath}Altogether_best_fit_model_{sorted_sightlines[i]}.csv'
   
    
    alto_model_data = pd.read_csv(alto_model_filename, sep = ',')
    T = Alto_fits_results[:,1][i]
    sigma = Alto_fits_results[:,2][i]
    sightline = int(Alto_fits_results[:,4][i])

    plt.plot(alto_model_data.iloc[:,0], alto_model_data.iloc[:,1] - offset, color = 'crimson', 
             label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma), linewidth = 1)
    
    
    #---------------------------------------------------------

    
    plt.xlabel('Wavenumber (cm$^{-1}$)', labelpad = 2, fontsize = 9)
    plt.ylabel('Normalized Intenisty - offset', labelpad = 2, fontsize = 9)
    plt.tick_params(axis='both', which='major', labelsize=9, direction = 'in')
    plt.title(f"B = {B:.4f} cm$^{{-1}}$  $\Delta$B = {delta_B:.2f}%  $\zeta$ = {zeta:.2f} cm$^{{-1}}$", fontsize=9)
    
    xy = (Obs_data['Wavelength'][110] , Obs_data['Flux'][120] - offset)
    
    
    

    # plt.annotate('HD ' + str(sightline), xy = xy , xytext = (3, alto_model_data.iloc[:,1][5] - offset + 0.007), fontsize = 9)
    
    
    # plt.annotate('T = {:.2f}'.format(T) + 'K', xy = xy , xytext = (-1.7, alto_model_data.iloc[:,1][5] - offset + 0.007), fontsize = 9)
    # plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + 'cm$^{-1}$', xy = xy , xytext = (-3.2, alto_model_data.iloc[:,1][5] - offset + 0.007), fontsize = 9)
   
    
       # Add the first annotation
    plt.annotate(
        'T = {:.2f}'.format(T) + 'K',
        (-3.2, alto_model_data.iloc[:,1][5] - offset + 0.015),  # Replace x with the desired x-coordinate
        xycoords='data',  # Assuming you're working with data coordinates
        fontsize=9,
        ha='center')
    
    plt.annotate(
        f"$\\sigma={sigma:.2f}\\,\\mathrm{{cm}}^{{-1}}$",       
        (-3.4, alto_model_data.iloc[:,1][5] - offset + 0.005),  # Replace x with the desired x-coordinate
       xycoords='data',  # Assuming you're working with data coordinates
       fontsize=9,
       ha='center')
    
    plt.annotate(
        'HD '+ str(sightline),       
        (3.2, alto_model_data.iloc[:,1][5] - offset + 0.009),  # Replace x with the desired x-coordinate
       xycoords='data',  # Assuming you're working with data coordinates
       fontsize=8,
       ha='center')
      
   
  
      
    plt.axvline(x = -0.70, color = 'gray', linestyle = 'dotted', alpha = 0.5)
    plt.axvline(x = -0.54, color = 'gray', linestyle = 'dotted', alpha = 0.5)
    plt.axvline(x = 0.74, color = 'gray', linestyle = 'dotted', alpha = 0.5)
    plt.axvline(x = 0.87, color = 'gray', linestyle = 'dotted', alpha = 0.5)
    
    #plt.xlim(-5,4.5)
    plt.xlim(4.5, -5)
   
   


    
# save_here = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Fitting_at_Jmax_1000/Altogther_at_maxJ_1000/"
# save_plot_as = save_here + "Alto_fits_EDIBLES_maxJ_1000.pdf"

plt.savefig('Alto_fits_Cami_2004_maxJ_1000.pdf', format = 'pdf', bbox_inches="tight")

# plt.show()
