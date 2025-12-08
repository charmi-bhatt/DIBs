#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:11:11 2024

@author: charmibhatt
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import matplotlib
from scipy.stats import pearsonr

module_path = os.path.abspath('/home/charmibhatt/Science/Spectools/')
if module_path not in sys.path:
    sys.path.insert(0, module_path)

import general_spectools


spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
    
filename = '6614_HD{}.txt'
sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']


model_path = '/home/charmibhatt/Science/DIBs/Figures/final/Fitting_at_Jmax_1000/Individual_at-maxJ_1000/'

def manual_integrate(wavelength, flux):
    # Ensure that wavelength and flux arrays are numpy arrays
    wavelength = np.array(wavelength)
    flux = np.array(flux)
    
    # Sort wavelength and flux by wavelength, just in case they are not sorted
    sorted_indices = np.argsort(wavelength)
    wavelength = wavelength[sorted_indices]
    flux = flux[sorted_indices]

    # Calculate the difference between each consecutive wavelength value
    delta_lambda = np.diff(wavelength)
    
    # Calculate the midpoints for the flux values between each wavelength interval
    midpoint_flux = (flux[:-1] + flux[1:]) / 2
    
    # Compute the integral as the sum of the areas of each small trapezoid
    integral = np.sum((1 - midpoint_flux) * delta_lambda)
    
    return integral



########################################
'''6614'''   
 ########################################

EW6614_list = []

depths6614 = [] 

stddev_6614 = []


for sightline in sightlines: 

    file = filename.format(sightline)
   
    
    Obs_data = pd.read_csv(spec_dir / file,
                            sep = ',')
    
    
    '''Getting std dev of the profile'''

    
    wavelength, flux = general_spectools.trim_data_based_on_x(Obs_data['Wavelength'], Obs_data['Flux'], 
                                                              x1 = 6611 , x2 = 6612)
    
    stddev_6614.append(np.std(flux))
    
    # plt.plot(Obs_data['Wavelength'], Obs_data['Flux'],)
    
    
   
    '''scaling model to be same depth as the observation'''
    min_obs_flux = np.min(Obs_data['Flux'])
       
    model_filename = f'{model_path}Individual_best_fit_model_{sightline}.csv'
    model = pd.read_csv(model_filename)
    
    model_wavelength = model.iloc[:,0] + 15120
    model_wavelength = (1/model_wavelength)*1e8 #wavenumber to wavelength
    
    
    model_wavelength = model_wavelength[::-1]
    model_flux = model.iloc[:,1] [::-1]
    
    scaled_model_flux = 1 - (1 - model_flux) * (1 - min_obs_flux) / 0.1  
    plt.plot(model_wavelength, scaled_model_flux, label = sightline)
    #plt.errorbar(model_wavelength, scaled_model_flux, yerr= np.std(flux) , label = sightline)
    plt.legend()
    
    #print(scaled_model_flux)
    
    
    '''Getting EW'''

    #Define the integration limits
    lower_limit = 6610
    upper_limit = 6616
    
    # Mask the data to include only the range from -5 to 5
    mask = (model_wavelength >= lower_limit) & (model_wavelength <= upper_limit)
    wavelength_integration_range = model_wavelength[mask]
    flux_integration_range = scaled_model_flux[mask]
    
    # print(flux_integration_range)
    
    # plt.plot(wavelength_integration_range, flux_integration_range)
    
    # Integrate using the Simpson's rule
    integrated_flux = simpson(y = 1 - flux_integration_range, x= wavelength_integration_range)
    
    EW6614_list.append(integrated_flux)
    
    
    
    
    '''Getting depth of the profile'''
    min_index = scaled_model_flux.idxmin()
    depth = scaled_model_flux[min_index]
    depth_wave = model_wavelength[min_index]
   
    depths6614.append(depth)
    # plt.scatter(depth_wave, depth)

# print(depths6614)


plt.xlim(6612, 6615)
plt.show()







 ########################################
'''CH+'''   
 ########################################

CHplus_filepath = Path("/home/charmibhatt/Science/DIBs/fitting_CHplus_and_K_lines/CHplus_coadded_data/")
filename = "HD {}_coadded_data_CHplus.txt"

EWCHplus_list = []
stddev_CHplus = []
depthsCHplus = []
for sightline in sightlines: 

    file = filename.format(sightline)
    
    Obs_data = pd.read_csv(CHplus_filepath / file,
                            sep = '\s+')
    
    CHplus_wavelength =  Obs_data.iloc[:,0]
    CHplus_flux = Obs_data.iloc[:,1] 
    
     
    '''Getting std dev of the profile'''

    
    wavelength, flux = general_spectools.trim_data_based_on_x(CHplus_wavelength, CHplus_flux, 
                                                              x1 = 4232.8 , x2 = 4233) 
    stddev_CHplus.append(np.std(flux))
    
    # CHplus_wavelength = (1 / CHplus_wavelength) * 1e8
    # min_index = np.argmin(CHplus_flux)
    #CHplus_wavelength =CHplus_wavelength - CHplus_wavelength[min_index]
    
    plt.plot(CHplus_wavelength, CHplus_flux, label = sightline)
    plt.legend()
    plt.xlim(4232, 4233)
    
    '''Getting EW'''
    
    #Define the integration limits
    lower_limit = 4232
    upper_limit = 4233
    
    # Mask the data to include only the range from -5 to 5
    mask = (CHplus_wavelength >= lower_limit) & (CHplus_wavelength <= upper_limit)
    wavelength_integration_range = CHplus_wavelength[mask]
    flux_integration_range = CHplus_flux[mask]
    
    # Integrate using the Simpson's rule
    integrated_flux = simpson(y = 1 - flux_integration_range, x= wavelength_integration_range)
    
    EWCHplus_list.append(integrated_flux)
    
    '''Getting depth of the profile'''
    min_index = CHplus_flux.idxmin()
    depth = CHplus_flux[min_index]
    depth_wave = CHplus_wavelength[min_index]
    depthsCHplus.append(depth)
    # plt.scatter(depth_wave, depth)
    
    
plt.title('CH+')
plt.show()



 ########################################
'''K'''  #KI line 
 ########################################

K_filepath = Path("/home/charmibhatt/Science/DIBs/fitting_CHplus_and_K_lines/K_line_coadded_data/")
filename = "HD {}_coadded_data_K_line.txt"

EWK_list = []
stddev_K = []
depthsK = []
for sightline in sightlines: 

    file = filename.format(sightline)
    
    Obs_data = pd.read_csv(K_filepath / file,
                            sep = '\s+')
    
    K_wavelength =  Obs_data.iloc[:,0]
    K_flux = Obs_data.iloc[:,1] 
    
     
    '''Getting std dev of the profile'''

    
    wavelength, flux1 = general_spectools.trim_data_based_on_x(K_wavelength, K_flux, 
                                                              x1 = 7698 , x2 = 7698.6) 
    
    
    
    # print(np.std(flux1))
    wavelength, flux2 = general_spectools.trim_data_based_on_x(K_wavelength, K_flux, 
                            
                                                               x1 = 7699.5 , x2 = 7699.9) 
    flux = np.append(flux1, flux2)
    
    # print(np.std(flux))
    # print('----------------')
    stddev_K.append(np.std(flux))
    
    # CHplus_wavelength = (1 / CHplus_wavelength) * 1e8
    # min_index = np.argmin(CHplus_flux)
    #CHplus_wavelength =CHplus_wavelength - CHplus_wavelength[min_index]
    
    plt.plot(K_wavelength, K_flux, label = sightline)
    plt.legend()
    plt.xlim(7698.6, 7699.3)
    
    '''Getting EW'''
    
    #Define the integration limits
    lower_limit = 7698.6
    upper_limit = 7699.3
    
    # Mask the data to include only the range from -5 to 5
    mask = (K_wavelength >= lower_limit) & (K_wavelength <= upper_limit)
    wavelength_integration_range = K_wavelength[mask]
    flux_integration_range = K_flux[mask]
    
    # Integrate using the Simpson's rule
    integrated_flux = simpson(y = 1 - flux_integration_range, x= wavelength_integration_range)
    
    # EWCHplus_list.append(integrated_flux)
    
    '''Getting depth of the profile'''
    min_index = K_flux.idxmin()
    depth = K_flux[min_index]
    depth_wave = K_wavelength[min_index]
    depthsK.append(depth)
    plt.scatter(depth_wave, depth)
    
    
plt.title('K')
plt.show()


# ##############################################
# '''Plotting EW'''
# ##############################################

#Create a color map with 12 unique colors
#colors = plt.cm.get_cmap('tab20b', len(sightlines))

colors = plt.get_cmap('Dark2', len(sightlines))


for i in range(len(sightlines)):
    plt.scatter(EW6614_list[i], EWCHplus_list[i], color=colors(i), label=f"HD {sightlines[i]}")
    

# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.xlabel('EW (6614) ($\AA$)')
plt.ylabel('EW (CH+) ($\AA$)')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()




# print(stddev_6614)

##############################################
'''Plotting depth of the profile'''
##############################################

for i in range(len(sightlines)):
    #plt.scatter(depths6614[i], depthsCHplus[i], color=colors(i), label=sightlines[i])


    plt.errorbar(depths6614[i], depthsCHplus[i], 
             xerr=stddev_6614[i], yerr=stddev_CHplus[i], 
             fmt='o', color=colors(i), label=sightlines[i], 
             markersize=5, ecolor='black')  # Set the error bars color to black

print(depths6614)
print(depthsCHplus)
print(stddev_6614)
print(stddev_CHplus)


# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


correlation_coefficient, p_value = pearsonr(depths6614, depthsCHplus)
plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.770, 0.96))



plt.xlabel('$\lambda$6614 central depth')
plt.ylabel('CH$^+$ central depth')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()


##############################################
'''Plotting depth of the profile - 6614 vs K'''
##############################################

for i in range(len(sightlines)):
    #plt.scatter(depths6614[i], depthsCHplus[i], color=colors(i), label=sightlines[i])


    plt.errorbar(depths6614[i], depthsK[i], 
             xerr=stddev_6614[i], yerr=stddev_K[i], 
             fmt='x', color=colors(i), label=sightlines[i], 
             markersize=5, ecolor='black')  # Set the error bars color to black

print(depthsK)
print(stddev_K)

#Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

correlation_coefficient, p_value = pearsonr(depths6614, depthsK)
plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.775, 0.8))


plt.xlabel('$\lambda$6614 central depth')
plt.ylabel('KI central depth')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()













# Calculate Pearson's correlation coefficient and the p-value
correlation_coefficient, p_value = pearsonr(depths6614, depthsCHplus)

print("Pearson Correlation Coefficient (r) between 6614 and CHplus:", correlation_coefficient)


# Calculate Pearson's correlation coefficient and the p-value
correlation_coefficient, p_value = pearsonr(depths6614, depthsK)

print("Pearson Correlation Coefficient (r) between 6614 and K:", correlation_coefficient)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    