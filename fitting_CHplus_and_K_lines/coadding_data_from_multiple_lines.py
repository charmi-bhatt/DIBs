#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: charmibhatt

This code loads EDIBLES spectra for 12 single cloud sightlines and then co-adds
the data when we have multiple observations towards a sightline. 

That co-added data is then exported as a text file. 
"""

##################################

'''
Importing Modules
'''
#%%

from edibles import DATADIR
from edibles import PYTHONDIR
from edibles.utils.edibles_spectrum import EdiblesSpectrum
from edibles.utils.edibles_oracle import EdiblesOracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
import astropy.constants as cst
from io import BytesIO
from lmfit import Model
from lmfit.models import VoigtModel
from edibles.utils.voigt_profile import voigt_absorption_line
from edibles.utils.functions import make_grid

##################################
#%%
##################################

''' 
Functions
'''

pythia = EdiblesOracle()


def load_EDIBLES_filenames(Wave, sightline):
    
    List = pythia.getFilteredObsList(object=[sightline] , OrdersOnly=True, Wave=Wave)

    #test = List.values.tolist()
    #filename = test[0]
    
    return List, sightline



def load_EDIBLES_spectrum(filename, sightline, make_plot):
     
    # loads in the spectrum, gives you 
    #radial velocity corrected wavelength in barycentric frame 9here called wave_rest
     
    sp = EdiblesSpectrum(filename)
    sp.getSpectrum(np.min(sp.raw_wave)+1,np.max(sp.raw_wave)-1)

    print(sp.target)

    wave = sp.bary_wave
    #search for V_rad from the csv file provided
    row = V_rad_data.loc[V_rad_data['Sightline'] == sightline]
    print(object)
    v_rad = row['V_rad'].values[0]
    #print(v_rad)
    #radial velocity correction
    wave_rest = wave / (1+v_rad/cst.c.to("km/s").value)

    flux = np.clip(sp.bary_flux, 0, None) 

    if make_plot == True:
          plt.plot(wave_rest, flux)

    return wave_rest, flux, sp.target, sp.datetime



def define_continuum(start_wavelength, start_flux, end_wavelength, end_flux):
    slope = (end_flux - start_flux) / (end_wavelength - start_wavelength)
    intercept = start_flux - slope * start_wavelength
    
    equation = f"y = {slope:.2f} * x + {intercept:.2f}"
    #print(f">>> Defined continuum equation: {equation}")
    return lambda w: slope * w + intercept



def put_together_all_obs (List,  sightline, plotrange, continuum_range_before, continuum_range_after, make_plot):
        
    '''
    This function removes continuum and return a 2 arrays: all_wave i.e wavelength array of all
    observations towards the given sightlines and all_flux is a big array containing fluxes of 
    each observation as a new row
    
    
    '''
    all_wave = []
    all_flux = []
    for i, filename in enumerate(List):  

        
        wave_rest, flux, target, datetime = load_EDIBLES_spectrum(filename, sightline, make_plot=False )

        bool_keep = (wave_rest > plotrange[0]) & (wave_rest < plotrange[1])
        plotwave = wave_rest[bool_keep]
        plotflux = flux[bool_keep]

        
        #For continuum before the peak:
        bool_keep = (wave_rest > continuum_range_before[0]) & (wave_rest < continuum_range_before[1])
        conitnuum_flux_before = flux[bool_keep]                
        start_flux = np.mean(conitnuum_flux_before)
        start_wavelength = np.mean(continuum_range_before)

        #For continuum after the peak:
        bool_keep = (wave_rest > continuum_range_after[0]) & (wave_rest < continuum_range_after[1])
        conitnuum_flux_after= flux[bool_keep]
        end_flux = np.mean(conitnuum_flux_after)
        end_wavelength = np.mean(continuum_range_after)

        
        defined_continuum = define_continuum(start_wavelength , start_flux, end_wavelength, end_flux)                
        
        #plot: for seeing where the continuum is
        plt.figure(figsize=(12,8))
        if make_plot == True:
            plt.plot(plotwave, plotflux,  'r-', label='Data')
            plt.plot(plotwave, defined_continuum(plotwave))
        
            # plt.plot(conitnuum_wave_before, conitnuum_flux_before)
            # plt.plot(conitnuum_wave_after, conitnuum_flux_after)

            plt.scatter(start_wavelength, start_flux)
            plt.scatter(end_wavelength, end_flux)
            
            plt.xlim(plotrange)
            plt.title(f'{sightline}_{filename}')
            plt.show()
        #     # save_plot_as = workdir + f'{sightline}_K_continuum_defined_obs{i}.png'
        #     # plt.savefig(save_plot_as, format = 'png')

            
            
        flux_without_continuum = plotflux / defined_continuum(plotwave)
        normflux = (flux_without_continuum - min(flux_without_continuum))/ (max(flux_without_continuum) - min(flux_without_continuum))
        
        #for seeing normlaized spectra  
        plt.figure(figsize=(12,8))
        plt.plot(plotwave, normflux)
        plt.xlim(plotrange)
        plt.title(f'{sightline}_{filename}')
        plt.show()

        # save_plot_as = workdir + f'{sightline}_K_continuum_subtracted_and_norm__obs{i}.png'
        # plt.savefig(save_plot_as, format = 'png')


        #saving normalized data
        # filename_column = [filename] * len(plotwave)
        # combined_array = np.array([filename_column, plotwave, normflux]).T
        # # Save the array to a text file
        # np.savetxt(workdir + f'norm_spectra_K_{sightline}_obs{i}.txt', combined_array, fmt='%s', delimiter=' ')
        
        
        all_wave.append(plotwave)
        all_flux.append(normflux)

    
    return all_wave, all_flux




def coadd_data(wavegrid, all_wave, all_flux): 

    fig, ax = plt.subplots()
    total_flux = np.zeros(len(wavegrid))

    for index in range(len(all_flux)):
        interpolated_flux = np.interp(wavegrid, all_wave[index], all_flux[index])

        
        plt.plot(wavegrid, interpolated_flux, label = index)
        total_flux += interpolated_flux
    
    coadded_flux = total_flux / len(all_flux)

    
    ax.plot(wavegrid, coadded_flux, label = 'co-added', color  = 'black', linewidth = 2)
    ax.set_title(sightline)
    ax.legend()
    #plt.show()


    coadded_data = np.array([wavegrid, coadded_flux]).T
    print(coadded_data)

    return fig, coadded_data



##################################
'''
Importing Data and directory patghs
'''

vrad_filename = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/cloudVels_readable.csv"
V_rad_data = pd.read_csv(vrad_filename)

savehere = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting_CHplus_and_K_lines/'

##################################


''' Running all the required functions here to export and save that co-added data'''


sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165', 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']
sightlines= ['HD 185859']
################################################
#CH+ coadding data
###############################################

Wave = 4232
for i, sightline in enumerate(sightlines):
    List, sightline = load_EDIBLES_filenames (Wave, sightline)
    
    all_wave, all_flux = put_together_all_obs(List, sightline, 
                                            plotrange=(4232.1, 4233), 
                                            continuum_range_before = (4232.1, 4232.3),
                                            continuum_range_after = (4232.75, 4233 ),
                                            make_plot= True)
    
    #manually removeing the (very) bad data of some observations towards HD 149757
    #keeping only the good ones (visually distinguishable)
    if sightline == 'HD 149757':
        del all_wave[5]
        del all_wave[3]

        del all_flux[5]
        del all_flux[3]

    wavegrid = make_grid(lambda_start= 4232, lambda_end= 4233, resolution=70000, oversample=2)
    fig, coadded_data = coadd_data(wavegrid, all_wave, all_flux)

    #plt.show()
    #saving the plot showing coadded data and the coadded spectrum as a text file
    plt.savefig(savehere + f'CHplus_coadded_data/{sightline}_see_coadded_data_CHplus.png', format = 'png')
    np.savetxt(savehere + f'CHplus_coadded_data/{sightline}_coadded_data_CHplus.txt', coadded_data)



################################################
#K coadding data
###############################################



# Wave = 7698
# for i, sightline in enumerate(sightlines):
#     List, sightline = load_EDIBLES_filenames (Wave, sightline)
    
#     all_wave, all_flux = put_together_all_obs(List, sightline, 
#                                             plotrange=(7698.1, 7699.9), 
#                                             continuum_range_before = (7698.1, 7698.5),
#                                             continuum_range_after = (7699.4, 7699.9),
#                                             make_plot= False)
    
#     # #manually removeing the (very) bad data of some observations towards HD 149757
#     # #keeping only the good ones (visually distinguishable)
#     if sightline == 'HD 149757':
#         del all_wave[1]
#         del all_flux[1]

#     wavegrid = make_grid(lambda_start= 7698, lambda_end= 7700, resolution=107000, oversample=2)
#     fig, coadded_data = coadd_data(wavegrid, all_wave, all_flux)

#     plt.show()
   
#   saving the plot showing coadded data and the coadded spectrum as a text file
    # plt.savefig(savehere + f'K_line_coadded_data/{sightline}_see_coadded_data_K_line.png', format = 'png')
    # np.savetxt(savehere + f'K_line_coadded_data/{sightline}_coadded_data_K_line.txt', coadded_data)
               

