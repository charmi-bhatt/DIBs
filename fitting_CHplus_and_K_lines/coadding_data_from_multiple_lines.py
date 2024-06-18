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


''' 
Functions
'''

pythia = EdiblesOracle()


def load_EDIBLES_filenames(Wave, sightline):
    
    List = pythia.getFilteredObsList(object=[sightline] , OrdersOnly=True, Wave=Wave)

    #test = List.values.tolist()
    #filename = test[0]
    
    return List, sightline



def load_EDIBLES_spectrum(filename, make_plot):

    sp = EdiblesSpectrum(filename)
    sp.getSpectrum(np.min(sp.raw_wave)+1,np.max(sp.raw_wave)-1)

    print(sp.target)

    wave = sp.bary_wave

    flux = np.clip(sp.bary_flux, 0, None) 

    if make_plot == True:
          plt.plot(wave, flux)
          plt.show()

    return wave, flux, sp.target, sp.datetime



def define_continuum(start_wavelength, start_flux, end_wavelength, end_flux):
    slope = (end_flux - start_flux) / (end_wavelength - start_wavelength)
    intercept = start_flux - slope * start_wavelength
    
    equation = f"y = {slope:.2f} * x + {intercept:.2f}"
    #print(f">>> Defined continuum equation: {equation}")
    return lambda w: slope * w + intercept


def remove_continuum(wave, flux, plotrange, continuum_range_before, continuum_range_after, show_continuum_plot, show_normalized_spectra):

    bool_keep = (wave > plotrange[0]) & (wave < plotrange[1])
    plotwave = wave[bool_keep]
    plotflux = flux[bool_keep]

    
    #For continuum before the peak:
    bool_keep = (wave > continuum_range_before[0]) & (wave < continuum_range_before[1])
    continuum_wave_before = wave[bool_keep]
    conitnuum_flux_before = flux[bool_keep]                
    start_flux = np.mean(conitnuum_flux_before)
    start_wavelength = np.mean(continuum_range_before)

    #For continuum after the peak:
    bool_keep = (wave > continuum_range_after[0]) & (wave < continuum_range_after[1])
    conitnuum_flux_after= flux[bool_keep]
    continuum_wave_after = wave[bool_keep]
    end_flux = np.mean(conitnuum_flux_after)
    end_wavelength = np.mean(continuum_range_after)

    
    defined_continuum = define_continuum(start_wavelength , start_flux, end_wavelength, end_flux) 

    #plot: for seeing where the continuum is
    plt.figure(figsize=(12,8))
    if show_continuum_plot == True:
        plt.plot(plotwave, plotflux,  'r-', label='Data')
        plt.plot(plotwave, defined_continuum(plotwave))
    
        plt.plot(continuum_wave_before, conitnuum_flux_before)
        plt.plot(continuum_wave_after, conitnuum_flux_after)

        plt.scatter(start_wavelength, start_flux)
        plt.scatter(end_wavelength, end_flux)
        
        plt.xlim(plotrange)
        plt.title(f'{sightline}_{filename}')
        plt.show()
    #     # save_plot_as = workdir + f'{sightline}_K_continuum_defined_obs{i}.png'
    #     # plt.savefig(save_plot_as, format = 'png')

        
        
    flux_without_continuum = plotflux / defined_continuum(plotwave)
    #normflux = (flux_without_continuum - min(flux_without_continuum))/ (max(flux_without_continuum)- min(flux_without_continuum))
    
    #for seeing normlaized spectra 

    if show_normalized_spectra == True:
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

    return plotwave, flux_without_continuum
        


def fit_voigt(wave, flux, fitting_range, lambda0, f, gamma, v_resolution, n_step, N, b, v_rad): 

    wavegrid = np.array(wave)
    model=Model(voigt_absorption_line, independent_vars=['wavegrid'])
    model.set_param_hint('lambda0', value=lambda0, vary=False) #4232.548 #7698.9
    model.set_param_hint('f',  value= f, vary=False) #for K: 3.393e-1 for Ch+: 0.005450
    model.set_param_hint('gamma', value= gamma, vary=False) # for K: 3.8e7 for CH+ : 1.0e8
    model.set_param_hint('v_resolution', value= v_resolution, vary=False) #3.75, v_resolution = c/R = 3e5/8e4 for EDIBBLES
    model.set_param_hint('n_step', value=n_step, vary=False)
    model.set_param_hint('N', value=N, min = 0)
    model.set_param_hint('b', value=b)
    model.set_param_hint('v_rad', value=v_rad)
    params=model.make_params()
    params.pretty_print()
    print(' ')

    bool_keep = (wave > fitting_range[0]) & (wave < fitting_range[1])
    fitting_wave = wave[bool_keep]
    fitting_flux= flux[bool_keep]



    result = model.fit(fitting_flux,params,wavegrid=fitting_wave)
    print(result.fit_report())
    result.params.pretty_print()

    #plt.plot(fitting_wave, result.best_fit, color = 'red')


    return result

    

def v_rad_correction(wave, result):

    v_rad = result.params['v_rad'].value
    wave_corrected = wave / (1+v_rad/cst.c.to("km/s").value)

    return wave_corrected


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
    # plt.show()


    coadded_data = np.array([wavegrid, coadded_flux]).T
    print(coadded_data)

    return fig, coadded_data



##### Inputs ###########

feature = 'CHplus' #'CHplus' or 'K'

sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165', 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']
sightlines= ['HD 166937']

########################

if feature == 'CHplus':
    #For removing continuum
    plotrange=(4231.8, 4233.2)
    continuum_range_before = (4232.0, 4232.2)
    continuum_range_after = (4232.85, 4233)


    #For fitting voigt profile
    Wave = 4232
    fitting_range= (4231.8, 4232.9)
    lambda0 = 4232.548
    f= 0.005450
    gamma=1.0e8
    v_resolution=3.75
    n_step=25
    N=7e12
    b=10
    v_rad= 5

    #For making a grid to interpolate all observations on
    lambda_start= 4232
    lambda_end= 4233
    resolution=70000
    oversample=2

    save_CHplus_labels = True

if feature == 'K':
    #For removing continuum
    plotrange=(7698.1, 7699.9)
    continuum_range_before = (7698.1, 7698.5)
    continuum_range_after = (7699.4, 7699.9)

    #For fitting voigt profile
    Wave = 7698
    fitting_range = (7698.5, 7699.4)
    lambda0 = 7698.965
    f=3.393e-1
    gamma=3.8e7
    v_resolution=3.75
    n_step=25
    N=7e12
    b=10
    v_rad= 5

    #For making a grid to interpolate all observations on
    lambda_start=7698
    lambda_end= 7700
    resolution=107000
    oversample=2

    save_K_labels = True



for i, sightline in enumerate(sightlines):

    #Filtering EDIBLES data based on wavelength and sightline
    List, sightline = load_EDIBLES_filenames (Wave, sightline)

    all_wave = []
    all_flux = []

    if feature == 'CHplus':
        if sightline == 'HD 149757':
            List = List.drop(List.index[5])
            List = List.drop(List.index[3])
        if sightline == 'HD 166937':
            continuum_range_after = (4232.6, 4232.8)
    
        if sightline == 'HD 184915':
            List = List.drop(List.index[4])
        
    if feature == 'K':
        if sightline == 'HD 149757':
            List = List.drop(List.index[1])


    for i, filename in enumerate(List):  

        #getting data from filtered EDIBLES data set list
        wave, flux, target, datetime = load_EDIBLES_spectrum(filename, make_plot=False )

        #removing continuum based on defined points

            
        plotwave, flux_without_continuum = remove_continuum(wave, flux, plotrange, continuum_range_before, 
                                          continuum_range_after, show_continuum_plot = True, 
                                          show_normalized_spectra = False)


        #fitting voigt porfile to v_rad that can be used for wavlength corection
        result = fit_voigt(plotwave, flux_without_continuum, fitting_range, lambda0, f, gamma, v_resolution, n_step, N, b, v_rad)

        
        #performing radial velocity correction
        wave_corrected = v_rad_correction(plotwave, result)

        #putting all observations towards a given sightline together, which can then be used for co-adding
        all_wave.append(wave_corrected)
        all_flux.append(flux_without_continuum)


    wavegrid = make_grid(lambda_start= lambda_start, lambda_end= lambda_end, resolution=resolution, oversample=oversample)
    fig, coadded_data = coadd_data(wavegrid, all_wave, all_flux)


    savehere = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting_CHplus_and_K_lines/'
    if feature == 'CHplus':
        plt.savefig(savehere + f'CHplus_coadded_data/{sightline}_see_coadded_data_CHplus.png', format = 'png')
        np.savetxt(savehere + f'CHplus_coadded_data/{sightline}_coadded_data_CHplus.txt', coadded_data)

    if feature == 'K':
        plt.savefig(savehere + f'K_line_coadded_data/{sightline}_see_coadded_data_K_line.png', format = 'png')
        np.savetxt(savehere + f'K_line_coadded_data/{sightline}_coadded_data_K_line.txt', coadded_data)
                








