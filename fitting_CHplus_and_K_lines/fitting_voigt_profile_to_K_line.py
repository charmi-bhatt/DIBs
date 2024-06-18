''' This code take the coadded data and fits voigt profile to K line line at 7698A'''



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

'''
Importing CHplus co-addded data and directory paths
'''

coadded_data_path = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting_CHplus_and_K_lines/K_line_coadded_data/"

savehere = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting_CHplus_and_K_lines/K_line_coadded_fits/'

##################################

'''
Functions
'''


def define_continuum(start_wavelength, start_flux, end_wavelength, end_flux):
    slope = (end_flux - start_flux) / (end_wavelength - start_wavelength)
    intercept = start_flux - slope * start_wavelength
    
    equation = f"y = {slope:.2f} * x + {intercept:.2f}"
    print(f">>> Defined continuum equation: {equation}")
    return lambda w: slope * w + intercept


def remove_continuum(plotwave, plotflux, continuum_range_before, continuum_range_after):
   
    
    #For continuum before the peak:
    
    bool_keep = (plotwave > continuum_range_before[0]) & (plotwave < continuum_range_before[1])
    continuum_wave_before = plotwave[bool_keep]
    continuum_flux_before = plotflux[bool_keep]                
    start_flux = np.mean(continuum_flux_before)
    start_wavelength = np.mean(continuum_range_before)

    #For continuum after the peak:

    bool_keep = (plotwave > continuum_range_after[0]) & (plotwave < continuum_range_after[1])
    continuum_wave_after = plotwave[bool_keep]
    continuum_flux_after= plotflux[bool_keep]
    end_flux = np.mean(continuum_flux_after)
    end_wavelength = np.mean(continuum_range_after)

    
    defined_continuum = define_continuum(start_wavelength , start_flux, end_wavelength, end_flux)
    flux_without_continuum = plotflux / defined_continuum(plotwave)
    normflux = (flux_without_continuum - min(flux_without_continuum))/ (max(flux_without_continuum) - min(flux_without_continuum))

    #To see the details of continuum removal
    # plt.figure()     
    # plt.plot(plotwave, plotflux,  'r-', label='Data')
    # plt.plot(plotwave, defined_continuum(plotwave))

    # plt.plot(continuum_wave_before, continuum_flux_before, color = 'green')
    # plt.plot(continuum_wave_after, continuum_flux_after, color = 'orange')

    # plt.scatter(start_wavelength, start_flux)
    # plt.scatter(end_wavelength, end_flux)
    # plt.show()

    #for seeing normlaized spectra  
    # plt.figure(figsize=(12,8))
    # plt.plot(plotwave, normflux)



    return normflux

def fit_voigt_to_K_line(wave, flux, fitting_range):

    wavegrid = np.array(wave)
    model=Model(voigt_absorption_line, independent_vars=['wavegrid'])
    model.set_param_hint('lambda0', value=7698.965, vary=False) #CH+: 4232.548 #K: 7698.965
    model.set_param_hint('f',  value= 3.393e-1, vary=False) #for K: 3.393e-1 for Ch+: 0.005450
    model.set_param_hint('gamma', value=3.8e7, vary=False) # for K: 3.8e7 for CH+ : 1.0e8
    model.set_param_hint('v_resolution', value=3.75, vary=False) #3.75, v_resolution = c/R = 3e5/8e4 for EDIBBLES
    model.set_param_hint('n_step', value=25, vary=False)
    model.set_param_hint('N', value=7e12)
    model.set_param_hint('b', value=3.36)
    model.set_param_hint('v_rad', value=0.6)
    params=model.make_params()
    params.pretty_print()
    print(' ')

    #  Args:
    # wavegrid (float64): Wavelength grid (in Angstrom) on which the final result is desired.
    # lambda0 (float64): Central (rest) wavelength for the absorption line, in Angstrom.
    # b (float64): The b parameter (Gaussian width), in km/s.
    # N (float64): The column density (in cm^{-2})
    # f (float64): The oscillator strength (dimensionless)
    # gamma (float64): Lorentzian gamma (=HWHM) component
    # v_rad (float64): Radial velocity of absorption line (in km/s)
    # v_resolution (float64): Instrument resolution in velocity space (in km/s)
    # n_step (int): no. of point per FWHM length, governing sampling rate and efficiency
    # debug (bool): If True, info on the calculation will be displayed


    bool_keep = (wave > fitting_range[0]) & (wave < fitting_range[1])
    fitting_wave = wave[bool_keep]
    fitting_flux= (flux[bool_keep])

    


    result = model.fit(fitting_flux,params,wavegrid=fitting_wave)
    print(result.fit_report())
    result.params.pretty_print()

    result_dic = {}
    result_dic['Sightline'] = f'{sightline}'  
    result_dic['Red Chi-Squared'] = f'{result.redchi}'
    
    # Update result_dic with parameter values
    for name, param in result.params.items():
        result_dic[name] = param.value
        result_dic[f'{name} uncertainty'] = param.stderr
    
    # Add fit_report to result_dic
    result_dic['fit_report'] = f'{result.fit_report()}'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(wave, flux, 'b-', label='Data')
    ax.plot(fitting_wave, fitting_flux, color='green', label='This goes into fitting')
    ax.plot(fitting_wave, result.best_fit, 'r-', label='Best Fit')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlabel('Plotwave')
    ax.set_ylabel('Continuum subtracted flux')
    plt.subplots_adjust(right=0.5)
    plt.figtext(0.55, 0.02, f'Sightline : {sightline} \n \n' + result.fit_report(),
                wrap=True, horizontalalignment='left', fontsize=10)


    return fig, result_dic




##################################

sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165', 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']

fit_results_list = []
for sightline in sightlines: 

    coadded_data = pd.read_csv(coadded_data_path + f'{sightline}_coadded_data_K_line.txt', sep = ' ')

    plotwave = coadded_data.iloc[:,0]
    plotflux = coadded_data.iloc[:,1]

    # normflux = remove_continuum(plotwave, plotflux, continuum_range_before = (4232.1, 4232.4),
    #                                                continuum_range_after = (4232.75, 4233))

    fig, result_dic = fit_voigt_to_K_line(plotwave, plotflux,fitting_range = (7698.5, 7699.4))

    plt.savefig(savehere + f'{sightline}_fit_result_k_line_coadded.png' , format = 'png')
        
        
    fit_results_list.append(result_dic) 

results_dataframe = pd.DataFrame(fit_results_list)
save_results_into_csv_as = savehere + 'K_line_coadded_voigt_fitting_results.csv'
results_dataframe.to_csv(save_results_into_csv_as)

