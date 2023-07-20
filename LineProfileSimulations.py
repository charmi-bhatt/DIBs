#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:22:21 2023

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
import numba as nb
from pathlib import Path
from lmfit import Parameters      



def allowed_perperndicular_transitions(Jmax):
    
    '''Take in Jmax and calculates the all allowed transitions 
    based on selection rules for perpendicular transitions. 
    (Jmax = Kmax)'''

    '''P Branch'''
    P_branch_Js = list(range(1, Jmax + 1))
    all_P_branch_Js = [j for j in P_branch_Js for _ in range(j)]
    P_branch_Jprimes = [j - 1 for j in all_P_branch_Js if j != 0]
    pP_Branch_K = [j - i for j in P_branch_Js for i in range(j)]
    pP_Branch_Kprime = [k - 1 for k in pP_Branch_K]
    rP_Branch_K = [i for j in P_branch_Js for i in sorted(range(0, j), reverse=True)]
    rP_Branch_Kprime = [k + 1 for k in rP_Branch_K]
    
    '''Q Branch'''
    Q_branch_Js = list(range(0, Jmax + 1))
    all_Q_branch_Js = [j for j in Q_branch_Js if j != 0 for _ in range(j)]
    Q_branch_Jprimes = all_Q_branch_Js[:]
    pQ_Branch_K = [j - i for j in Q_branch_Js for i in range(j)]
    pQ_Branch_Kprime = [k - 1 for k in pQ_Branch_K]
    rQ_Branch_K = [i for j in Q_branch_Js for i in sorted(range(0, j), reverse=True)]
    rQ_Branch_Kprime = [k + 1 for k in rQ_Branch_K]
    
    '''R Branch'''
    R_branch_Js = list(range(0, Jmax))
    all_R_branch_Js = [j for j in R_branch_Js if j == 0 or j != 0 for _ in range(j + 1)]
    R_branch_Jprimes = [j + 1 for j in all_R_branch_Js if j <= Jmax - 1]
    pR_Branch_K = [j - (i - 1) for j in R_branch_Js for i in range(j + 1)]
    pR_Branch_Kprime = [k - 1 for k in pR_Branch_K]
    rR_Branch_K = [i for j in R_branch_Js for i in sorted(range(0, j + 1), reverse=True)]
    rR_Branch_Kprime = [k + 1 for k in rR_Branch_K]
    
    '''Combine results'''
    Allowed_Js = (all_P_branch_Js * 2) + (all_Q_branch_Js * 2) + (all_R_branch_Js * 2)
    Allowed_Jprimes = (P_branch_Jprimes * 2) + (Q_branch_Jprimes * 2) + (R_branch_Jprimes * 2)
    Allowed_Ks = pP_Branch_K + rP_Branch_K + pQ_Branch_K + rQ_Branch_K + pR_Branch_K + rR_Branch_K
    Allowed_Kprimes = pP_Branch_Kprime + rP_Branch_Kprime + pQ_Branch_Kprime + rQ_Branch_Kprime + pR_Branch_Kprime + rR_Branch_Kprime
    
    columns = {'ground_J': Allowed_Js, 'excited_J': Allowed_Jprimes, 'ground_K': Allowed_Ks, 'excited_K': Allowed_Kprimes}
    combinations = pd.DataFrame(columns)
  
    
    combinations['delta_J'] = combinations['excited_J'] - combinations['ground_J']
    combinations['delta_K'] = combinations['excited_K'] - combinations['ground_K']
    
    
    delta_J_values = combinations['delta_J']
    delta_K_values = combinations['delta_K']
    
    label = [
        'pP' if delta_J == -1 and delta_K == -1
        else 'rP' if delta_J == -1 and delta_K == 1
        else 'pQ' if delta_J == 0 and delta_K == -1
        else 'rQ' if delta_J == 0 and delta_K == 1
        else 'pR' if delta_J == 1 and delta_K == -1
        else 'rR'
        for delta_J, delta_K in zip(delta_J_values, delta_K_values)
    ]
    
    combinations['label'] = label
        
    return combinations


def get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin):
    
    '''Takes in 6 parameters (molecular and environmental) 
    and calculates spectra for it. It returns linelist and model data (i.e profile after convolution)
    '''
    startg = timeit.default_timer()

    combinations  = allowed_perperndicular_transitions(Jmax)
    # rotational constants in cm-1
    ground_B = B
    ground_C = ground_B / 2
    delta_C = delta_B
    excited_B = ground_B + ((delta_B / 100) * ground_B)
    excited_C = ground_C + ((delta_C / 100) * ground_C)

    ground_Js = combinations['ground_J']
    excited_Js = combinations['excited_J']
    ground_Ks = combinations['ground_K']
    excited_Ks = combinations['excited_K']

    linelist = combinations

    delta_J = linelist['excited_J'] - linelist['ground_J']
    delta_K = linelist['excited_K'] - linelist['ground_K']

    '''Calculating Linelist'''

    ground_Es = []
    for J, K in zip(ground_Js, ground_Ks):
        ground_E = ground_B * J * (J + 1) + (ground_C - ground_B) * (K ** 2)
        ground_Es.append(ground_E)

    linelist['ground_Es'] = ground_Es

    excited_Es = []
    for J, K, del_K in zip(excited_Js, excited_Ks, delta_K):
        if del_K == -1:
            excited_E = excited_B * J * (J + 1) + (excited_C - excited_B) * (K ** 2) - (
                (-2 * excited_C * zeta)) * K + excited_C ** 2
        elif del_K == 1:
            excited_E = excited_B * J * (J + 1) + (excited_C - excited_B) * (K ** 2) + (
                (-2 * excited_C * zeta)) * K + excited_C ** 2

        excited_Es.append(excited_E)

    linelist['excited_Es'] = excited_Es

    wavenos = []
    for i in range(len(linelist.index)):
        wavenumber = origin + excited_Es[i] - ground_Es[i]
        wavenos.append(wavenumber)

    linelist['wavenos'] = wavenos

    HL_factors = []

    for J, K, delta_J, delta_K in zip(ground_Js, ground_Ks, delta_J, delta_K):
        if delta_J == -1 and delta_K == -1:
            HL_factor = ((J - 1 + K) * (J + K)) / (J * ((2 * J) + 1))
        elif delta_J == -1 and delta_K == 1:
            HL_factor = ((J - 1 - K) * (J - K)) / (J * ((2 * J) + 1))
        elif delta_J == 0 and delta_K == -1:
            HL_factor = (J + 1 - K) * (J + K) / (J * (J + 1))
        elif delta_J == 0 and delta_K == 1:
            HL_factor = (J + 1 + K) * (J - K) / (J * (J + 1))
        elif delta_J == 1 and delta_K == -1:
            HL_factor = (J + 2 - K) * (J + 1 - K) / ((J + 1) * ((2 * J) + 1))
        elif delta_J == 1 and delta_K == 1:
            HL_factor = (J + 2 + K) * (J + 1 + K) / ((J + 1) * ((2 * J) + 1))

        HL_factors.append(HL_factor)

    linelist['HL_factors'] = HL_factors

    BD_factors = []

    h = const.h.cgs.value
    c = const.c.to('cm/s').value
    k = const.k_B.cgs.value

    for J, K, E in zip(ground_Js, ground_Ks, ground_Es):
        if K == 0:
            boltzmann_equation = (2 * ((2 * J) + 1)) * (np.exp((-h * c * E) / (k * T)))
        else:
            boltzmann_equation = (1 * ((2 * J) + 1)) * (np.exp((-h * c * E) / (k * T)))

        BD_factors.append(boltzmann_equation)

    linelist['BD_factors'] = BD_factors

    intensities = []
    for i in range(len(linelist.index)):
        strength = (HL_factors[i] * BD_factors[i])
        intensities.append(strength)

    linelist['intensities'] = intensities

    # endl = timeit.default_timer()
    # print('>>>> linelist calculation takes   ' + str(endl - startl) + '  sec')

    '''Smoothening the linelist'''

    smooth_wavenos = np.linspace(np.min(linelist['wavenos']) - 1, np.max(linelist['wavenos']) + 1, 1000)  # grid_size

    Wavenos_arr = np.array(linelist['wavenos'])
    Intenisty_arr = np.array(linelist['intensities'])

    @nb.njit(parallel=True)
    def calculate_smooth_intensities(wavenos, intensities, smooth_wavenos, sigma):
        smooth_intensities = np.zeros(smooth_wavenos.shape)

        for i in nb.prange(len(smooth_wavenos)):
            wavepoint = smooth_wavenos[i]
            w_int = np.exp(-(wavenos - wavepoint) ** 2 / (2 * sigma ** 2)) * intensities
            smooth_intensities[i] = np.sum(w_int)

        return smooth_intensities

    # call the numba function with input data
    smooth_intensities = calculate_smooth_intensities(Wavenos_arr, Intenisty_arr, smooth_wavenos, sigma)

    smooth_data = np.array([smooth_wavenos, smooth_intensities]).transpose()
    #smooth_data = np.delete(smooth_data, np.where(smooth_data[:, 1] <= 0.00001 * (max(smooth_data[:, 1]))), axis=0)

    simu_waveno = smooth_data[:, 0]
    simu_intenisty = 1 - 0.1 * (smooth_data[:, 1] / max(smooth_data[:, 1]))
    
    model_data = np.array([simu_waveno, simu_intenisty]).transpose()


    # for units in wavelength
    # simu_wavelength = (1/simu_waveno)*1e8
    # model_data = np.array([simu_wavelength, simu_intenisty]).transpose()
    # model_data = model_data[::-1]

    
    endg = timeit.default_timer()

    print('>>>> Time taken to simulate thi profile  ' + str(endg - startg) + '  sec')
    print('==========')
    return linelist, model_data









def obs_curve_to_fit(sightline): 
    
        '''This function reads in data, removes wings and provides just 
        the triple peak for fitting and calculates std dev for each sightline '''
    
        file = filename.format(sightline)
        Obs_data = pd.read_csv(spec_dir / file,
                               delim_whitespace=(True))
        
        Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
        Obs_data = Obs_data.iloc[::-1].reset_index(
            drop=True)  # making it ascending order as we transformed wavelength into wavenumbers

        # shifting to zero and scaling flux between 0.9 and 1
        min_index = np.argmin(Obs_data['Flux'])
        Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] 
        Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
        
        #plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset,  label = 'Data (HD ' + str(sightline) + ')' , color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))

        
        # removing red wing
        Obs_data_trp = Obs_data[(Obs_data['Flux'] <= 0.95)]  # trp = triple peak 
        
        # making data evenly spaced
        x_equal_spacing = np.linspace(min(Obs_data_trp['Wavelength']), max(Obs_data_trp['Wavelength']), 100)
        y_obs_data = np.interp(x_equal_spacing, Obs_data_trp['Wavelength'], Obs_data_trp['Flux'])

        Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >= 2) & (Obs_data['Wavelength']<= 5)]
        std_dev = np.std(Obs_data_continuum['Flux'])
        
        return Obs_data, x_equal_spacing, y_obs_data, std_dev
    
    


def get_multi_spectra( **params_list):
    """
    Calculating a model for each sight line using 'get_rotational_spectrum'.
   
    Always using the same molecular parameters, but different T.
    Args:
        xx:
        B:
        T1:
        T2:
        delta_B:
        zeta:
        sigma:
        origin:

    Returns:
    np.array
        Model flux array. Fluxes for both sight lines are appended to one 1D array.
    """
    
   
    print('---------')
    
    B = params_list['B']
    delta_B = params_list['delta_B']
    zeta = params_list['zeta']
    
   
    first_T_index = 3
    last_T_index = first_T_index + len(sightlines) 
 
    first_sigma_index = last_T_index  
    last_sigma_index = first_sigma_index + len(sightlines) 
 
    first_origin_index = last_sigma_index  
    last_origin_index = first_origin_index +len(sightlines) 
 
    # T_values = params_list[first_T_index:last_T_index]
    # sigma_values = params_list[first_sigma_index:last_sigma_index]
    # origin_values = params_list[first_origin_index:last_origin_index]
    
    T_values = [params_list[f'T{i+1}'] for i in range(len(sightlines))]
    sigma_values = [params_list[f'sigma{i+1}'] for i in range(len(sightlines))]
    origin_values = [params_list[f'origin{i+1}'] for i in range(len(sightlines))]

    all_y_model_data = np.array([])
    
    for T, sigma, origin, sightline in zip(T_values, sigma_values, origin_values, sightlines):
        
        Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
        linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
        
        one_sl_y_model_data  = np.interp(x_equal_spacing, model_data[:, 0], model_data[:, 1])
        
        all_y_model_data = np.concatenate((all_y_model_data, one_sl_y_model_data))
        
    
    
    return all_y_model_data


def fit_model(B, delta_B, zeta, T, sigma, origin):
    mod = Model(get_multi_spectra) 
    
    
    
    params_list = [B, delta_B, zeta]
    
    T_list = [T] * len(sightlines)
    sigma_list = [sigma] * len(sightlines)
    origin_list = [origin] * len(sightlines)

    params_list.extend(T_list)
    params_list.extend(sigma_list)
    params_list.extend(origin_list)
    
    
    print(params_list)
    
    
    
    first_T_index = 3
    last_T_index = first_T_index + len(sightlines) 
 
    first_sigma_index = last_T_index  
    last_sigma_index = first_sigma_index + len(sightlines) 
 
    first_origin_index = last_sigma_index  
    last_origin_index = first_origin_index +len(sightlines) 
 
    params = Parameters()
    params.add('B', value = B, min = 0.0005, max = 0.01)
    params.add('delta_B', value = delta_B, min = -1, max =0)
    params.add('zeta', value = zeta, min = -1, max = 1)
    
    for i, param_value in enumerate(params_list[first_T_index:last_T_index]):
        params.add(f'T{i+1}', value=param_value, min = 2.7, max = 500)
        
    for i, param_value in enumerate(params_list[first_sigma_index:last_sigma_index]):
        params.add(f'sigma{i+1}', value=param_value, min = 0.05, max = 0.3)
        
    for i, param_value in enumerate(params_list[first_origin_index:last_origin_index]):
        params.add(f'origin{i+1}', value=param_value, min = -1, max = 1)
        
   
    result = mod.fit(flux_list, params, xx=wave_list, weights = 1/stddev_array )  # method = 'leastsq', fit_kws={'ftol': 1e-12, 'xtol': 1e-12}
    print(result.fit_report())
    
    def plot_best_fit(result, x_equal_spacing, y_obs_data):
        plt.figure()
        plt.scatter(x_equal_spacing, y_obs_data, label='Observations')
        plt.plot(x_equal_spacing, result.best_fit, 'r-', label='Best Fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
            
    plot_best_fit(result, x_equal_spacing, y_obs_data)
    
    return result




'''Inputs'''    
Jmax = 300

#Cami 2004
spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/Cami_2004_data/heliocentric/6614/")
sightlines = ['184915'] #['144217', '144470'] #, '145502', '147165', '149757', '179406', '184915']
filename = 'hd{}_dib6614.txt'


#EDIBLES data
#spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Data/Heather's_data")
#filename = '6614_HD{}.txt'
#sightlines = ['23180', '24398'] #, '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']


flux_list = np.array([])
wave_list = np.array([])
stddev_array = np.array([])
for sightline in sightlines:
    
    Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)

    flux_list = np.concatenate((flux_list, y_obs_data))
    wave_list = np.concatenate((wave_list, x_equal_spacing))
    
    one_sl_stddev = [std_dev] * len(x_equal_spacing)
    stddev_array = np.concatenate((stddev_array, one_sl_stddev))
    
    

#result = fit_model(B = 0.002, T = 82.5, delta_B = -0.45, zeta = -0.01, sigma = 0.17, origin =  0.012)


'''PLotting'''


# plt.figure(figsize = (15,8))
B =       0.00330308 #+/- 8.7988e-05 (3.54%) (init = 0.0023)		
delta_B =  -0.026643322 #+/- 0.00301885 (4.41%) (init = -0.0353)		
zeta =  -0.11860631 #+/- 0.00953055 (3.05%) (init = -0.4197)		

Ts = [86.19, 84.64, 93.08, 96.69, 82.35, 75.69, 79.07]
origins =  [0.034, 0.0033, -0.052, -0.0197, 0.070, 0.016]
sigma = 0.0289 #, 0.11, 0.20]

for T,  origin in zip(Ts,  origins):
    linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
    plt.plot(model_data[:,0], model_data[:,1], label = sigma)
    plt.legend()


#plt.plot(x_equal_spacing, y_obs_data, label = 'Data (HD ' + str(sightline) + ')')

#data = pd.read_excel("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting methods/master_fitting_results_in_ a_table copy.xlsx", header = 0)




# Ts = list(data['Temp'])
# sigmas = list(data['sigma'])
# origins = list(data['origin'])
# offset = np.arange(0, 6, 0.06)
# # sightlines = list(data['Sightline'])

# for T, origin, offset, sightline in zip(Ts, origins, offset, sightlines):
#     Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
#     # plt.plot(x_equal_spacing, y_obs_data - offset , label = 'Data (HD ' + str(sightline) + ')', color = blue)


#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'red', label = 'Model')
#     plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
#     plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     plt.annotate('HD' + str(sightline), xy = (Obs_data['Wavelength'][150] , Obs_data['Flux'][150] - offset) , xytext = (4, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     plt.annotate('T = {:.2f}'.format(T) + ' K', xy = (Obs_data['Wavelength'][40] , Obs_data['Flux'][40] - offset) , xytext = (-7, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + '  cm$^{-1}$', xy = (Obs_data['Wavelength'][50] , Obs_data['Flux'][50] - offset) , xytext = (-5, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17)
#     plt.xlim(-7.5, 6)
#     plt.legend(loc = 'lower left', fontsize = 16)
    

# def one_sl_fit_model(B, delta_B, zeta, T, sigma, origin):
#     mod = Model(model_curve_to_fit) 
#     params = mod.make_params( B = B, delta_B = delta_B, zeta = zeta, T=T,sigma = sigma, origin = origin)
    
#     print(params)
#     params['B'].min = 0.0005 
#     params['B'].max = 0.01
#     params['T'].min = 2.7
#     params['T'].max = 300
#     params['origin'].min = -2
#     params['origin'].max = 2
#     params['delta_B'].min = -1
#     params['delta_B'].max = 0
#     params['zeta'].min = -1
#     params['zeta'].max = 1
#     params['sigma'].min = 0.05
#     params['sigma'].max = 0.3

#     Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
#     #print(std_dev)
#     result = mod.fit(y_obs_data, params, x_equal_spacing = x_equal_spacing, weights = 1/std_dev)
#     print(result.fit_report())
    
#     def plot_best_fit(result, x_equal_spacing, y_obs_data):
#         plt.figure()
#         plt.scatter(x_equal_spacing, y_obs_data, label='Observations')
#         plt.plot(x_equal_spacing, result.best_fit, 'r-', label='Best Fit')
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.legend()
#         plt.show()
            
#     plot_best_fit(result, x_equal_spacing, y_obs_data)
#     return result


# def model_curve_to_fit(x_equal_spacing, B, delta_B, zeta, T, sigma, origin):
    
#     '''This function does interpolation and makes sure model and observations 
#      have same data point over x-axis. Output of this function is provided for fitting. 
#     '''
    
#     linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
    
#     y_model_data = model_data[:,1]
    
    
#     Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
    
#     y_model_data = np.interp(x_equal_spacing, model_data[:,0], model_data[:,1])
    
#     return y_model_data
    

