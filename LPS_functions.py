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


def get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin, combinations):

    

    '''Takes in 6 parameters (molecular and environmental) 
    and calculates spectra for it. It returns linelist and model data (i.e profile after convolution)
    '''
    
    combinations = combinations
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
    
    
    #Compute HL_factors directly in a list comprehension
    HL_factors = [((J - 1 + K) * (J + K)) / (J * ((2 * J) + 1)) if (delta_J == -1 and delta_K == -1) else
                  ((J - 1 - K) * (J - K)) / (J * ((2 * J) + 1)) if (delta_J == -1 and delta_K == 1) else
                  (J + 1 - K) * (J + K) / (J * (J + 1)) if (delta_J == 0 and delta_K == -1) else
                  (J + 1 + K) * (J - K) / (J * (J + 1)) if (delta_J == 0 and delta_K == 1) else
                  (J + 2 - K) * (J + 1 - K) / ((J + 1) * ((2 * J) + 1)) if (delta_J == 1 and delta_K == -1) else
                  (J + 2 + K) * (J + 1 + K) / ((J + 1) * ((2 * J) + 1)) if (delta_J == 1 and delta_K == 1) else
                  None for J, K, delta_J, delta_K in zip(ground_Js, ground_Ks, delta_J, delta_K)]
    
    linelist['HL_factors'] = HL_factors
    

    
    BD_factors = []

    h = const.h.cgs.value
    c = const.c.to('cm/s').value
    k = const.k_B.cgs.value

    ground_Js_np = np.array(ground_Js)
    ground_Ks_np = np.array(ground_Ks)
    ground_Es_np = np.array(ground_Es)
    
    # Calculation of static part
    static_part = (-h * c) / (k * T)
    
    # For the condition K == 0, the equation becomes twice the general equation
    # we can pre-calculate that part
    factor = (2 * ground_Js_np + 1) * np.exp(static_part * ground_Es_np)
    
    # Wherever ground_Ks == 0, double the result
    boltzmann_equation = np.where(ground_Ks_np == 0, 2 * factor, factor)
    
    # Convert the numpy array back to list if necessary
    BD_factors = boltzmann_equation.tolist()

    linelist['BD_factors'] = BD_factors
    

    intensities = []
    for i in range(len(linelist.index)):
        strength = (HL_factors[i] * BD_factors[i])
        intensities.append(strength)

    linelist['intensities'] = intensities
    
   
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

    return linelist, model_data


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
    
    n_points = (
        round(
            (np.log(lambda_end / lambda_start)) / (np.log(-(1 + 2 * R) / (1 - 2 * R)))
        )
        + 1
    )
    f = -(1 + 2 * R) / (1 - 2 * R)
    
    factor = f ** np.arange(n_points)
    wave = np.full(int(n_points), lambda_start, dtype=np.float)
    grid = wave * factor
    return grid


def obs_curve_to_fit(sightline, spec_dir, filename, common_grid_for_all): 
    
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
        
        #plt.plot(Obs_data['Wavelength'], Obs_data['Flux'])
        
        Obs_y_data_to_fit  = np.interp(common_grid_for_all, Obs_data['Wavelength'], Obs_data['Flux'])
        
       
        
        #Obs_data_95 = Obs_data [(Obs_data['Flux'] <=0.95)]
       

        Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >=2) & (Obs_data['Wavelength']<= 5)]
        std_dev = np.std(Obs_data_continuum['Flux'])
        
       
        return Obs_data, Obs_y_data_to_fit, std_dev
    
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
        
        linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
        
        one_sl_y_model_data  = np.interp(common_grid_for_all, model_data[:, 0], model_data[:, 1])
        
        all_y_model_data = np.concatenate((all_y_model_data, one_sl_y_model_data))
        
    
    # plt.plot(common_grid_for_all, all_y_model_data)
    # plt.show()
    return all_y_model_data