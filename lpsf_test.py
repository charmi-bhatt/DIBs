#To fix: everything should be given as an argument : get_multi_spectra (**params_list, sightline??)



import LPS_functions as lps
import matplotlib.pyplot as plt
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
    params.add('B', value = B, min = 0.0005, max = 0.05, vary = False)
    params.add('delta_B', value = delta_B, min = -1, max =0, vary = False)
    params.add('zeta', value = zeta, min = -1, max = 1, vary = False)
    
    for i, param_value in enumerate(params_list[first_T_index:last_T_index]):
        params.add(f'T{i+1}', value=param_value, min = 2.7, max = 500)
        
    for i, param_value in enumerate(params_list[first_sigma_index:last_sigma_index]):
        params.add(f'sigma{i+1}', value=param_value, min = 0.05, max = 0.3)
        
    for i, param_value in enumerate(params_list[first_origin_index:last_origin_index]):
        params.add(f'origin{i+1}', value=param_value, min = -1, max = 1)
        
   
    result = mod.fit(flux_list, params, xx=wave_list, weights = 1/stddev_array , method = method) #, fit_kws={'ftol': 1e-2, 'xtol': 1e-2} )
    print(result.fit_report())
    
    # def plot_best_fit(result, x_equal_spacing, y_obs_data):
    #     plt.figure()
    #     plt.scatter(x_equal_spacing, y_obs_data, label='Observations')
    #     plt.plot(x_equal_spacing, result.best_fit, 'r-', label='Best Fit')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.legend()
    #     plt.show()
            
    #plot_best_fit(result, x_equal_spacing, y_obs_data)
    
    return result

Jmax = 300
combinations  = lps.allowed_perperndicular_transitions(Jmax)

flux_list = np.array([])
wave_list = np.array([])
stddev_array = np.array([])
for sightline in sightlines:
    
    Obs_data, Obs_y_data_to_fit, std_dev= lps.obs_curve_to_fit(sightline)

    flux_list = np.concatenate((flux_list, Obs_y_data_to_fit))
    wave_list = np.concatenate((wave_list, common_grid_for_all))
    
    one_sl_stddev = [std_dev] * len(common_grid_for_all)
    stddev_array = np.concatenate((stddev_array, one_sl_stddev))

B=       0.00241776 
delta_B= -0.06991853 
zeta=   -0.312885821 
T=       93.3047616 
sigma=   0.19567868 
origin=  0.02034466 

linelist, model_data =  lps.get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin, combinations)
plt.plot(model_data[:,0], model_data[:,1], color = 'red', label = 'Model')