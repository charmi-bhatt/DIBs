#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:55:26 2023

@author: charmibhatt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:28:52 2023

@author: charmibhatt
"""

# import numpy as np
# import pandas as pd
# import astropy.constants as const
# import matplotlib.pyplot as plt
# import timeit
# import scipy.stats as ss
# from scipy.signal import argrelextrema
# from matplotlib import cm
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


flux_list = np.array([])
wave_list = np.array([])
stddev_array = np.array([])
spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/Cami_2004_data/heliocentric/6614/")


sightlines = ['184915'] #, '184915', '144470', '145502', '147165', '149757', '179406', '184915']

#sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']


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
        # removing red wing
        # Obs_data_trp = Obs_data [(Obs_data['Wavelength'] >= -1) & (Obs_data['Wavelength']<= 1.2)]
        Obs_data_trp = Obs_data[(Obs_data['Flux'] <= 0.95)]  # trp = triple peak structure

        # making data evenly spaced
        x_equal_spacing = np.linspace(min(Obs_data_trp['Wavelength']), max(Obs_data_trp['Wavelength']), 100)
        
        return x_equal_spacing
    

for sightline in sightlines:
    file ='hd{}_dib6614.txt'.format(sightline)
    Obs_data = pd.read_csv(spec_dir / file,
                            delim_whitespace=(True))
    print(Obs_data['Wavelength'])
        
    
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    Obs_data = Obs_data.iloc[::-1].reset_index(
        drop=True)  # making it ascending order as we transformed wavelength into wavenumbers

    # shifting to zero and scaling flux between 0.9 and 1
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index]
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    
    plt.plot(Obs_data['Wavelength'], Obs_data['Flux'], label  = sightline)
    # plt.legend()
    # plt.show()
    # removing red wing
    Obs_data_trp = Obs_data[(Obs_data['Flux'] <= 0.95)]  # trp = triple peak structure

    # making data evenly spaced
    x_equal_spacing = np.linspace(min(Obs_data_trp['Wavelength']), max(Obs_data_trp['Wavelength']),
                                  100)
   
    
    #print(x_equal_spacing)
    y_obs_data = np.interp(x_equal_spacing, Obs_data_trp['Wavelength'], Obs_data_trp['Flux'])
    flux_list = np.concatenate((flux_list, y_obs_data))
    wave_list = np.concatenate((wave_list, x_equal_spacing))
    
    Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >= 2) & (Obs_data['Wavelength']<= 5)]
    std_dev = np.std(Obs_data_continuum['Flux'])
    one_sl_stddev = [std_dev] * len(x_equal_spacing)
    stddev_array = np.concatenate((stddev_array, one_sl_stddev))
    

    
combinations = pd.read_csv(r"/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/edibles/edibles/utils/simulations/Charmi/Jmax=300.txt", delim_whitespace=(True))

startl = timeit.default_timer()


def get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin):
    
    #recalculate this
    startg = timeit.default_timer()

    # print(B)
    # print(T)
    # print('-----------')

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

    endl = timeit.default_timer()
    print('>>>> linelist calculation takes   ' + str(endl - startl) + '  sec')

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
    smooth_data = np.delete(smooth_data, np.where(smooth_data[:, 1] <= 0.001 * (max(smooth_data[:, 1]))), axis=0)

    simu_waveno = smooth_data[:, 0]
    simu_intenisty = 1 - 0.1 * (smooth_data[:, 1] / max(smooth_data[:, 1]))
    
    # with sns.color_palette("flare", n_colors=2):

    # plt.plot(simu_waveno, simu_intenisty, color='red')

    # for units in wavelngth
    # simu_wavelength = (1/simu_waveno)*1e8
    # model_data = np.array([simu_wavelength, simu_intenisty]).transpose()
    # model_data = model_data[::-1]

    model_data = np.array([simu_waveno, simu_intenisty]).transpose()
    # print(model_data)
    # plt.figure(figsize=(15,8))
    # plt.plot(x_equal_spacing, y_obs_data, color= 'green')

    # plt.plot(x_equal_spacing, y_model_data)
    # plt.xlabel('Wavelength')
    # plt.ylabel('Normalized Intenisty')
    # plt.title('Temperature = ' + str(T) + '  K  ground_B =  ' + str(ground_B) + ' cm-1  ground_C=  ' + str(ground_C) + ' cm-1  Delta_B = ' + str(delta_B) + '    $\sigma$ = ' + str(sigma) +    '    zeta = ' +  str(zeta)) 

    # plt.show()

    # plt.figure(figsize=(15,8))

    # with sns.color_palette("flare", n_colors=2):

    # plt.stem(x_equal_spacing, y_obs_data - y_model_data)
    # plt.axhline(y=0)
    endg = timeit.default_timer()

    print('>>>> full takes   ' + str(endg - startg) + '  sec')
    print('==========')
    return linelist, model_data


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
    T = params_list['T']
    sigma = params_list['sigma']
    origin = params_list['origin']
    
   
    first_T_index = 3
    last_T_index = first_T_index + len(sightlines) 
 
    # first_sigma_index = last_T_index  
    # last_sigma_index = first_sigma_index + len(sightlines) 
 
    first_origin_index = last_T_index  #last_sigma_index
    last_origin_index = first_origin_index +len(sightlines) 
 
    # T_values = params_list[first_T_index:last_T_index]
    # sigma_values = params_list[first_sigma_index:last_sigma_index]
    # origin_values = params_list[first_origin_index:last_origin_index]
    
    # T_values = [params_list[f'T{i+1}'] for i in range(len(sightlines))]
    # #sigma_values = [params_list[f'sigma{i+1}'] for i in range(len(sightlines))]
    # origin_values = [params_list[f'origin{i+1}'] for i in range(len(sightlines))]


    
    #xx = x_equal_spacing


    all_y_model_data = np.array([])
    
    # for T, origin, sightline in zip(T_values, origin_values, sightlines): #sigma_values
    #     print(B)
    #     print(T)
    #     print(delta_B)
    #     print(zeta)
    #     #print(sigma)
    #     print(origin)
    x_equal_spacing = curve_to_fit_wavenos(sightline)
    model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
    #print(model_data)
    one_sl_y_model_data  = np.interp(x_equal_spacing, model_data[:, 0], model_data[:, 1])
    #print(one_sl_y_model_data )
    all_y_model_data = np.concatenate((all_y_model_data, one_sl_y_model_data))
    
    print(len(all_y_model_data))
    
    return all_y_model_data



def master_fit_model(B, delta_B, zeta, T, sigma, origin):
    mod = Model(get_multi_spectra) 
    
    
    
    params_list = [B, delta_B, zeta]
    
    T_list = [T] * len(sightlines)
    #sigma_list = [sigma] * len(sightlines)
    origin_list = [origin] * len(sightlines)

    params_list.extend(T_list)
    #params_list.extend(sigma_list)
    params_list.extend(origin_list)
    
    
    # print(params_list)
    
    
    
    first_T_index = 3
    last_T_index = first_T_index + len(sightlines) 
 
    # first_sigma_index = last_T_index  
    # last_sigma_index = first_sigma_index + len(sightlines) 
 
    first_origin_index = last_T_index  #last_sigma_index  
    last_origin_index = first_origin_index +len(sightlines) 
 
    params = Parameters()
    params.add('B', value = B, min = 0.0005, max = 0.01)
    params.add('delta_B', value = delta_B, min = -1, max =0)
    params.add('zeta', value = zeta, min = -1, max = 1)
    
    params.add('T', value = T, min = 2.7, max = 500)
    params.add('sigma', value = T, min = 0.05, max = 3)
    params.add('origin', value = origin, min = -1, max =1)
    
    # for i, param_value in enumerate(params_list[first_T_index:last_T_index]):
    #     #params.add(f'T{i+1}', value=param_value, min = 2.7, max = 500)

    # # for i, param_value in enumerate(params_list[first_sigma_index:last_sigma_index]):
    # #     params.add(f'sigma{i+1}', value=param_value, min = 0.05, max = 0.3)
        
    # for i, param_value in enumerate(params_list[first_origin_index:last_origin_index]):
    #     params.add(f'origin{i+1}', value=param_value, min = -1, max = 1)
        
   
    #params = mod.make_params(B=B, T1=T, T2=T, delta_B=delta_B, zeta=zeta, sigma1=sigma, sigma2=sigma, origin1=origin, origin2=origin)

    params['B'].min = 0.0005
    params['B'].max = 0.01
    params['T'].min = 2.7
    params['T'].max = 300
    params['origin'].min = -2
    params['origin'].max = 2
    params['delta_B'].min = -1
    params['delta_B'].max = 0
    params['zeta'].min = -1
    params['zeta'].max = 1
    params['sigma'].min = 0.01
    params['sigma'].max = 0.3  
    
    

    #my_list = [xx, B, T1, T2, delta_B, zeta, sigma, origin]

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

B =       0.00327308 #+/- 8.7988e-05 (3.54%) (init = 0.0023)		
delta_B =  -0.0260743322 #+/- 0.00301885 (4.41%) (init = -0.0353)		
zeta =  -0.11700631 #+/- 0.00953055 (3.05%) (init = -0.4197)		

Ts = [86.19, 85.339, 93.43, 96.44, 83.18, 76.53, 79.64]
origins =  [0.033, 0.0025, -0.054, -0.024, 0.068, 0.078, 0.016]
sigma = 0.029

# Ts = list(data['Temp'])
# sigmas = list(data['sigma'])
# origins = list(data['origin'])
offset = np.arange(0, 6, 0.06)
# # sightlines = list(data['Sightline'])

for T, origin, offset, sightline in zip(Ts, origins, offset, sightlines):
    #Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
    # plt.plot(x_equal_spacing, y_obs_data - offset , label = 'Data (HD ' + str(sightline) + ')', color = blue)


    linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
    plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'red', label = 'Model')
    plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
    plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.annotate('HD' + str(sightline), xy = (Obs_data['Wavelength'][150] , Obs_data['Flux'][150] - offset) , xytext = (4, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
    plt.annotate('T = {:.2f}'.format(T) + ' K', xy = (Obs_data['Wavelength'][40] , Obs_data['Flux'][40] - offset) , xytext = (-7, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
    plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + '  cm$^{-1}$', xy = (Obs_data['Wavelength'][50] , Obs_data['Flux'][50] - offset) , xytext = (-5, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17)
    plt.xlim(-7.5, 6)
    plt.legend(loc = 'lower left', fontsize = 16)
    





#result1= master_fit_model(B = 0.01,  delta_B = -0.1, zeta = -0.1, T = 2.7, sigma = 0.15, origin =  0)
#result2= master_fit_model(B = 0.002,  delta_B = -0.4, zeta = -0.9,T = 92.7,  sigma = 0.2, origin =  0.039)
# result3= master_fit_model(B = 0.002, delta_B = -0.8, zeta = -0.5, T = 62.5,   origin =  0.072)
# result4= master_fit_model(B = 0.0003,  delta_B = -0.45, zeta = -0.01,T = 32.0 , origin =  0.012)
# result5= master_fit_model(B = 0.0075, delta_B = -0.23, zeta = -0.23,T = 92.5 , origin =  0.02)

#results_list = [result2] #, result2, result3, result4, result5]


def write_results_to_csv(results_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['result_name', 'B_init', 'T_init', 'delta_B_init', 'zeta_init'  ,'sigma_init'  ,'origin_init' ,  'B', 'T',  'delta_B', 'zeta', 'sigma', 'origin', 'chi2', 'redchi', 'func_evals', 'B_unc', 'T_unc', 'delta_B_unc', 'zeta_unc', 'sigma_unc', 'origin_unc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(results_list):
            result_name = f'result{i+1}'
            params = result.params
            row = {
                'result_name': result_name,
                'B_init': params['B'].init_value,
                'T_init': params['T'].init_value,
                'delta_B_init': params['delta_B'].init_value,
                'zeta_init': params['zeta'].init_value,
                'sigma_init': params['sigma'].init_value,
                'origin_init': params['origin'].init_value,
                'B': params['B'].value,
                'T': params['T'].value,
                'delta_B': params['delta_B'].value,
                'zeta': params['zeta'].value,
                'sigma': params['sigma'].value,
                'origin': params['origin'].value,
                'chi2': result.chisqr,
                'redchi': result.redchi,
                'func_evals': result.nfev,
                'B_unc': params['B'].stderr,
                'T_unc': params['T'].stderr,
                'delta_B_unc': params['delta_B'].stderr,
                'zeta_unc': params['zeta'].stderr,
                # 'sigma_unc': params['sigma'].stderr,
                'origin_unc': params['origin'].stderr,

            }
            writer.writerow(row)
            
name_of_file = str(sightlines[0])+ "_Cami_2004_variable_sigma.csv"
print(name_of_file)
#write_results_to_csv(results_list, name_of_file)     
            


def save_lmfit_report(filename, report):
    with open(filename, 'w') as f:
        # Save variables
        f.write("Variables:\n")
        for param_name, param_value in report.params.valuesdict().items():
            f.write(f"{param_name}: {param_value}\n")
        f.write("\n")

        # Save uncertainties
        f.write("Uncertainties:\n")
        for param_name, param_stderr in report.params.stderr.items():
            f.write(f"{param_name}: {param_stderr}\n")
        f.write("\n")

        # Save function evaluations
        f.write("Function evaluations:\n")
        for func_name, func_value in report.userkws.items():
            f.write(f"{func_name}: {func_value}\n")
        f.write("\n")

# Example usage
# Assuming you have an lmfit report object called 'fit_report'
#save_lmfit_report('output.txt', fit_report)

def obs_curve_to_fit(sightline): 
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
        
        #plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset,  label = 'Data (HD ' + str(sightline) + ')' , color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))

        
        # removing red wing
        # Obs_data_trp = Obs_data [(Obs_data['Wavelength'] >= -1) & (Obs_data['Wavelength']<= 1.2)]
        Obs_data_trp = Obs_data[(Obs_data['Flux'] <= 1)]  # trp = triple peak structure

        # making data evenly spaced
        x_equal_spacing = np.linspace(min(Obs_data_trp['Wavelength']), max(Obs_data_trp['Wavelength']), 100)
        y_obs_data = np.interp(x_equal_spacing, Obs_data_trp['Wavelength'], Obs_data_trp['Flux'])

        Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >= 2) & (Obs_data['Wavelength']<= 5)]
        std_dev = np.std(Obs_data_continuum['Flux'])
        
        return Obs_data, x_equal_spacing, y_obs_data, std_dev
    
# Example usage
# Assuming you have an lmfit report object called 'fit_report'
#save_lmfit_report('output.txt', fit_report)
sightlines = ['144217', '144470', '145502', '147165', '149757', '179406', '184915']

plt.figure(figsize = (15, 30))
B =       0.00330308 #+/- 8.7988e-05 (3.54%) (init = 0.0023)		
delta_B =  -0.026643322 #+/- 0.00301885 (4.41%) (init = -0.0353)		
zeta =  -0.11860631 #+/- 0.00953055 (3.05%) (init = -0.4197)		

Ts = [86.19, 84.64, 93.08, 96.69, 82.35, 75.69, 79.07]
origins =  [0.034, 0.0033, -0.052, -0.0197, 0.070, 0.079, 0.016]
sigma = 0.0289 #, 0.11, 0.20]
offset = np.arange(0, 7, 0.06)
# sightlines = list(data['Sightline'])

for T, origin, offset, sightline in zip(Ts, origins, offset, sightlines):
    Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
    plt.plot(x_equal_spacing, y_obs_data - offset , label = 'HD ' + str(sightline) + ',  T = {} K' .format(T), color = 'black')


    model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
    plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'red')
    plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
    plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.title('Cami_2004_data: ground_B = {:.5f} cm-1   Delta_B = {:.5f}    zeta = {:.5f}'.format(B, delta_B, zeta)) 

    #plt.annotate('HD' + str(sightline), xy = (Obs_data['Wavelength'][120] , Obs_data['Flux'][150] - offset) , xytext = (2, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
    # plt.annotate('T = {:.2f}'.format(T) + ' K', xy = (Obs_data['Wavelength'][40] , Obs_data['Flux'][40] - offset) , xytext = (-7, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
    # plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + '  cm$^{-1}$', xy = (Obs_data['Wavelength'][50] , Obs_data['Flux'][50] - offset) , xytext = (-5, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17)
    plt.xlim(-6, 6)
    plt.legend(loc = 'lower left', fontsize = 16)
    
    
    

'''Previously used code'''
 # plt.figure(figsize = (15,8))


 # plt.plot(x_equal_spacing, y_obs_data)
 # B=        0.00299988
 # delta_B= -0.02255252 
 # zeta=    -0.11856624
 # T=       93.6023585
 # sigma=    0.0289
 # origin=   0.03259578

 # B= 0.00330374
 # delta_B= 0.0266478
 # zeta= 0.11822396
 # T = 86.19
 # origin = 0.034
 # model_data =  get_rotational_spectrum(B, delta_B, zeta, T, origin)
 # plt.plot(model_data[:,0], model_data[:,1], color = 'red')
 # plt.xlabel('Wavelength')
 # plt.ylabel('Normalized Intenisty')
 # plt.title('ground_B = {:.5f} cm-1   Delta_B = {:.5f}    zeta = {:.5f} Temperature = {:.5f} K   $\sigma$ = {:.5f}    origin= {:.5f}\n\n'.format(B, delta_B, zeta, T, sigma, origin)) 
 # plt.legend(loc = 'lower left')
 # plt.xlim(-4,4)
