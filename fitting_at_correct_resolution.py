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

def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        print(f"Function {func.__name__} has been called {wrapper.call_count} times.")
        return func(*args, **kwargs)
    wrapper.call_count = 0
    return wrapper

startfull = timeit.default_timer()

@count_calls
def get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin):

    start1 = timeit.default_timer()

    '''Takes in 6 parameters (molecular and environmental) 
    and calculates spectra for it. It returns linelist and model data (i.e profile after convolution)
    '''
    
   


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

    
    end4 = timeit.default_timer()
    # print('>>>> Time taken for convolution  ' + str(end4 - start4) + '  sec')
    # print('==========')

    endfull = timeit.default_timer()
    print('>>>> Time taken for this profile  ' + str(endfull - startfull) + '  sec')
    print('=====')
    
    print('B = ' , ground_B)
    print('C =' , ground_C)
    print('excited_B = '  , excited_B)
    print( 'excited_C =', excited_C)
    print('delta_B = ' , delta_B)
    print('zeta = ' , zeta)
    print('T = ', T)
    print('sigma = ', sigma)
    print('origin = ' , origin) 
    print('Jmax = ',  max(combinations['excited_J']))
   
    

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
    
    #print('R = ' , R)
    n_points = (
        round(
            (np.log(lambda_end / lambda_start)) / (np.log(-(1 + 2 * R) / (1 - 2 * R)))
        )
        + 1
    )
    #print('n_points = ' , n_points)
    f = -(1 + 2 * R) / (1 - 2 * R)
    
    #print('f = ', f)
    factor = f ** np.arange(n_points)
    #print('factor = ' , factor)
    wave = np.full(int(n_points), lambda_start, dtype=np.float64)
    #print('wave = ' , wave)
    grid = wave * factor
    #print('grid = ', grid)
    return grid



def obs_curve_to_fit(sightline): 
    
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
        # shifting to 0 and scaling flux between 0.9 and 1
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

def write_results_to_csv(results_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['result_name', 'B_init', 'delta_B_init', 'zeta_init' , 'T1_init', 'sigma1_init' ,'origin1_init' ,  'B',   'delta_B', 'zeta', 'T1','sigma1', 'origin1', 'chi2', 'redchi', 'func_evals', 'B_unc', 'delta_B_unc', 'zeta_unc', 'T1_unc',  'sigma1_unc', 'origin1_unc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, result in enumerate(results_list):
            result_name = f'result{i+1}'
            params = result.params
            row = {
                'result_name': result_name,
                'B_init': params['B'].init_value,
                'delta_B_init': params['delta_B'].init_value,
                'zeta_init': params['zeta'].init_value,
                'T1_init': params['T1'].init_value,
                'sigma1_init': params['sigma1'].init_value,
                'origin1_init': params['origin1'].init_value,
                'B': params['B'].value,
                'delta_B': params['delta_B'].value,
                'zeta': params['zeta'].value,
                'T1': params['T1'].value,
                'sigma1': params['sigma1'].value,
                'origin1': params['origin1'].value,
                'chi2': result.chisqr,
                'redchi': result.redchi,
                'func_evals': result.nfev,
                'B_unc': params['B'].stderr,
                'delta_B_unc': params['delta_B'].stderr,
                'zeta_unc': params['zeta'].stderr,
                'T1_unc': params['T1'].stderr,
                'sigma1_unc': params['sigma1'].stderr,
                'origin1_unc': params['origin1'].stderr,

            }
            writer.writerow(row)



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
    params.add('B', value = B, min = 0.0005, max = 0.05) #, vary = False)
    params.add('delta_B', value = delta_B, min = -1, max =0) #, vary = False)
    params.add('zeta', value = zeta, min = -1, max = 1) #, vary = False)
    
    for i, param_value in enumerate(params_list[first_T_index:last_T_index]):
        params.add(f'T{i+1}', value=param_value, min = 2.7, max = 500)
        
    for i, param_value in enumerate(params_list[first_sigma_index:last_sigma_index]):
        params.add(f'sigma{i+1}', value=param_value, min = 0.05, max = 0.3)
        
    for i, param_value in enumerate(params_list[first_origin_index:last_origin_index]):
        params.add(f'origin{i+1}', value=param_value, min = -1, max = 1)
        
   
    result = mod.fit(flux_list, params, xx=wave_list, weights = 1/stddev_array , method = method) #, fit_kws={'ftol': 1e-2, 'xtol': 1e-2} )
    
    
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





############################################################################################
############################################################################################

#Inputs

############################################################################################
############################################################################################





#checking effect of Jmax
# B = 0.0013337 # 0.0026 
# delta_B = -0.037003020 # -0.07 
# zeta = -0.30513462
# T = 160 #95 #
# sigma = 0.2
# origin = 0
# Jmax = [1000, 800, 300]#, 400, 500, 600, 700]

# plt.figure(figsize=(30,8))

# colors = [ 'red', 'green', 'blue']
# plt.figure(figsize=(30,8))
# for idx, J in enumerate(Jmax): 
    
#     combinations = allowed_perperndicular_transitions(J)
#     linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)

#     P_branch = linelist[(linelist['delta_J'] == 1)]
#     plt.stem(P_branch['wavenos'], P_branch['intensities'], label=str(J), linefmt=colors[idx]) #, markerfmt='o'+colors[idx])    
#     plt.legend() 
#     plt.title('R Branch')  
#     #plt.plot(model_data[:,0], model_data[:,1], label = J)
#     #plt.plot(linelist['wavenos'], linelist['intensities'])

# plt.show()



Jmax = 1000
combinations  = allowed_perperndicular_transitions(Jmax) 

#Cami 2004
# spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/Cami_2004_data/heliocentric/6614/")
# sightlines = ['144217', '144470',  '145502', '147165', '149757', '179406', '184915'] 
# filename = 'hd{}_dib6614.txt'
method = 'leastsq'

#EDIBLES data
spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Data/Heather's_data")
filename = '6614_HD{}.txt'
sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']
#sightlines = ['185418']


lambda_start = 6611 #6612.5453435174495 #-1.134 #
lambda_end = 6616 # 6615 #1.039609311008462 #


common_grid_for_all = make_grid(lambda_start, lambda_end, resolution=220000, oversample=2)
common_grid_for_all = (1 / common_grid_for_all) * 1e8
common_grid_for_all = common_grid_for_all - 15119.4
common_grid_for_all = common_grid_for_all[::-1]


common_grid_for_all = common_grid_for_all[(common_grid_for_all > -1.14) & (common_grid_for_all < 1.6)]
print(common_grid_for_all.shape)

flux_list = np.array([])
wave_list = np.array([])
stddev_array = np.array([])
for sightline in sightlines:
    
    Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)

    flux_list = np.concatenate((flux_list, Obs_y_data_to_fit))
    wave_list = np.concatenate((wave_list, common_grid_for_all))
    
    one_sl_stddev = [std_dev] * len(common_grid_for_all)
    stddev_array = np.concatenate((stddev_array, one_sl_stddev))
  
plt.plot(wave_list, flux_list)
#plt.show()

result1 = fit_model(B = 0.001, delta_B = -0.1, zeta = -0.312, T = 100, sigma = 0.18 , origin =  0.014)
result2 = fit_model(B = 0.002, delta_B = -0.1, zeta = -0.312, T = 90, sigma = 0.18 , origin =  0.014)
result3 = fit_model(B = 0.0001, delta_B = -0.1, zeta = -0.312, T = 180, sigma = 0.18 , origin =  0.014)
result4  = fit_model(B = 0.005, delta_B = -0.1, zeta = -0.312, T = 60, sigma = 0.18 , origin =  0.014)
result5  = fit_model(B = 0.009, delta_B = -0.1, zeta = -0.312, T = 40, sigma = 0.18 , origin =  0.014)

report = result1.fit_report()
print(report)

workdir = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Fitting_at_Jmax_1000/"
save_fit_result_as = workdir + 'fit_report_Jmax_1000_T_init{:.2f}.txt'.format(T)
np.savetxt(save_fit_result_as, report)

# with open("Alto_fit_report_Cami_3.txt", "w") as f:
#     f.write(report)

# results_list = [result1, result2, result3] #, result4, result5]
# fit_report_filename = str(sightline) + '_Correct_res_3_init_conditions.csv'
# write_results_to_csv(results_list,fit_report_filename  )




############################################################################################
############################################################################################

#Altogether Fits Plotting

############################################################################################
############################################################################################

'''EDIBLES'''
B = 0.00260337
delta_B = -0.07003020
zeta = -0.30513462
Ts = [79.1337551, 86.3583384, 88.8431642, 95.5643204, 86.0534748, 80.1042636, 89.7854061, 81.9806263, 89.0445017, 81.2361554, 89.3558085, 80.9561796]
sigmas = [0.18055114, 0.19760700, 0.18423023, 0.18244127, 0.21244470, 0.15821120, 0.17367824, 0.19180354, 0.20377079, 0.21578868, 0.24029108, 0.18563300]
origins = [0.02073839, -0.01342350, 0.00257408, -0.01106586, 0.02076492, -0.00374113, 0.06300594, 0.01604905, 0.11597056, 0.07103430, 0.03415804, 0.07097977]
PR_sep = [1.27, 1.34, 1.39, 1.38, 1.3, 1.36, 1.46, 1.33, 1.37, 1.27, 1.27, 1.29]
PR_sep_unc = [0.09, 0.05, 0.06, 0.06, 0.09, 0.05, 0.07, 0.03, 0.05, 0.03, 0.05, 0.07]

#At jmax = 1000
# Ts = [57.6011016, 62.1023003, 63.8959281, 65.9367659, 61.9449444, 62.9214533, 64.0519755, 60.1955110, 60.5262702, 59.4114021, 62.0635621, 59.5961519]
# sigmas = [0.16338840, 0.17788725, 0.16570998, 0.16102132, 0.19401663, 0.14327215, 0.15361382, 0.17375430, 0.17862829, 0.19656620, 0.21245939, 0.16766571]




#Including P-wind and Jmax = 600 : 
# B =   0.00133648 
# delta_B =-0.03712074
# zeta =  -0.30504896 
# Ts = [149.107701, 164.132291, 171.459785, 180.845204, 164.716622, 161.620226, 171.711267, 156.955738, 160.809469, 154.989294, 165.244363, 155.22443]
# sigmas = [0.17608069, 0.19488198, 0.18383431, 0.17904947, 0.21184813, 0.16204406, 0.17162014, 0.1900463, 0.19341891, 0.21389315, 0.23227593, 0.18328798]
# origins = [0.01799722, -0.01379366, 0.00602909, -0.01255523, 0.02292328, 0.00511943, 0.06467863, 0.01711942, 0.10205201, 0.07132686, 0.02814641, 0.07115158]

# #diff. init condition:

# B = 0.00716641 
# delta_B = -0.22060769 
# zeta=   -0.28300579 
# Ts = [26.0087408, 28.0541958, 28.8195137, 29.7400086, 27.988017, 28.4493686, 28.9159318, 27.1925374, 27.3247924, 26.8480177, 28.0843381, 26.9233065]
# sigmas = [0.16404873, 0.17819887, 0.16614902, 0.16149411, 0.19438546, 0.1435633, 0.15385787, 0.17390746, 0.17899859, 0.19682442, 0.21241429, 0.16796392]
# origins = [6.9908e-04, -0.03351346, -0.01441214, -0.03710527, 8.5599e-04, -0.01306664, 0.0465879, -8.7645e-05, 0.0879105, 0.05304292, 0.00728663, 0.05613493]



# Alto_fits_results = np.array([PR_sep,PR_sep_unc, Ts, sigmas, origins, sightlines]).T

# #sort from smallest to biggest PR_sep
# sorted_indices = np.lexsort((Alto_fits_results[:, 1], Alto_fits_results[:, 0]))

# Alto_fits_results = Alto_fits_results[sorted_indices].astype(float)

# #Alto_fits_results = Alto_fits_results[:2]

# plt.figure(figsize = (15,35))
# start = 0
# spacing = 0.07
# count = 12

# offset = np.arange(start, start + spacing * count, spacing)

# print(offset)
# for i, offset in enumerate(offset):  
#     T = Alto_fits_results[:,2][i]
#     sigma = Alto_fits_results[:,3][i]
#     origin = Alto_fits_results[:,4][i]
#     sightline = int(Alto_fits_results[:,5][i])

#     Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)
#     plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset, color = 'black') #, label = 'HD ' + str(sightline) , color = 'black')


#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'crimson', label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))

#     plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
#     plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     plt.title(f"B = {B:.4f} cm$^{{-1}}$  $\Delta$B = {delta_B:.2f}%  $\zeta$ = {zeta:.2f} cm$^{{-1}}$", fontsize=22)
    
#     xy = (Obs_data['Wavelength'][120] , Obs_data['Flux'][120] - offset)
#     plt.annotate('HD' + str(sightline), xy = xy , xytext = (3, Obs_data['Flux'][25] - offset + 0.0095), fontsize = 17 )
#     plt.annotate('T = {:.2f}'.format(T) + ' K', xy = xy , xytext = (-4.7, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + '  cm$^{-1}$', xy = xy , xytext = (-3.5, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17)
    
#     plt.axvline(x = -0.70, color = 'gray', linestyle = 'dotted', alpha = 0.5)
#     plt.axvline(x = -0.54, color = 'gray', linestyle = 'dotted', alpha = 0.5)
#     plt.axvline(x = 0.74, color = 'gray', linestyle = 'dotted', alpha = 0.5)
#     plt.axvline(x = 0.87, color = 'gray', linestyle = 'dotted', alpha = 0.5)
    
#     plt.xlim(-5,4.5)
    
# #plt.show()
# workdir = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/effect_of_Jmax/"
# save_plot_as = workdir + "Alto_fits_EDIBLES_6614_Jmax_300.pdf"
# plt.savefig(save_plot_as, format = 'pdf', bbox_inches="tight")

'''Cami 2004'''
# B = 0.00252688
# delta_B = -0.02816744
# zeta = -0.33395654
# Ts = [101.080285, 96.1085760, 100.637741, 102.787284, 102.310020, 87.7427099, 94.6884889]
# sigmas = [0.19084834, 0.17532329, 0.18244590, 0.18657923, 0.19612793, 0.19852667, 0.21209539]
# origins = [0.02652663, -0.02126270, -0.07778282, -0.05086524, 0.05448191, 0.06379547, 0.01083778]
# PR_sep = [1.527, 1.518, 1.507, 1.481, 1.386, 1.356, 1.330]

# Alto_fits_results = np.array([PR_sep, Ts, sigmas, origins, sightlines]).T
# sorted_indices = ( Alto_fits_results[:, 0].argsort())
# Alto_fits_results = Alto_fits_results[sorted_indices].astype(float)

# #Alto_fits_results = Alto_fits_results[:2]

# start = 0
# spacing = 0.07
# count = 7

# offset = np.arange(start, start + spacing * count, spacing)

# plt.figure(figsize = (15,35))
# for i, offset in enumerate(offset):  
#     T = Alto_fits_results[:,1][i]
#     sigma = Alto_fits_results[:,2][i]
#     origin = Alto_fits_results[:,3][i]
#     sightline = int(Alto_fits_results[:,4][i])

#     Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)
#     print(len(Obs_data['Wavelength']))
#     plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset, color = 'black') #, label = 'HD ' + str(sightline) , color = 'black')


#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'crimson', label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))

#     plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
#     plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     plt.title(f"B = {B:.4f} cm$^{{-1}}$  $\Delta$B = {delta_B:.2f}%  $\zeta$ = {zeta:.2f} cm$^{{-1}}$", fontsize=22)
    
#     xy = (Obs_data['Wavelength'][300] , Obs_data['Flux'][300] - offset)
   
#     plt.annotate('HD' + str(sightline), xy = xy , xytext = (2.5, model_data[:,1][5] - offset + 0.006), fontsize = 17 )
#     plt.annotate('T = {:.2f}'.format(T) + 'K', xy = xy , xytext = (-4.7, model_data[:,1][5] - offset + 0.005), fontsize = 17 )
#     plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + 'cm$^{-1}$', xy = xy , xytext = (-3, model_data[:,1][5] - offset + 0.005), fontsize = 17)
    
#     plt.axvline(x = -0.70, color = 'gray', linestyle = 'dotted', alpha = 0.5)
#     plt.axvline(x = -0.54, color = 'gray', linestyle = 'dotted', alpha = 0.5)
#     plt.axvline(x = 0.74, color = 'gray', linestyle = 'dotted', alpha = 0.5)
#     plt.axvline(x = 0.87, color = 'gray', linestyle = 'dotted', alpha = 0.5)
#     plt.xlim(-5,4.5)
    
# plt.savefig("Alto_fits_Cami_2004_6614.pdf", format = 'pdf', bbox_inches="tight")

############################################################################################
############################################################################################

#One sightline at a time


############################################################################################
############################################################################################

# results = pd.read_excel("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Fit results at correct resolution/min_redchi_data.xlsx")

# for i in range(len(results)):
    
#     print(i)
#     plt.figure(figsize = (12, 8))
    
#     B = results['B'][i]
#     delta_B = results['delta_B'][i]
#     zeta = results['zeta'][i]
#     T = results['T1'][i]
#     sigma = results['sigma1'][i]
#     origin = results['origin1'][i]
#     sightline = results['sightlines'][i]  
#     redchi = results['redchi'][i]
    
#     print('B = ' , B)
#     print('delta_B = ' , delta_B)
#     print('zeta = ' , zeta)
#     print('T = ', T)
#     print('sigma = ', sigma)
#     print('origin = ' , origin) 
    
#     Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)
#     plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'], color = 'black', label = 'HD' +str(sightline)) #, label = 'HD ' + str(sightline) , color = 'black')

#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     plt.plot(model_data[:,0], model_data[:,1], label = f"B = {B:.4f} cm$^{{-1}}$  $\Delta B =$ {delta_B:.2f}%  $\zeta^{{\prime}}  = ${zeta:.2f} cm$^{{-1}}$  \n T = {T:.2f} K $\sigma =$ {sigma:.3f} cm$^{{-1}}$ $\chi^2 = {redchi:.2f}$") # , color = '#1f77b4', label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))
    
#     #plt.title(f"B = {B:.4f} cm$^{{-1}}$  $\Delta B =$ {delta_B:.2f}%  $\zeta^{{\prime}}  = ${zeta:.2f} cm$^{{-1}}$  T = {T:.2f} K $\sigma =$ {sigma:.3f} cm$^{{-1}}$ $\chi^2 = {redchi:.2f}$", fontsize=22, pad = 10)
    
#     plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
#     plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
#     plt.legend(loc = 'lower right')
#     plt.xlim(-4, 4)
#     save_as_name  = str(sightline) + 'one_sl_fit_EDIBLES.pdf'
#     plt.savefig(save_as_name, format = 'pdf', bbox_inches="tight")

#     plt.show()
    





############################################################################################
############################################################################################

#Equi width calculation

############################################################################################
############################################################################################


def equivalent_width(wavelength, flux, continuum_level):
    """
    Calculate the equivalent width of a spectral line.

    Parameters:
    - wavelength: Array of wavelength values.
    - flux: Array of flux values corresponding to the wavelengths.
    - continuum_level: The continuum level to subtract from the flux.

    Returns:
    - Equivalent width of the spectral line.
    """

    # Subtract the continuum
    line_flux = continuum_level - flux

    # Integrate using the trapezoidal rule
    area = np.trapz(wavelength, line_flux)
   
    
    # Calculate the equivalent width
    EW = area / continuum_level
   
    return EW

# EWs = []
# for T, sigma, origin, sightline in zip(Ts, sigmas, origins, sightlines):
#         linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#         Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)
  
#         y_model = np.interp(Obs_data['Wavelength'] , model_data[:,0], model_data[:,1])
  
#         residual_y = Obs_data['Flux'] - y_model
#         residual_data = np.array([Obs_data['Wavelength'], residual_y]).transpose()
#         #residual_data = residual_data [(residual_data[:,0] >= -3.5) & (residual_data[:,0] <= -0.5)]
       
#         wavelength = residual_data[:,0] #Obs_data['Wavelength']
#         flux = residual_data[:,1] #residual_y
#         continuum_level = 1.0
      
#         print('Residuals:')
#         EW = equivalent_width(wavelength, flux, continuum_level)
#         print('EW is  ' , EW)  
#         EWs.append(EW) 


# print(EWs)
     

############################################################################################
############################################################################################

#Correlation calculation

############################################################################################
############################################################################################

sigmas = [0.18055114, 0.19760700, 0.18423023, 0.18244127, 0.21244470, 0.15821120, 0.17367824, 0.19180354, 0.20377079, 0.21578868, 0.24029108, 0.18563300]
bs=  [2.367146593, 1.789353453, 2.293734891, 1.925977983, 1.688799352, 1.634991188, 1.830094729, 2.515187171, 1.618003562, 2.418691161, 2.024811568, 2.338233138]



def calculate_pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Both lists should have the same length")

    correlation = np.corrcoef(x, y)[0, 1]
    return correlation


print("Correlation coefficient:", calculate_pearson_correlation(sigmas, bs))
# print("Correlation coefficient:", calculate_pearson_correlation(sigmas, PR_sep))
# print("Correlation coefficient:", calculate_pearson_correlation(sigmas, Ts))


    

    
# B = 0.00263686
# delta_B = -0.07094847
# zeta = -0.30450211
# Ts = [77.6051193, 84.3844641, 87.2852205, 93.3360293, 83.8822524, 79.5465432, 87.9420139, 80.2861379, 86.0165888, 79.6913374, 86.5009653, 79.6918109]
# sigmas = [0.17978137, 0.19617752, 0.18346550, 0.18017502, 0.21097759, 0.15886251, 0.17259367, 0.19082668, 0.20134184, 0.21524517, 0.23758045, 0.18463089]
# origins = [0.01890060, -0.01492546, 0.00137921, -0.01375697, 0.01816983, -0.00341084, 0.06238340, 0.01488341, 0.11077462, 0.06937389, 0.03062378, 0.06949553]

# EWs = []
# for T, sigma, origin, sightline in zip(Ts, sigmas, origins, sightlines):
#      linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#      Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)

#      y_model = np.interp(Obs_data['Wavelength'] , model_data[:,0], model_data[:,1]   )

#      residual_y = Obs_data['Flux'] - y_model
#      residual_data = np.array([Obs_data['Wavelength'], residual_y]).transpose()
#      #residual_data = residual_data [(residual_data[:,0] >= -3.5) & (residual_data[:,0] <= -0.5)]
     
#      # wavelength = residual_data[:,0] #Obs_data['Wavelength']
#      # flux = residual_data[:,1] #residual_y
#      continuum_level = 1.0
    
#      # print('Residuals:')
#      # print('area under curve is  ' + str(area_under_curve(wavelength, flux)))  
#      # print('EW is  ' + str(equivalent_width(wavelength, flux, continuum_level)))  
     

     
     
#      wavelength = Obs_data['Wavelength']
#      flux = y_model
       
#      print('Model:')
#      print('area under curve is  ' + str(area_under_curve(wavelength, flux)))  
#      print('EW is  ' + str(equivalent_width(wavelength, flux, continuum_level)))  
     
#      # wavelength = Obs_data['Wavelength']
#      # flux = Obs_data['Flux']
     
#      # print('Full:')
#      # print('area under curve is  ' + str(area_under_curve(wavelength, flux)))  
#      # print('EW is  ' + str(equivalent_width(wavelength, flux, continuum_level)))  
     

#      # plt.plot(wavelength, flux, label = sightline)
#      # plt.legend()
#      # plt.show()
#      EW = equivalent_width(wavelength, flux, continuum_level)
#      EWs.append(EW) 
     
# print(EWs)

# model_EWs = [-0.17857261439020172, -0.18579811631813803, -0.18340336176104957, -0.18435501375533628, -0.1892866739697753, -0.17325809758390145, -0.18045266239977248, -0.18272507314093656, -0.18776160186797575, -0.18839385536044073, -0.1962749895405297, -0.1808112875277313]
# residual_EWs = [-0.007017051110805382, -0.008246868460265493, -0.059847325821851254, -0.05190883401427335, -0.05531269723080303, -0.017223112870620734, -0.021397158338242028, -0.027349956321356138, -0.01841273588394792, -0.029436113404436706, -0.023259871166877052, -0.012111798172947844]

# import numpy as np

# def calculate_pearson_correlation(x, y):
#     if len(x) != len(y):
#         raise ValueError("Both lists should have the same length")

#     correlation = np.corrcoef(x, y)[0, 1]
#     return correlation

# T = [77.6051193, 84.3844641, 87.2852205, 93.3360293, 83.8822524, 79.5465432, 87.9420139, 80.2861379, 86.0165888, 79.6913374, 86.5009653, 79.6918109]

# print("Correlation coefficient:", calculate_pearson_correlation(model_EWs, residual_EWs))
# print("Correlation coefficient:", calculate_pearson_correlation(model_EWs, T))
# print("Correlation coefficient:", calculate_pearson_correlation(residual_EWs, T))



# B = 0.00263686
# delta_B = -0.07094847
# zeta = -0.30450211
# Ts = [77.6051193, 84.3844641, 87.2852205, 93.3360293, 83.8822524, 79.5465432, 87.9420139, 80.2861379, 86.0165888, 79.6913374, 86.5009653, 79.6918109]
# sigmas = [0.17978137, 0.19617752, 0.18346550, 0.18017502, 0.21097759, 0.15886251, 0.17259367, 0.19082668, 0.20134184, 0.21524517, 0.23758045, 0.18463089]
# origins = [0.01890060, -0.01492546, 0.00137921, -0.01375697, 0.01816983, -0.00341084, 0.06238340, 0.01488341, 0.11077462, 0.06937389, 0.03062378, 0.06949553]

# offset = np.arange(0, 12, 0.06)

# plt.figure(figsize = (15,30))
# for T, sigma, origin, offset, sightline in zip(Ts, sigmas, origins, offset, sightlines):
#     Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)
#     #plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset, color = 'black' ) #, label = 'HD ' + str(sightline) , color = 'black')


#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     #plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'red', label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))
#     y_model = np.interp(Obs_data['Wavelength'] , model_data[:,0], model_data[:,1]   )
#     plt.plot(Obs_data['Wavelength'], Obs_data['Flux'] - y_model - offset, label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma) , color = 'black')
#     plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
#     plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     # plt.annotate('HD' + str(sightline), xy = (Obs_data['Wavelength'][150] , Obs_data['Flux'][150] - offset) , xytext = (4, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     # plt.annotate('T = {:.2f}'.format(T) + ' K', xy = (Obs_data['Wavelength'][40] , Obs_data['Flux'][40] - offset) , xytext = (-7, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     #plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + '  cm$^{-1}$', xy = (Obs_data['Wavelength'][50] , Obs_data['Flux'][50] - offset) , xytext = (-5, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17)
    
    
#     title_text = ' B = {:.5f} cm-1, $\Delta$B = {:.4f}, $\zeta$ = {:.4f}'.format(B, delta_B,  zeta)
#     plt.title(title_text, fontsize = 22) 
#     plt.xlim(-7.5, 6)
#     # plt.legend(loc = 'lower left', fontsize = 16)
#     plt.legend(bbox_to_anchor=(1.8, 0.7), loc='lower right', fontsize = 22)
    
# plt.savefig("hot_band_residuals.pdf", format = 'pdf', bbox_inches="tight")

    

# B = 0.00336
# delta_B = -0.17
# zetas=  (-1, -0.5, 0, 0.5, 1)
# T = 61.2
# sigma = 0.1953
# origin = 0

# for zeta in zetas:
#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     linelist['intensities'] = 1-0.1*(linelist['intensities']/max(linelist['intensities']))
#     plt.stem(linelist['wavenos'], linelist['intensities'], label = 'zeta = ' + str(zeta), bottom = 1)
    
#     plt.plot(model_data[:,0], model_data[:,1], color = 'red', label = 'Model')
#     plt.title('kerr condition c')
#     plt.legend()
#     plt.show()
    



#reviewing 3 init conditions one slightline at a time fit results
# for sightline in sightlines: 
#     fit_report = pd.read_csv('/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/{}3_init_conditions_Cami_2004.csv'.format(sightline))
#     for i in range(len(fit_report['B'])):

#         plt.figure(figsize = (15,8))
    
#         B = fit_report['B'][i]
#         delta_B = fit_report['delta_B'][i] 
#         zeta = fit_report['zeta'][i]
#         T = fit_report['T1'][i]
#         sigma = fit_report['sigma1'][i]
#         origin = fit_report['origin1'][i]
        
#         linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#         plt.plot(model_data[:,0], model_data[:,1], color = 'red', label = 'Model')
    
    
#         Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
#         plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'],  label = 'Data (HD ' + str(sightline) + ')' , color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
#         title_text = ' B = {:.4f} cm-1, Delta_B = {:.4f}, zeta = {:.4f}, Temperature = {:.4f} K, $\sigma$ = {:.4f}, origin = {:.4f}'.format(B, delta_B,  zeta, T, sigma, origin)
#         plt.title(title_text, fontsize = 17) 
    
#         plt.legend(fontsize = 15)
#         plt.show()
    

#altogether init 1
# B = 0.00258792
# delta_B = -0.02696335
# zeta = -0.32505451
# Ts = [95.6801784, 92.6800681, 101.434804, 106.927171, 92.9660592, 81.9853734, 88.8462821]
# sigmas = [0.18544432, 0.17102981, 0.18105244, 0.18667330, 0.18745625, 0.19205089, 0.20584935]
# origins = [0.03058103, -0.01568777, -0.07118868, -0.04417317, 0.05996478, 0.06902381, 0.01248981]

#altogether init 2
# B = 0.00256618
# delta_B = -0.0270427
# zeta = -0.32709176
# Ts = [97.3630470, 94.1524257, 103.356815, 109.216188, 94.6232728, 83.1577894, 90.3789475]
# sigmas = [0.18683958, 0.17238475, 0.18252429, 0.18832571, 0.18890352, 0.19325940, 0.20730193]
# origins = [0.03082452, -0.01550851, -0.07090288, -0.04426105, 0.06003777, 0.06930614, 0.01284413]

#altogether init 3
# B = 0.00358824
# delta_B = -0.03592810
# zeta = -0.30557765
# Ts = [64.2055690, 64.1357596, 68.4328938, 70.4409091, 61.6433262, 56.3053709, 59.2680825]
# sigmas = [0.16881947, 0.15561425, 0.16380323, 0.16631262, 0.16997689, 0.17793917, 0.18717446]
# origins = [0.02773232, -0.01778257, -0.07371238, -0.04116312, 0.05825675, 0.06593325, 0.00224110]

#correct resolution: 
    
# B = 0.00263686
# delta_B = -0.07094847
# zeta = -0.30450211
# Ts = [77.6051193, 84.3844641, 87.2852205, 93.3360293, 83.8822524, 79.5465432, 87.9420139, 80.2861379, 86.0165888, 79.6913374, 86.5009653, 79.6918109]
# sigmas = [0.17978137, 0.19617752, 0.18346550, 0.18017502, 0.21097759, 0.15886251, 0.17259367, 0.19082668, 0.20134184, 0.21524517, 0.23758045, 0.18463089]
# origins = [0.01890060, -0.01492546, 0.00137921, -0.01375697, 0.01816983, -0.00341084, 0.06238340, 0.01488341, 0.11077462, 0.06937389, 0.03062378, 0.06949553]

# offset = np.arange(0, 12, 0.06)

# B = 0.00247690
# delta_B = -0.06864414
# zeta = -0.31136112

# Ts = [84.8194157, 95.5401280, 97.2287829, 116.540106, 99.2097303, 86.6632524, 98.4034623, 89.0236501, 97.3496743, 87.8768841, 103.793851, 86.2283890]

# sigmas = [0.18496609, 0.20292189, 0.18936606, 0.19319891, 0.22004922, 0.16349391, 0.17929137, 0.19690720, 0.20881159, 0.22038457, 0.24982774, 0.18963846]

# origins = [0.02918830, -0.00869084, 0.01065963, -0.00253786, 0.02965486, -0.00276476, 0.06965812, 0.02549114, 0.12524227, 0.07957677, 0.04544019, 0.08117755]

# plt.figure(figsize = (15,30))
# for T, sigma, origin, offset, sightline in zip(Ts, sigmas, origins, offset, sightlines):
#     Obs_data, Obs_y_data_to_fit, std_dev= obs_curve_to_fit(sightline)
#     #plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset, color = 'black' ) #, label = 'HD ' + str(sightline) , color = 'black')


#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     #plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'red', label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))
    
#     plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
#     plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     # plt.annotate('HD' + str(sightline), xy = (Obs_data['Wavelength'][150] , Obs_data['Flux'][150] - offset) , xytext = (4, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     # plt.annotate('T = {:.2f}'.format(T) + ' K', xy = (Obs_data['Wavelength'][40] , Obs_data['Flux'][40] - offset) , xytext = (-7, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     #plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + '  cm$^{-1}$', xy = (Obs_data['Wavelength'][50] , Obs_data['Flux'][50] - offset) , xytext = (-5, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17)
    
    
#     title_text = ' B = {:.5f} cm-1, $\Delta$B = {:.4f}, $\zeta$ = {:.4f}'.format(B, delta_B,  zeta)
#     plt.title(title_text, fontsize = 22) 
#     plt.xlim(-7.5, 6)
#     # plt.legend(loc = 'lower left', fontsize = 16)
#     plt.legend(bbox_to_anchor=(1.8, 0.7), loc='lower right', fontsize = 22)
    

# B=       0.00701776 
# delta_B= -0.04991853 
# zeta=   -0.29885821 
# T=       33.3047616 
# sigma=   0.16767868 
# origin=  0.02034466 
# linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
# plt.plot(model_data[:,0], model_data[:,1], color = 'red', label = 'Model')


# Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
# plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'],  label = 'Data (HD ' + str(sightline) + ')' , color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))





# plt.figure(figsize = (15,8))
# B =       0.00330308 #+/- 8.7988e-05 (3.54%) (init = 0.0023)		
# delta_B =  -0.026643322 #+/- 0.00301885 (4.41%) (init = -0.0353)		
# zeta =  -0.11860631 #+/- 0.00953055 (3.05%) (init = -0.4197)		

# Ts = [86.19, 84.64, 93.08, 96.69, 82.35, 75.69, 79.07]
# origins =  [0.034, 0.0033, -0.052, -0.0197, 0.070, 0.016]
# sigma = 0.0289 #, 0.11, 0.20]

# for T,  origin in zip(Ts,  origins):
#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     plt.plot(model_data[:,0], model_data[:,1], label = sigma)
#     plt.legend()


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
    

# def fwhm(x, y):
#     """
#     Compute Full Width at Half Maximum (FWHM) of a peak in y.
#     x and y are arrays of the x and y data of the peak.
#     """
#     half_max = max(y) / 2.
#     # indices of points above half max
#     indices = np.where(y > half_max)[0]
    
#     # In case the peak is not well-defined or other data irregularities
#     if len(indices) == 0:
#         return None
    
    
#     # Width in x of data above half max
#     return x[indices[2]] - x[indices[1]]

# for sightline in sightlines: 
    
#     file = filename.format(sightline)
#     # Obs_data = pd.read_csv(spec_dir / file,
#     #                         delim_whitespace=(True))
    
#     Obs_data = pd.read_csv(spec_dir / file,
#                             sep = ',')
#     Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9

#     plt.plot(Obs_data['Wavelength'], Obs_data['Flux'])

#     Obs_data_trp = Obs_data[(Obs_data['Flux'] <= 0.95)]  # trp = triple peak 
#     #print(max(Obs_data_trp['Wavelength']))
#     plt.plot(Obs_data_trp['Wavelength'], Obs_data_trp['Flux'])
#     print('============')
    
    
#     Obs_data_new = Obs_data[(Obs_data['Wavelength'] >= 6612.8) & (Obs_data['Wavelength']<= 6614.0)]
#     plt.plot(Obs_data_new['Wavelength'], Obs_data_new['Flux'], color = 'green')

#     delta_lambda = fwhm(Obs_data['Wavelength'], Obs_data['Flux'])
#     print("Estimated resolution (FWHM):", delta_lambda, "nm")
    
#     central_lambda = Obs_data['Wavelength'][np.argmin(Obs_data['Flux'])]
#     print(central_lambda)
#     resolution_R = central_lambda / delta_lambda
#     print(resolution_R)
#     plt.show()


#Default blue = ##1f77b4


# h = const.h.cgs.value
# c = const.c.to('cm/s').value
# k = const.k_B.cgs.value

# B = 0.0005
# T = 500
# J_peak = np.sqrt(k*T/(2*h*c*B)) -1/2

# print('J_peak = ' , J_peak)