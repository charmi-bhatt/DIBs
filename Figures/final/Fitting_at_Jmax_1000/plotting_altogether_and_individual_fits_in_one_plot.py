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

    #start1 = timeit.default_timer()

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
    
    smooth_wavenos = np.linspace(np.min(linelist['wavenos']) - 1, np.max(linelist['wavenos']) + 1, 10000)  # grid_size
    smooth_wavenos = np.linspace(-10, 10,  1000)  # grid_size



    Wavenos_arr = np.array(linelist['wavenos'])
    Intenisty_arr = np.array(linelist['intensities'])


    #@nb.njit(parallel=True)
    def calculate_smooth_intensities(wavenos, intensities, smooth_wavenos, sigma):
        smooth_intensities = np.zeros(smooth_wavenos.shape)

        for i in range(len(smooth_wavenos)):
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

    
    #end4 = timeit.default_timer()
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

#EDIBLES data
spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Data/Heather's_data")
filename = '6614_HD{}.txt'
sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']

'''Inputs'''    
Jmax = 1000
combinations  = allowed_perperndicular_transitions(Jmax) 

spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
filename = '6614_HD{}.txt'

############################################################################################
############################################################################################

#Altogether Fits Plotting

############################################################################################
############################################################################################



PR_sep = [1.27, 1.34, 1.39, 1.38, 1.3, 1.36, 1.46, 1.33, 1.37, 1.27, 1.27, 1.29]
PR_sep_unc = [0.09, 0.05, 0.06, 0.06, 0.09, 0.05, 0.07, 0.03, 0.05, 0.03, 0.05, 0.07]



'''Now, Jmax = 1000 and EDIBLES resolution'''
B=0.0022
delta_B= -0.07
zeta = -0.289
Ts = [81.18658, 87.365539, 89.9437666, 92.4637866, 87.0832607, 88.3588874, 90.2125208, 84.7285498, 85.168429, 83.6203956, 87.5451781, 83.7653075]
sigmas = [0.16286939, 0.17714702, 0.16487818, 0.16119119, 0.19473806, 0.14253173, 0.1532089, 0.1730355, 0.17778417, 0.19593083, 0.21270658, 0.16688067]
origins = [0.01108004, -0.02271621, -0.0045765, -0.02639547, 0.01160903, -0.00360227, 0.05564258, 0.01000857, 0.09846675, 0.06436222, 0.01817963, 0.06677305]




Alto_fits_results = np.array([PR_sep,PR_sep_unc, Ts, sigmas, origins, sightlines]).T

#sort from smallest to biggest PR_sep
sorted_indices = np.lexsort((Alto_fits_results[:, 1], Alto_fits_results[:, 0]))

Alto_fits_results = Alto_fits_results[sorted_indices].astype(float)

#Alto_fits_results = Alto_fits_results[:2]

plt.figure(figsize = (3.5, 10))
start = 0
spacing = 0.07
count = 12

offset = np.arange(start, start + spacing * count, spacing)

print('offset', offset)

y_values = []  # List to store y positions
labels = []  # List to store labels

print(Alto_fits_results[:,5])


sorted_sightlines = ['185418', '185859',  '23180', '203532', '147683', '170740',  '24398', '149757', '184915' , '147165',  '144470',  '166937']



for i, offset in enumerate(offset):  
    
    print(i)

    #Data --------------------------------------------------
    sightline  = sorted_sightlines[i]
    file = filename.format(sightline)

    Obs_data = pd.read_csv(spec_dir / file,
                            sep = ',')
    
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index]   
    
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    
    #print(Obs_data['Wavelength'] , Obs_data['Flux'])

    plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset, color = 'black', linewidth = 1) #, label = 'HD ' + str(sightline) , color = 'black')

    #---------------------------------------------------------


    #Individual model --------------------------------------------------
   
    filepath ='/home/charmibhatt/Science/DIBs/Figures/final/Fitting_at_Jmax_1000/Individual_at-maxJ_1000/'
    individual_model_filename = f'{filepath}Individual_best_fit_model_{sorted_sightlines[i]}.csv'
   
    
    indi_model_data = pd.read_csv(individual_model_filename, sep = ',')
   
   
    plt.plot(indi_model_data.iloc[:,0], indi_model_data.iloc[:,1] - offset, color = 'blue', linewidth = 1) #, label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))
    
    #---------------------------------------------------------




    #Altogether model --------------------------------------------------

    filepath ='/home/charmibhatt/Science/DIBs/Figures/final/Fitting_at_Jmax_1000/Altogther_at_maxJ_1000/'
    alto_model_filename = f'{filepath}Altogether_best_fit_model_{sorted_sightlines[i]}.csv'
   
    
    alto_model_data = pd.read_csv(alto_model_filename, sep = ',')
    T = Alto_fits_results[:,2][i]
    sigma = Alto_fits_results[:,3][i]
    sightline = int(Alto_fits_results[:,5][i])

    plt.plot(alto_model_data.iloc[:,0], alto_model_data.iloc[:,1] - offset, color = 'crimson', 
             label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma), linewidth = 1)
    
    
    #---------------------------------------------------------

    
    
    
    
    plt.xlabel('Wavenumber (cm$^{-1}$)', labelpad = 2, fontsize = 9)
    plt.ylabel('Normalized Intenisty - offset', labelpad = 2, fontsize = 9)
    plt.tick_params(axis='both', which='major', labelsize=9,  direction = 'in')
    plt.title(f"B = {B:.4f} cm$^{{-1}}$  $\Delta$B = {delta_B:.2f}%  $\zeta$ = {zeta:.2f} cm$^{{-1}}$", fontsize=9)
    
    xy = (Obs_data['Wavelength'][110] , Obs_data['Flux'][120] - offset)
    
    
    

    # plt.annotate('HD ' + str(sightline), xy = xy , xytext = (3, alto_model_data.iloc[:,1][5] - offset + 0.007), fontsize = 9)
    
    
    # plt.annotate('T = {:.2f}'.format(T) + 'K', xy = xy , xytext = (-1.7, alto_model_data.iloc[:,1][5] - offset + 0.007), fontsize = 9)
    # plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + 'cm$^{-1}$', xy = xy , xytext = (-3.2, alto_model_data.iloc[:,1][5] - offset + 0.007), fontsize = 9)
   
    
       # Add the first annotation
    plt.annotate(
        'T = {:.2f}'.format(T) + 'K',
        (-3.2, alto_model_data.iloc[:,1][5] - offset + 0.027),  # Replace x with the desired x-coordinate
        xycoords='data',  # Assuming you're working with data coordinates
        fontsize=9,
        ha='center')
    
    plt.annotate(
        f"$\\sigma={sigma:.2f}\\,\\mathrm{{cm}}^{{-1}}$",       
        (-3.5, alto_model_data.iloc[:,1][5] - offset + 0.005),  # Replace x with the desired x-coordinate
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
    plt.ylim(0.12, 1.05)
    
# save_here = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Fitting_at_Jmax_1000/Altogther_at_maxJ_1000/"
# save_plot_as = save_here + "Alto_fits_EDIBLES_maxJ_1000.pdf"

plt.savefig('Altogether_and_individual_fits_in_one_plot.pdf', format = 'pdf', bbox_inches="tight")

plt.show()
