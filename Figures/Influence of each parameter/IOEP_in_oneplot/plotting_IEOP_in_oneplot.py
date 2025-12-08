#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:13:34 2024

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




'''Inputs'''    
Jmax = 1000
combinations  = allowed_perperndicular_transitions(Jmax) 

'''For finer binning, we have chosen 1000 points between -10 and 10 wavenumbers. 
smooth_wavenos = np.linspace(-10, 10,  1000) 
While exporting the whole wavelength range, this is not good trim the x axis '''

# T = 61.2
# B = 0.0501187
# delta_B = -0.17
# delta_C = -0.17
# origin = 0
# zeta = -0.49
# sigma = 0.1953


# linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
# plt.scatter(model_data[:,0], model_data[:,1])
# plt.xlim(3, -3)

plot_B_study = True
plot_delta_B_study = False
plot_zeta_study = False
plot_T_study = False
plot_sigma_study = False

#########################################################
'''============== Parameter Study : B  ================'''
#########################################################

if plot_B_study == True:

    #Ts = (2.7, 10, 30, 70, 100)
    delta_Bs = (0, -0.2, -0.4, -0.6, -0.8)
    ground_Bs = (0.000501, 0.001584, 0.005011, 0.0158489, 0.0501187)
    #zetas = (-1, -0.5, 0, 0.5, 1)
    sigmas  = (0.05, 0.11, 0.17, 0.23, 0.3)
    
    
    
    
    T = 61.2
    B = 0.00336
    delta_B = -0.17
    delta_C = -0.17
    origin = 0
    zeta = -0.49
    sigma = 0.1953
    
    
    plt.figure(figsize=(3.5, 2.3))
    colours = sns.color_palette("flare", len(sigmas))
    colours = ['red', 'darkorange',  'darkgreen', 'dodgerblue', 'purple'  ]
    B_label = ('5.0x10$^{-4}$', '1.6x10$^{-3}$', '5.0x10$^{-3}$', '1.6x10$^{-2}$', '5.0x10$^{-2}$' )

    
    for i, B in enumerate(ground_Bs): 
        
        # linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
        # np.savetxt(f"IOEP_B_model{i+1}.csv", model_data, delimiter=',')

        model_data = pd.read_csv(f'IOEP_B_model{i+1}.csv')

        plt.plot(model_data.iloc[:,0], model_data.iloc[:,1], linewidth = 1, 
         label = '{}'.format(B_label[i])      , color=colours[i]) #,label = 'B = ' + str(B) + ' cm$^{-1}$'
        plt.axhline(y=1, linestyle = '--', color = 'gray', linewidth = 0.5)
        plt.axvline(x=0, linestyle = '--', color = 'gray', linewidth = 0.5)
        
    
    # Customize tick labels and reduce the number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on x-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis
    
    # Increase tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Add minor ticks
    plt.minorticks_on()
    plt.tick_params(which='both', width=1, direction = 'in')
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='gray')
      
    plt.xlim(6.5, -6.8)
    plt.ylim(0.83, 1.01)
    plt.ylabel('Intensity', rotation=90, labelpad=0, fontsize = 8)
    plt.xlabel('Wavenumber (cm$^{-1}$)', labelpad =0, fontsize = 8)
    #plt.title(' 'r'$\Delta B =$ ' + str(delta_B) + '%   'r'$\zeta^{\prime}  = $' + str(zeta) +' cm$^{-1}$   T = ' + str(T) + ' K   'r'$\sigma = $'+ str(sigma) + ' cm$^{-1}$ ' , 
              #fontsize =20, y =1.05)


    #plotting observed profile
    spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
    filename = '6614_HD185418.txt'
    Obs_data = pd.read_csv(spec_dir / filename,
                            sep = ',')
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    Obs_data = Obs_data.iloc[::-1].reset_index(
        drop=True)
    # shifting to 0 and scaling flux between 0.9 and 1
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] #+ 6614
    
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >=2) & (Obs_data['Wavelength']<= 5)]
    std_dev = np.std(Obs_data_continuum['Flux'])
    
     
    plt.errorbar( Obs_data['Wavelength'], Obs_data['Flux'] - 0.05, yerr=std_dev, linewidth = 1, color = 'black', label = 'HD 185418')
    legend = plt.legend(title = 'B (cm$^{-1}$)',  loc='lower right', fontsize= 7, frameon = False)
    legend.get_title().set_fontsize(7)


    plt.annotate('(a)', xy=(0.2, 0.25), xycoords='figure fraction', fontsize=9)
    

    plt.savefig('parameter_study_of_B.pdf' , format = 'pdf', bbox_inches = 'tight')
    plt.show()


#########################################################
'''============== Parameter Study : delta_B  ================'''
#########################################################

if plot_delta_B_study == True:

    Ts = (2.7, 10, 30, 70, 100)
    delta_Bs = (0, -0.2, -0.4, -0.6, -0.8)
    ground_Bs = (0.000501, 0.001584, 0.005011, 0.0158489, 0.0501187)
    #zetas = (-1, -0.5, 0, 0.5, 1)
    sigmas  = (0.05, 0.11, 0.17, 0.23, 0.3)
    
    T = 61.2
    B = 0.00336
    delta_B = -0.17
    delta_C = -0.17
    origin = 0
    zeta = -0.49
    sigma = 0.1953
    
    
    plt.figure(figsize=(3.5, 2.3))
    colours = ['red', 'darkorange',  'darkgreen', 'dodgerblue', 'purple'  ]
    
    for i, delta_B in enumerate(delta_Bs): 
        
        # linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
        # np.savetxt(f"IOEP_deltaB_model{i+1}.csv", model_data, delimiter=',')

        model_data = pd.read_csv(f'IOEP_deltaB_model{i+1}.csv')

        plt.plot(model_data.iloc[:,0], model_data.iloc[:,1],  linewidth = 1,
                label =  str(delta_B) + '% ', color=colours[i]) #, label = str(delta_B))
        plt.axhline(y=1, linestyle = '--', color = 'gray', linewidth = 0.5)
        plt.axvline(x=0, linestyle = '--', color = 'gray', linewidth = 0.5)
        
    
    # Customize tick labels and reduce the number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))  # Fewer ticks on x-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis
    
    # Increase tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Add minor ticks
    plt.minorticks_on()
    plt.tick_params(which='both', width=1, direction = 'in')
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='gray')
    plt.xlim(2.5, -3.2)
    
    
    plt.xlabel('Wavenumber (cm$^{-1}$)', labelpad =0, fontsize = 8)
    plt.ylabel('Intensity', labelpad =2, fontsize = 8)

    #plt.title(' 'r'$\Delta B =$ ' + str(delta_B) + '%   'r'$\zeta^{\prime}  = $' + str(zeta) +' cm$^{-1}$   T = ' + str(T) + ' K   'r'$\sigma = $'+ str(sigma) + ' cm$^{-1}$ ' , 
              #fontsize =20, y =1.05)


    #plotting observed profile
    spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
    filename = '6614_HD185418.txt'
    Obs_data = pd.read_csv(spec_dir / filename,
                            sep = ',')
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    Obs_data = Obs_data.iloc[::-1].reset_index(
        drop=True)
    # shifting to 0 and scaling flux between 0.9 and 1
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] #+ 6614
    
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >=2) & (Obs_data['Wavelength']<= 5)]
    std_dev = np.std(Obs_data_continuum['Flux'])
    
     
    plt.errorbar( Obs_data['Wavelength'], Obs_data['Flux'] - 0.03, yerr=std_dev, linewidth = 1, color = 'black', label = 'HD 185418')
    legend = plt.legend(title = '$\Delta$B (cm$^{-1}$)',  loc='lower right', fontsize= 7, frameon = False)
    legend.get_title().set_fontsize(7)


    plt.annotate('(b)', xy=(0.2, 0.25), xycoords='figure fraction', fontsize=9)
    


    plt.savefig('parameter_study_of_delta_B.pdf' , format = 'pdf', bbox_inches = 'tight')
    plt.show()
 

#########################################################
'''============== Parameter Study : zeta  ================'''
#########################################################

if plot_zeta_study == True:

    Ts = (2.7, 10, 30, 70, 100)
    delta_Bs = (0, -0.2, -0.4, -0.6, -0.8)
    ground_Bs = (0.000501, 0.001584, 0.005011, 0.0158489, 0.0501187)
    zetas = (-1, -0.5, 0, 0.5, 1)
    sigmas  = (0.05, 0.11, 0.17, 0.23, 0.3)
    
    T = 61.2
    B = 0.00336
    delta_B = -0.17
    delta_C = -0.17
    origin = 0
    zeta = -0.49
    sigma = 0.1953
    
    
    plt.figure(figsize=(3.5, 2.3))
    colours = ['red', 'darkorange',  'darkgreen', 'dodgerblue', 'purple'  ]
    
    for i, zeta in enumerate(zetas): 
        
        # linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
        # np.savetxt(f"IOEP_zeta_model{i+1}.csv", model_data, delimiter=',')

        model_data = pd.read_csv(f'IOEP_zeta_model{i+1}.csv')

        plt.plot(model_data.iloc[:,0], model_data.iloc[:,1],  linewidth = 1, 
                label =  str(zeta), color=colours[i]) #, label = str(delta_B))
        plt.axhline(y=1, linestyle = '--', color = 'gray', linewidth = 0.5)
        plt.axvline(x=0, linestyle = '--', color = 'gray', linewidth = 0.5)
        
    
    # Customize tick labels and reduce the number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))  # Fewer ticks on x-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis
    
    # Increase tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Add minor ticks
    plt.minorticks_on()
    plt.tick_params(which='both', width=1, direction = 'in')
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='gray')
      
    plt.xlim(3, -3.5)
    plt.xlabel('Wavenumber (cm$^{-1}$)', labelpad =0, fontsize = 8)
    plt.ylabel('Intensity', labelpad =2, fontsize = 8)

    #plt.title(' 'r'$\Delta B =$ ' + str(delta_B) + '%   'r'$\zeta^{\prime}  = $' + str(zeta) +' cm$^{-1}$   T = ' + str(T) + ' K   'r'$\sigma = $'+ str(sigma) + ' cm$^{-1}$ ' , 
              #fontsize =20, y =1.05)


    #plotting observed profile
    spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
    filename = '6614_HD185418.txt'
    Obs_data = pd.read_csv(spec_dir / filename,
                            sep = ',')
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    Obs_data = Obs_data.iloc[::-1].reset_index(
        drop=True)
    # shifting to 0 and scaling flux between 0.9 and 1
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] #+ 6614
    
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >=2) & (Obs_data['Wavelength']<= 5)]
    std_dev = np.std(Obs_data_continuum['Flux'])
    
     
    plt.errorbar( Obs_data['Wavelength'], Obs_data['Flux'] - 0.03, yerr=std_dev, linewidth = 1, color = 'black', label = 'HD 185418')
    legend = plt.legend(title = '$\zeta$ (cm$^{-1}$)',  loc='lower right', fontsize= 7, frameon = False)
    legend.get_title().set_fontsize(7)


    plt.annotate('(c)', xy=(0.2, 0.25), xycoords='figure fraction', fontsize=9)
    

    plt.savefig('parameter_study_of_zeta.pdf' , format = 'pdf', bbox_inches = 'tight')
    plt.show()
    

#########################################################
'''============== Parameter Study : T  ================'''
#########################################################

if plot_T_study == True:

    Ts = (2.7, 10, 30, 70, 100)
    delta_Bs = (0, -0.2, -0.4, -0.6, -0.8)
    ground_Bs = (0.000501, 0.001584, 0.005011, 0.0158489, 0.0501187)
    #zetas = (-1, -0.5, 0, 0.5, 1)
    sigmas  = (0.05, 0.11, 0.17, 0.23, 0.3)
    
    T = 61.2
    B = 0.00336
    delta_B = -0.17
    delta_C = -0.17
    origin = 0
    zeta = -0.49
    sigma = 0.1953
    
    
    plt.figure(figsize=(3.5, 2.3))
    colours = ['red', 'darkorange',  'darkgreen', 'dodgerblue', 'purple'  ]
    
    for i, T in enumerate(Ts):
        
        # linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
        # np.savetxt(f"IOEP_T_model{i+1}.csv", model_data, delimiter=',')

        model_data = pd.read_csv(f'IOEP_T_model{i+1}.csv')


        plt.plot(model_data.iloc[:,0], model_data.iloc[:,1], linewidth = 1, 
                label = str(T), color=colours[i]) #, label = str(delta_B))
        plt.axhline(y=1, linestyle = '--', color = 'gray', linewidth = 0.5)
        plt.axvline(x=0, linestyle = '--', color = 'gray', linewidth = 0.5)
        
    
    # Customize tick labels and reduce the number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on x-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis
    
    # Increase tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Add minor ticks
    plt.minorticks_on()
    plt.tick_params(which='both', width=1, direction = 'in')
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='gray')
       
    plt.xlim(2, -2.9)
    plt.xlabel('Wavenumber (cm$^{-1}$)', labelpad =0, fontsize = 8)
    plt.ylabel('Intensity', labelpad =2, fontsize = 8)

    #plt.title(' 'r'$\Delta B =$ ' + str(delta_B) + '%   'r'$\zeta^{\prime}  = $' + str(zeta) +' cm$^{-1}$   T = ' + str(T) + ' K   'r'$\sigma = $'+ str(sigma) + ' cm$^{-1}$ ' , 
              #fontsize =20, y =1.05)


    #plotting observed profile
    spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
    filename = '6614_HD185418.txt'
    Obs_data = pd.read_csv(spec_dir / filename,
                            sep = ',')
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    Obs_data = Obs_data.iloc[::-1].reset_index(
        drop=True)
    # shifting to 0 and scaling flux between 0.9 and 1
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] #+ 6614
    
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >=2) & (Obs_data['Wavelength']<= 5)]
    std_dev = np.std(Obs_data_continuum['Flux'])
    
     
    plt.errorbar( Obs_data['Wavelength'], Obs_data['Flux'] - 0.03, yerr=std_dev, linewidth = 1, color = 'black', label = 'HD 185418')
    legend = plt.legend(title = 'T (K)',  loc='lower right', fontsize= 7, frameon = False)
    legend.get_title().set_fontsize(7)


    plt.annotate('(d)', xy=(0.2, 0.25), xycoords='figure fraction', fontsize=9)
    



    plt.savefig('parameter_study_of_T.pdf' , format = 'pdf', bbox_inches = 'tight')
    plt.show()

#########################################################
'''============== Parameter Study : Sigma ================'''
#########################################################

if plot_sigma_study == True:

#Ts = (2.7, 10, 30, 70, 100)
    delta_Bs = (0, -0.2, -0.4, -0.6, -0.8)
    ground_Bs = (0.000501, 0.001584, 0.005011, 0.0158489, 0.0501187)
    #zetas = (-1, -0.5, 0, 0.5, 1)
    sigmas  = (0.05, 0.11, 0.17, 0.23, 0.3)
    
    T = 61.2
    B = 0.00336
    delta_B = -0.17
    delta_C = -0.17
    origin = 0
    zeta = -0.49
    sigma = 0.1953
    
    
    plt.figure(figsize=(3.5, 2.3))
    colours = ['red', 'darkorange',  'darkgreen', 'dodgerblue', 'purple'  ]
    
    
    
    for i, sigma in enumerate(sigmas): 
        
        
        # linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
        # np.savetxt(f"IOEP_sigma_model{i+1}.csv", model_data, delimiter=',')

        model_data = pd.read_csv(f'IOEP_sigma_model{i+1}.csv')

        
        plt.plot(model_data.iloc[:,0], model_data.iloc[:,1], linewidth = 1, 
                 label =  str(sigma), color=colours[i]) #, label = str(delta_B))
        plt.axhline(y=1, linestyle = '--', color = 'gray', linewidth = 0.5)
        plt.axvline(x=0, linestyle = '--', color = 'gray', linewidth = 0.5)
        
    
    # Customize tick labels and reduce the number of ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on x-axis
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis
    
    # Increase tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # Add minor ticks
    plt.minorticks_on()
    plt.tick_params(which='both', width=1, direction = 'in')
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='gray')
      
    plt.xlim(3, -3.5)
    plt.xlabel('Wavenumber (cm$^{-1}$)', labelpad =0, fontsize = 8)
    plt.ylabel('Intensity', labelpad =2, fontsize = 8)

    #plt.title(' 'r'$\Delta B =$ ' + str(delta_B) + '%   'r'$\zeta^{\prime}  = $' + str(zeta) +' cm$^{-1}$   T = ' + str(T) + ' K   'r'$\sigma = $'+ str(sigma) + ' cm$^{-1}$ ' , 
              #fontsize =20, y =1.05)


    #plotting observed profile
    spec_dir = Path("/home/charmibhatt/Science/DIBs/Data/Heather's_data")
    filename = '6614_HD185418.txt'
    Obs_data = pd.read_csv(spec_dir / filename,
                            sep = ',')
    Obs_data['Wavelength'] = (1 / Obs_data['Wavelength']) * 1e8
    Obs_data = Obs_data.iloc[::-1].reset_index(
        drop=True)
    # shifting to 0 and scaling flux between 0.9 and 1
    min_index = np.argmin(Obs_data['Flux'])
    Obs_data['Wavelength'] = Obs_data['Wavelength'] - Obs_data['Wavelength'][min_index] #+ 6614
    
    Obs_data['Flux'] = (Obs_data['Flux'] - min(Obs_data['Flux'])) / (1 - min(Obs_data['Flux'])) * 0.1 + 0.9
    Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >=2) & (Obs_data['Wavelength']<= 5)]
    std_dev = np.std(Obs_data_continuum['Flux'])
    
     
    plt.errorbar( Obs_data['Wavelength'], Obs_data['Flux'] - 0.03, yerr=std_dev, linewidth = 1, color = 'black', label = 'HD 185418')
    legend = plt.legend(title = '$\sigma$ (cm$^{-1}$)',  loc='lower right', fontsize= 7, frameon = False)
    legend.get_title().set_fontsize(7)


    plt.annotate('(e)', xy=(0.2, 0.25), xycoords='figure fraction', fontsize=9)
    



    plt.savefig('parameter_study_of_sigma.pdf' , format = 'pdf', bbox_inches = 'tight')
    plt.show()



    
