#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:30:46 2023

@author: charmibhatt
"""
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


report = """
# fitting method   = leastsq
    # function evals   = 394
    # data points      = 700
    # variables        = 17
    chi-square         = 14924.9044
    reduced chi-square = 21.8519830
    Akaike info crit   = 2175.79434
    Bayesian info crit = 2253.16270
    R-squared          = -148441.550
[[Variables]]
    B:        0.00330374 +/- 1.9512e-04 (5.91%) (init = 0.002)
    delta_B: -0.02664780 +/- 0.00492759 (18.49%) (init = -0.053)
    zeta:    -0.11822396 +/- 0.00296148 (2.50%) (init = -0.197)
    T1:       86.1948602 +/- 6.52126263 (7.57%) (init = 87)
    T2:       84.6487688 +/- 5.21640019 (6.16%) (init = 87)
    T3:       93.0839361 +/- 6.31980338 (6.79%) (init = 87)
    T4:       96.6994185 +/- 6.85795955 (7.09%) (init = 87)
    T5:       82.3545579 +/- 6.01598767 (7.30%) (init = 87)
    T6:       75.6956595 +/- 4.58367796 (6.06%) (init = 87)
    T7:       79.0078136 +/- 5.35300565 (6.78%) (init = 87)
    origin1:  0.03458457 +/- 0.01398432 (40.44%) (init = 0.061)
    origin2:  0.00336620 +/- 0.00818635 (243.19%) (init = 0.061)
    origin3: -0.05263649 +/- 0.01161394 (22.06%) (init = 0.061)
    origin4: -0.01972189 +/- 0.01018906 (51.66%) (init = 0.061)
    origin5:  0.07003104 +/- 0.01347507 (19.24%) (init = 0.061)
    origin6:  0.07986274 +/- 0.00684422 (8.57%) (init = 0.061)
    origin7:  0.01699546 +/- 0.00922854 (54.30%) (init = 0.061)

"""

#Save the report to a text file
with open('Cami_2004_fitting_7_sightlines_same_sigma.txt', 'w') as file:
    file.write(report)
    



   # Define a class to store the variables
class Variables:
 def __init__(self):
     self.variables = {}

 def __getattr__(self, name):
     return self.variables.get(name, None)

# Create an instance of the Variables class
v = Variables()


variables = {}
import re
with open('Cami_2004_fitting_7_sightlines_same_sigma.txt', 'r') as file:
    lines = file.readlines()
    parsing_variables = False
    for line in lines:
        if line.startswith("[[Variables]]"):
            parsing_variables = True
        elif parsing_variables and not line.startswith("#") and line.strip() != "":
            key_value = line.split(":")
            key = key_value[0].strip()
            value = re.search(r'\d+\.\d+', key_value[1]).group()
            variables[key] = float(value)

B = variables.get("B")
delta_B = variables.get("delta_B")

parameter_names = ['B', 'delta_B', 'zeta', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'sigma1', 'sigma2', 'sigma3',
                   'sigma4', 'sigma5', 'sigma6', 'sigma7', 'origin1', 'origin2', 'origin3', 'origin4', 'origin5',
                   'origin6', 'origin7']

# Print parameter values
for parameter in parameter_names:
    value = variables.get(parameter)
    print(parameter + ":", value)
    


flux_list = np.array([])
wave_list = np.array([])
stddev_array = np.array([])
#spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/edibles/edibles/utils/simulations/Charmi/Heather's_data/")
spec_dir = Path("//Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/Cami_2004_data/heliocentric/6614/")




combinations = pd.read_csv(r"/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/edibles/edibles/utils/simulations/Charmi/Jmax=300.txt", delim_whitespace=(True))


def get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin):
    startg = timeit.default_timer()
    
    print(B)
    print(T)
    print(delta_B)
    print(zeta)
    print(sigma)
    print(origin)
    
    #combinations  = allowed_perperndicular_transitions(Jmax)
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

        data_to_plot = np.array([Obs_data['Wavelength'], Obs_data['Flux'] ]).transpose()
        
        # removing red wing
        # Obs_data_trp = Obs_data [(Obs_data['Wavelength'] >= -1) & (Obs_data['Wavelength']<= 1.2)]
        Obs_data_trp = Obs_data[(Obs_data['Flux'] <= 0.95)]  # trp = triple peak structure
        
        # making data evenly spaced
        x_equal_spacing = np.linspace(min(Obs_data_trp['Wavelength']), max(Obs_data_trp['Wavelength']), 25)
        y_obs_data = np.interp(x_equal_spacing, Obs_data_trp['Wavelength'], Obs_data_trp['Flux'])
        Obs_data_continuum = Obs_data [(Obs_data['Wavelength'] >= 2) & (Obs_data['Wavelength']<= 5)]
        std_dev = np.std(Obs_data_continuum['Flux'])
        
        return x_equal_spacing, y_obs_data, std_dev, data_to_plot
    
sightlines = ['144217', '144470', '145502', '147165', '149757', '179406', '184915']

for i, sightline in enumerate(sightlines, start=1):

    B = variables.get('B', None)
    delta_B = variables.get('delta_B', None)
    zeta = variables.get('zeta', None)
    T = variables.get('T{}'.format(i), None)
    sigma = variables.get('sigma{}'.format(i), None)
    origin = variables.get('origin{}'.format(i), None)
    sigma = 0.0289
    linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)

    #x_equal_spacing, y_obs_data, std_dev, data_to_plot = curve_to_fit_wavenos(sightline)
    
    plt.figure(figsize = (15,8))
    #print(model_data)
    #plt.plot(data_to_plot[:,0], data_to_plot[:,1], color = 'black' , label = str('HD') + str(sightline ))
    plt.plot(model_data[:,0], model_data[:,1], color = 'red') #, label = str(r"$\chi^2$ =  ") + str('{:.3f}'.format(reduced_chi_squared)) + str('  (Altogether)') )
    