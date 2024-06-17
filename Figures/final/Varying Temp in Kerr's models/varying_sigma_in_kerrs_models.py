

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

startfull = timeit.default_timer()

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

    end1 = timeit.default_timer()
    # print('>>>> Time taken to import parameters ' + str(end1 - start1) + '  sec')
    # print('==========')
    '''Calculating Linelist'''

    start2 = timeit.default_timer()
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
    
    end2 = timeit.default_timer()
    # print('>>>> Time taken to calculate wavenos  ' + str(end2 - start2) + '  sec')
    # print('==========')

    start3 = timeit.default_timer()

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

    end3 = timeit.default_timer()
    # print('>>>> Time taken to calculate HL factors' + str(end3 - start3) + '  sec')
    # print('==========')
    
    startbd = timeit.default_timer()

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
    endbd = timeit.default_timer()
    # print('>>>> Time taken to calculate BD factors' + str(endbd - startbd) + '  sec')
    # print('==========')

    starti = timeit.default_timer()

    intensities = []
    for i in range(len(linelist.index)):
        strength = (HL_factors[i] * BD_factors[i])
        intensities.append(strength)

    linelist['intensities'] = intensities
    
    endi = timeit.default_timer()
    # print('>>>> Time taken to calculate intenisties' + str(endi - starti) + '  sec')
    # print('==========')

    # endl = timeit.default_timer()
    # print('>>>> linelist calculation takes   ' + str(endl - startl) + '  sec')

    '''Smoothening the linelist'''
    start4 = timeit.default_timer() 
    
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
    return linelist, model_data



def obs_curve(sightline): 
    
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
        
        
       
        
       
        return Obs_data



'''Inputs'''    
Jmax = 1000
combinations  = allowed_perperndicular_transitions(Jmax)
# lambda_start = 6611 #6612.5453435174495 #-1.134 #
# lambda_end = 6616 # 6615 #1.039609311008462 #


# common_grid_for_all = make_grid(lambda_start, lambda_end, resolution=107000, oversample=2) #220000
# common_grid_for_all = (1 / common_grid_for_all) * 1e8
# common_grid_for_all = common_grid_for_all - 15119.4
# common_grid_for_all = common_grid_for_all[::-1]


#Cami 2004
# spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/Cami_2004_data/heliocentric/6614/")
# #sightlines = ['144217', '144470',  '145502', '147165', '149757', '179406', '184915'] 
# sightlines = ['144217']
# filename = 'hd{}_dib6614.txt'
#method = 'ampgo'

#EDIBLES data
spec_dir = Path("/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Data/Heather's_data")
filename = '6614_HD{}.txt'
sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']




'''Varying T in kerr's C model'''

#All paramters are from kerr's model c, except the one varying here (T)
# B = 0.00336
# delta_B = -0.17
# zeta=  -0.49
# Ts = (40,55)
# sigma = 0.1953
# origin = 0.12
# sightlines = ('166937', '185418')
# plt.figure(figsize=(16,9))
# vertical_offset = 0.05

# for T in Ts:
#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     linelist['intensities'] = 1-0.1*(linelist['intensities']/max(linelist['intensities']))
#     #plt.stem(linelist['wavenos'], linelist['intensities'], label = 'zeta = ' + str(zeta), bottom = 1)
#     plt.plot(model_data[:,0], model_data[:,1], label = 'Simulated at T = ' + str(T) + ' K', )

#     #plt.plot(model_data[:,0], model_data[:,1], label = 'Simulated at $\sigma$ = ' + str(sigma) + ' cm$^{{-1}}$')
#     # plt.axvline(x = -0.70, color = 'black', linestyle = 'dotted')
#     # plt.axvline(x = -0.58, color = 'black', linestyle = 'dotted')
#     # plt.axvline(x = 0.64, color = 'black', linestyle = 'dotted')
#     # plt.axvline(x = 0.78, color = 'black', linestyle = 'dotted')
#     # plt.title(f"B = {B} $cm^{{-1}}$  $\Delta B = ${delta_B} cm$^{{-1}}$  $\zeta^{{\prime}}  = ${zeta} cm$^{{-1}}$ $\sigma = ${sigma} cm$^{{-1}}$", fontsize = 20)
   
# colors = ('red', 'green')
# for sightline, color in zip(sightlines, colors):
    
#     Obs_data = obs_curve(sightline)
    
#     plt.plot(Obs_data['Wavelength'], Obs_data['Flux'] - vertical_offset, label = 'HD ' + str(sightline), color = color)
#     plt.xlim(3,-3)
#     plt.legend()


   
# plt.legend(loc = "lower right", fontsize = 15)
# plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize = 20,  labelpad= 10)
# plt.ylabel('Normalized Intenisty', fontsize = 20, labelpad= 10)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize= 20)
# plt.show()


# plt.savefig("Varying_T_in_kerr_model_c_flipped_x_axis.pdf", format="pdf", bbox_inches="tight")


'''Plotting 6614'''

sightline = 166937  
file = filename.format(sightline)
# Obs_data = pd.read_csv(spec_dir / file,
#                         delim_whitespace=(True))

Obs_data = pd.read_csv(spec_dir / file,
                        sep = ',')
    
plt.plot(Obs_data['Wavelength'], Obs_data['Flux'])
plt.xlim(6612,6616)
plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize = 15)
plt.ylabel('Normalized Intenisty', fontsize = 15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize= 15)
plt.show()
    

'''Varying sigma in kerr's C model'''

#All paramters are from kerr's model c, except the one varying here (sigma)
# B = 0.00336
# delta_B = -0.17
# zeta=  -0.49
# T = 61.2
# sigmas = (0.1512, 0.1953)
# origin = 0.12
# sightlines = ('166937', '185418')
# plt.figure(figsize=(16,9))
# vertical_offset = 0.05

# for sigma in sigmas:
#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     # linelist['intensities'] = 1-0.1*(linelist['intensities']/max(linelist['intensities']))
#     # plt.stem(linelist['wavenos'], linelist['intensities'], label = 'zeta = ' + str(zeta), bottom = 1)
#     #plt.plot(model_data[:,0], model_data[:,1], label = 'Simulated at T = ' + str(T) + ' K')

#     plt.plot(model_data[:,0], model_data[:,1], label = 'Simulated at $\sigma$ = ' + str(sigma) + ' cm$^{{-1}}$')
    
#     plt.axvline(x = -0.70, color = 'black', linestyle = 'dotted')
#     plt.axvline(x = -0.58, color = 'black', linestyle = 'dotted')
#     plt.axvline(x = 0.64, color = 'black', linestyle = 'dotted')
#     plt.axvline(x = 0.78, color = 'black', linestyle = 'dotted')
#     plt.title(f"B = {B} $cm^{{-1}}$  $\Delta B = ${delta_B} cm$^{{-1}}$  $\zeta^{{\prime}}  = ${zeta} cm$^{{-1}}$ T = {T} K", fontsize = 20)
   
    
# for sightline in sightlines:
    
#     Obs_data = obs_curve(sightline)
    
#     plt.plot(Obs_data['Wavelength'], Obs_data['Flux'] - vertical_offset, label = 'HD ' + str(sightline))
#     plt.xlim(3,-3)
#     plt.legend()
    
    
    
   
# plt.legend(loc = "lower right", fontsize = 15)
# plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize = 20,  labelpad= 10)
# plt.ylabel('Normalized Intenisty', fontsize = 20, labelpad= 10)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize= 20)


# plt.savefig("Varying_sigma_in_kerr_model_c_flipped_x_axis.pdf", format="pdf", bbox_inches="tight")


'''Test: After including linewidth, can models reproduce peak separations and width of Q branch?'''

#All paramters are from kerr's model c, except the one varying here (T and sigma)

# B = 0.00336
# delta_B = -0.17
# zeta=  -0.49
# Ts = (59.41, 64.05)
# sigmas = (0.197, 0.154)
# origins = (0.06, 0.053)
# vertical_offset = 0.05
# plt.figure(figsize=(16,9))

# #sightlineos = ['185418', '166937']
# sightlines = ('166937', '185418')

# for T, sigma, origin in zip(Ts, sigmas, origins):
#     linelist, model_data = get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     plt.plot(model_data[:,0], model_data[:,1], label = 'Models: T = ' + str(T) + ' K, $\sigma = $' + str(sigma) + 'cm$^{{-1}}$')  
    
#     # plt.axvline(x = -0.70, color = 'black', linestyle = 'dotted')
#     # plt.axvline(x = -0.58, color = 'black', linestyle = 'dotted')
#     # plt.axvline(x = 0.64, color = 'black', linestyle = 'dotted')
#     # plt.axvline(x = 0.78, color = 'black', linestyle = 'dotted')
#     # plt.title(f"B = {B} $cm^{{-1}}$  $\Delta B = ${delta_B} cm$^{{-1}}$  $\zeta^{{\prime}}  = ${zeta} cm$^{{-1}}$" , fontsize = 20) #"$\sigma = ${sigma} cm$^{{-1}}$" , fontsize = 20)
#     plt.legend()
    
#     plt.xlim(-3, 3)
    
# for sightline in sightlines:
    
#     Obs_data = obs_curve(sightline)
    
#     plt.plot(Obs_data['Wavelength'], Obs_data['Flux'] - vertical_offset, label = 'HD ' + str(sightline))
#     plt.xlim(3,-3)
#     #plt.legend()
    
    
# plt.legend(loc = "lower right", fontsize = 14)
# #plt.legend(bbox_to_anchor=(1.8, 0.7), loc='lower right', fontsize = 22)
# plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize = 20,  labelpad= 10)
# plt.ylabel('Normalized Intenisty', fontsize = 20, labelpad= 10)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize= 20)

# plt.show()

#plt.savefig("Varying_T_and_sigma_together_in_kerr_model_c_flipped_x_axis.pdf", format="pdf", bbox_inches="tight")









'''PLotting'''

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


# offset = np.arange(0, 12, 0.06)

# B = 0.00247690
# delta_B = -0.06864414
# zeta = -0.31136112

# Ts = [84.8194157, 95.5401280, 97.2287829, 116.540106, 99.2097303, 86.6632524, 98.4034623, 89.0236501, 97.3496743, 87.8768841, 103.793851, 86.2283890]

# sigmas = [0.18496609, 0.20292189, 0.18936606, 0.19319891, 0.22004922, 0.16349391, 0.17929137, 0.19690720, 0.20881159, 0.22038457, 0.24982774, 0.18963846]

# origins = [0.02918830, -0.00869084, 0.01065963, -0.00253786, 0.02965486, -0.00276476, 0.06965812, 0.02549114, 0.12524227, 0.07957677, 0.04544019, 0.08117755]

# plt.figure(figsize = (15,30))
# for T, sigma, origin, offset, sightline in zip(Ts, sigmas, origins, offset, sightlines):
#     Obs_data, x_equal_spacing, y_obs_data, std_dev = obs_curve_to_fit(sightline)
#     plt.plot(Obs_data['Wavelength'] , Obs_data['Flux'] - offset, color = 'black' ) #, label = 'HD ' + str(sightline) , color = 'black')


#     linelist, model_data =  get_rotational_spectrum(B, delta_B, zeta, T, sigma, origin)
#     plt.plot(model_data[:,0], model_data[:,1] - offset, color = 'red', label = 'HD{}, T = {:.3f} K, sigma = {:.3f} cm-1'.format(sightline, T, sigma))
#     plt.xlabel('Wavenumber', labelpad = 14, fontsize = 22)
#     plt.ylabel('Normalized Intenisty', labelpad = 14, fontsize = 22)
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     # plt.annotate('HD' + str(sightline), xy = (Obs_data['Wavelength'][150] , Obs_data['Flux'][150] - offset) , xytext = (4, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     # plt.annotate('T = {:.2f}'.format(T) + ' K', xy = (Obs_data['Wavelength'][40] , Obs_data['Flux'][40] - offset) , xytext = (-7, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17 )
#     #plt.annotate(r"$\sigma$ = {:.3f}".format(sigma) + '  cm$^{-1}$', xy = (Obs_data['Wavelength'][50] , Obs_data['Flux'][50] - offset) , xytext = (-5, Obs_data['Flux'][25] - offset + 0.009), fontsize = 17)
    
    
#     title_text = ' B = {:.5f} cm-1, Delta_B = {:.4f}, zeta = {:.4f}'.format(B, delta_B,  zeta)
#     plt.title(title_text, fontsize = 22) 
#     plt.xlim(-7.5, 6)
#    # plt.legend(loc = 'lower left', fontsize = 16)
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
    

