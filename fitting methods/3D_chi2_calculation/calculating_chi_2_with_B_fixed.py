
import numpy as np
import pandas as pd
import astropy.constants as const
import matplotlib.pyplot as plt
import timeit
import scipy.stats as ss
from scipy.signal import argrelextrema
from matplotlib import cm

#Obs_data = pd.read_csv(r"/Users/charmibhatt/Desktop/Local_GitHub/edibles/edibles/utils/simulations/Charmi/Heather's_data/6614_HD_166937.txt", sep = ',')
#185418
Obs_data = pd.read_csv(r"/Users/charmibhatt/Desktop/Local_GitHub/edibles/edibles/utils/simulations/Charmi/Heather's_data/HD185418_avg_spectra.txt", sep = ',')

y_obs_data =  np.array(Obs_data['Flux'])
x_obs_data = np.array(Obs_data['Wavelength'])

y_obs_data=  (y_obs_data - min(y_obs_data)) / (1 - min(y_obs_data)) * 0.1 + 0.9





x_for_model = np.linspace(min(x_obs_data), max(x_obs_data), len(x_obs_data))
y_obs_data = np.interp(x_for_model, x_obs_data, y_obs_data)


# stepsize_B = 0.00001
# Bs  = np.arange(0.0028, 0.0032, stepsize_B)
# #Bs  = np.arange(0.0005, 0.01, stepsize_B)
# #Bs  = np.arange(1, 10, 4)
# print(Bs)
# print(len(Bs))

stepsize_T= 1
Ts =  np.arange(35, 95, stepsize_T)
#Ts =  np.arange(5, 100, 40)
# print(Ts)
# print(len(Ts))


min_index = np.argmin(y_obs_data)

central_peak_wavelength = x_obs_data[min_index]

origin = (1/central_peak_wavelength)*1e8

#print(origin)


combinations = pd.read_csv(r"/Users/charmibhatt/Desktop/Local_GitHub/edibles/edibles/utils/simulations/Charmi/Jmax=300.txt", delim_whitespace=(True))

startl = timeit.default_timer()
def get_rotational_spectrum(xx, T):
    B = 0.00305
    x_obs_data = xx
    ground_B = B
    delta_B = -0.17
    delta_C = -0.17
    zeta = -0.49
    sigma = 0.1953
    
    #rotational constants in cm-1
    ground_C = B/2
    delta_C = delta_C
    excited_B = ground_B + ((delta_B/100)*ground_B)
    excited_C = ground_C + ((delta_C/100)*ground_C)
    
    global combinations
    ground_Js = combinations['ground_J']
    excited_Js = combinations['excited_J']
    ground_Ks = combinations['ground_K']
    excited_Ks = combinations['excited_K']
    
    linelist = combinations
   
    delta_J = linelist['excited_J'] - linelist['ground_J']
    delta_K = linelist['excited_K'] - linelist['ground_K']
    
    '''Calculating Linelist'''
    #%%
    
    ground_Es = []
    for J,K in zip(ground_Js, ground_Ks):
                ground_E = ground_B*J*(J + 1) + (ground_C - ground_B)*(K**2)
                ground_Es.append(ground_E)
                
    linelist['ground_Es'] = ground_Es 
    
    excited_Es = []
    for J,K, del_K in zip(excited_Js, excited_Ks, delta_K):
        if del_K == -1:
                excited_E = excited_B*J*(J + 1) + (excited_C - excited_B)*(K**2) - ((-2*excited_C*zeta))*K + excited_C**2
        elif del_K == 1:
                excited_E = excited_B*J*(J + 1) + (excited_C - excited_B)*(K**2) + ((-2*excited_C*zeta))*K + excited_C**2

        excited_Es.append(excited_E)
                
    linelist['excited_Es'] = excited_Es
    
    wavenos = []
    for i in range(len(linelist.index)):
        wavenumber = origin + excited_Es[i] - ground_Es[i]
        wavenos.append(wavenumber)
        
    linelist['wavenos'] = wavenos
    
    HL_factors = []
    
    for J,K, delta_J, delta_K in zip(ground_Js, ground_Ks, delta_J, delta_K):
                if delta_J == -1 and delta_K == -1:
                    HL_factor = ((J - 1 + K)*(J + K))/(J*((2*J)+1))
                elif delta_J  == -1 and delta_K == 1:
                    HL_factor = ((J - 1 - K)*(J - K))/(J*((2*J)+1))
                elif delta_J == 0 and delta_K == -1:
                    HL_factor = (J+1-K)*(J+K)/ (J*(J + 1))
                elif delta_J == 0 and delta_K == 1:
                    HL_factor = (J+1+K)*(J-K) / (J*(J + 1))
                elif delta_J == 1 and delta_K == -1:
                    HL_factor = ( J + 2 - K)*( J + 1 - K)/ ((J + 1)*((2*J) +1))
                elif delta_J == 1 and delta_K == 1:
                    HL_factor = ( J + 2 + K)*( J + 1 + K)/ ((J + 1)*((2*J) + 1))
    
                HL_factors.append(HL_factor)
    
    linelist['HL_factors'] = HL_factors
                
    BD_factors = []
            
    h = const.h.cgs.value
    c = const.c.to('cm/s').value
    k = const.k_B.cgs.value
    
    
    for J,K,E in zip(ground_Js, ground_Ks, ground_Es):
        if K == 0:
            boltzmann_equation = (2*((2*J) + 1))*(np.exp((-h * c * E) / (k*T)))
        else:
            boltzmann_equation = (1*((2*J) + 1))*(np.exp((-h * c * E) / (k*T)))
            
        BD_factors.append(boltzmann_equation)
        
    linelist['BD_factors'] = BD_factors
    
    intensities = [] 
    for i in range(len(linelist.index)):
                strength = (HL_factors[i] * BD_factors[i])
                intensities.append(strength)
      
    linelist['intensities'] = intensities
    
    # normalized_intensities = (intensities / max(intensities))
    # linelist['normalized_intensities'] = normalized_intensities
    
   
    
    endl = timeit.default_timer()
    print('>>>> linelist calculation takes   ' + str(endl-startl) + '  sec')
    #%%
   
    '''Smoothening the linelist'''
    
    #%%
   
    smooth_wavenos = np.linspace(np.min(linelist['wavenos']) - 1 ,np.max(linelist['wavenos']) + 1, 1000) # grid_size)
    smooth_intensities = np.zeros(smooth_wavenos.shape)
    
    startg = timeit.default_timer()
    
    for idx,wavepoint in np.ndenumerate(smooth_wavenos):
        w_int = ss.norm.pdf(linelist['wavenos'], wavepoint, sigma) * (linelist['intensities']) 
        smooth_intensities[idx] = w_int.sum()
    

    endg = timeit.default_timer()
    
    print('>>>> gaussian takes   ' + str(endg -startg) + '  sec') 
    
    smooth_data = np.array([smooth_wavenos, smooth_intensities]).transpose()    
    smooth_data = np.delete(smooth_data, np.where(smooth_data[:,1] <= 0.001*(max(smooth_data[:,1]))), axis = 0)
    
    simu_waveno = smooth_data[:, 0]
    simu_wavelength = (1/simu_waveno)*1e8
    simu_intenisty = 1-0.1*(smooth_data[:, 1]/max(smooth_data[:, 1]))
    
    
    #%%
    
    P_Branch = linelist[(linelist['label'].str[1] == "P")]
    Q_Branch = linelist[(linelist['label'].str[1] == "Q")]
    R_Branch = linelist[(linelist['label'].str[1] == "R")]

    
    
    peak_p = linelist[linelist['intensities'] == max(P_Branch['intensities'])]
    peak_q = linelist[linelist['intensities'] == max(Q_Branch['intensities'])]
    peak_r = linelist[linelist['intensities'] == max(R_Branch['intensities'])]

    #linelist.to_excel(r"C:\Users\Charmi Bhatt\OneDrive\Desktop\my_local_github\edibles\edibles\utils\simulations\Charmi\Calculated_linelist_kerr_condition_c.xlsx", index=False)

    #with sns.color_palette("flare", n_colors=2):
        
    # plt.plot(simu_wavelength, simu_intenisty, color = 'red')
    # plt.plot(x_obs_data, y_obs_data)
    # plt.xlabel('Wavelength')
    # plt.ylabel('Normalized Intenisty')
    # plt.title('Temperature = ' + str(T) + '  K  ground_B =  ' + str(ground_B) + ' cm-1  ground_C=  ' + str(ground_C) + ' cm-1  Delta_B = ' + str(delta_B) + '    Delta_C = ' + str(delta_C) +    '    zeta = ' +  str(zeta)) 
    
    model_data = np.array([simu_wavelength, simu_intenisty]).transpose()
    model_data = model_data[::-1]
    
    x_for_model = np.linspace(min(x_obs_data), max(x_obs_data), len(x_obs_data))
    
    y_model_data = np.interp(x_for_model, model_data[:,0], model_data[:,1])
    
    
    
    return  y_model_data
 
# BB, TT = np.meshgrid(Bs, Ts)

# ax = plt.axes(projection='3d')



   
# red_chi_2D = np.zeros(shape = (1,len(Bs)))
# ax = plt.axes(projection='3d')
reduced_chis = []
Temps = []
for T in Ts:
    Temps.append(T)
    num = (get_rotational_spectrum(x_obs_data, T) - y_obs_data)**2
    chi_squared = np.sum((num)/(0.004)**2)
    reduced_chi_squared = chi_squared/(len(y_obs_data) - 2)
    reduced_chis.append(reduced_chi_squared)
      
    
print(reduced_chis)
print(Ts)

T_and_chi  = np.array([reduced_chis, Ts]).transpose()
print(T_and_chi)


    #red_chi_2D  = np.vstack([red_chi_2D , reduced_chis])   
    
# red_chi_2D = np.delete(red_chi_2D,0, 0)
# print(red_chi_2D)
# ax.plot_surface(BB, TT, red_chi_2D, cmap=plt.cm.YlGnBu_r,
#                             linewidth=0, antialiased=False)
# print(BB.shape)
# print(TT.shape)
# print(red_chi_2D.shape)

# np.savetxt('185418_BB_Bmin_' + str(min(Bs)) + '_Bmax_' + str(max(Bs)) + '_stepsize_' + str(stepsize_B) + '_.txt',  BB, delimiter = ' ')
# np.savetxt('185418_Bmax_0.0032_TT_Tmin_' + str(min(Ts)) + '_Tmax_' + str(max(Ts)) + '_stepsize_' + str(stepsize_T) + '_.txt',  TT, delimiter = ' ')
# np.savetxt('185418_red_chi_finer_B_0.0028_to_0.0032_57_to_68.txt', red_chi_2D, delimiter = ' ')







#np.savetxt('chi_plane_cordinates_meshgrid.txt', chi_plane, delimiter=' ')

# chi_arr = np.zeros(shape = (1,len(Bs)))
#         for T in Ts:
#             chis = []
#             for B in Bs:
#                 chi =  B
#                 chis.append(chi)
#             print(chis)
#             print('----')
#             chi_arr = np.vstack([chi_arr, chis])   
#             print(chi_arr)
#             print('----')
            
#         chis = np.delete(chi_arr, 0, 0)
#         print(chis)
    
#     rc_array = np.vstack([reduced_chis]) 
#     print(rc_array)
#     #ax.scatter3D(B, T, rc_array)
#     ax.plot_surface(B, T, rc_array, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False) #rstride=1, cstride=1,
#     print('--------------')
    
    
# T_arr = np.array([[25.9, 22.84, 22.53],
#                  [16.46, 2.448, 9.312]])
# ax.plot_surface(BB, TT, T_arr.transpose(), cmap=cm.coolwarm,
#                    linewidth=5, antialiased=True)



            
# chi_plane = np.array([B_grid, T_grid, reduced_chis]).transpose()
# np.savetxt('chi_plane_cordinates.txt', chi_plane, delimiter=' ')
# print(chi_plane)



# plt.plot(x_obs_data, y_obs_data)
# plt.plot(x_obs_data, get_rotational_spectrum(x_obs_data, B = 0.003, T = 61.2))




# y_obs_data =  np.array(Obs_data['Flux'])
# minima = [argrelextrema(y_obs_data, np.less)]
# minima_ind = minima[0]
# print(y_obs_data(minima_ind))
