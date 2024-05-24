from edibles import DATADIR
from edibles import PYTHONDIR
from edibles.utils.edibles_spectrum import EdiblesSpectrum
from edibles.utils.edibles_oracle import EdiblesOracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
import astropy.constants as cst
from io import BytesIO
from lmfit import Model
from lmfit.models import VoigtModel
from edibles.utils.voigt_profile import voigt_absorption_line

workdir = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_K_lines_linewidth_check/continuum_defined/' #subtracted_and_norm/' #Norm_spectra_data/'
pythia = EdiblesOracle()

vrad_filename = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_K_lines_linewidth_check/cloudVels_readable.csv"
V_rad_data = pd.read_csv(vrad_filename)



def load_EDIBLES_filenames(Wave, sightline):
    
    List = pythia.getFilteredObsList(object=[sightline] , OrdersOnly=True, Wave=Wave)

    #test = List.values.tolist()
    #filename = test[0]
    
    return List, sightline



def load_EDIBLES_spectrum(filename, sightline, make_plot):
     
    # loads in the spectrum, gives you 
    #radial velocity corrected wavelength in barycentric frame 9here called wave_rest
     
    sp = EdiblesSpectrum(filename)
    sp.getSpectrum(np.min(sp.raw_wave)+1,np.max(sp.raw_wave)-1)

    print(sp.target)

    wave = sp.bary_wave
    #search for V_rad from the csv file provided
    row = V_rad_data.loc[V_rad_data['Sightline'] == sightline]
    print(object)
    v_rad = row['V_rad'].values[0]
    print(v_rad)
    #radial velocity correction
    wave_rest = wave / (1+v_rad/cst.c.to("km/s").value)

    flux = np.clip(sp.bary_flux, 0, None) 

    if make_plot == True:
          plt.plot(wave_rest, flux)

    return wave_rest, flux, sp.target, sp.datetime




def define_continuum(start_wavelength, start_flux, end_wavelength, end_flux):
    slope = (end_flux - start_flux) / (end_wavelength - start_wavelength)
    intercept = start_flux - slope * start_wavelength
    
    equation = f"y = {slope:.2f} * x + {intercept:.2f}"
    print(f">>> Defined continuum equation: {equation}")
    return lambda w: slope * w + intercept






def fit_conitnuum (Wave,  sightlines, plotrange, conitnuum_range_before, conitnuum_range_after , fitting_range, make_plot):
    
    #sp = EdiblesSpectrum(filename)
    for i, sightline in enumerate(sightlines):
        List, sightline = load_EDIBLES_filenames (Wave, sightline)
        
        print(List)
        for i, filename in enumerate(List):  

            # print(filename)
            # print('>>>>>>>>>>>>...')
            #if i == 0: 
                wave_rest, flux, target, datetime = load_EDIBLES_spectrum(filename, sightline, make_plot=False )



                bool_keep = (wave_rest > plotrange[0]) & (wave_rest < plotrange[1])
                plotwave = wave_rest[bool_keep]
                plotflux = flux[bool_keep]

                
                #For continuum before the peak:
                
                bool_keep = (wave_rest > conitnuum_range_before[0]) & (wave_rest < conitnuum_range_before[1])
                #conitnuum_wave_before = wave_rest[bool_keep]
                conitnuum_flux_before = flux[bool_keep]                
                start_flux = np.mean(conitnuum_flux_before)
                start_wavelength = np.mean(conitnuum_range_before)

                #For continuum after the peak:

                bool_keep = (wave_rest > conitnuum_range_after[0]) & (wave_rest < conitnuum_range_after[1])
                #conitnuum_wave_after = wave_rest[bool_keep]
                conitnuum_flux_after= flux[bool_keep]
                end_flux = np.mean(conitnuum_flux_after)
                end_wavelength = np.mean(conitnuum_range_after)

                
                defined_continuum = define_continuum(start_wavelength , start_flux, end_wavelength, end_flux)

                
               
                #for seeing where the continuum is
                # plt.figure(figsize=(12,8))
                # if make_plot == True:
                #     plt.plot(plotwave, plotflux,  'r-', label='Data')
                #     plt.plot(plotwave, defined_continuum(plotwave))
                
                #     plt.plot(conitnuum_wave_before, conitnuum_flux_before)
                #     plt.plot(conitnuum_wave_after, conitnuum_flux_after)

                #     plt.scatter(start_wavelength, start_flux)
                #     plt.scatter(end_wavelength, end_flux)
                   
                #     plt.xlim(plotrange)
                #     plt.title(f'{sightline}_{filename}')
                #     #plt.show()
                #     save_plot_as = workdir + f'{sightline}_K_continuum_defined_obs{i}.png'
                #     plt.savefig(save_plot_as, format = 'png')

                    
                 
                flux_without_continuum = plotflux / defined_continuum(plotwave)
                normflux = (flux_without_continuum - min(flux_without_continuum))/ (max(flux_without_continuum) - min(flux_without_continuum))
                
                #for seeing normlaized spectra  
                # plt.figure(figsize=(12,8))
                # plt.plot(plotwave, normflux)
                # plt.xlim(plotrange)
                # plt.title(f'{sightline}_{filename}')

                # save_plot_as = workdir + f'{sightline}_K_continuum_subtracted_and_norm__obs{i}.png'
                # plt.savefig(save_plot_as, format = 'png')


                #saving normalized data
                # filename_column = [filename] * len(plotwave)
                # combined_array = np.array([filename_column, plotwave, normflux]).T
                # # Save the array to a text file
                # np.savetxt(workdir + f'norm_spectra_K_{sightline}_obs{i}.txt', combined_array, fmt='%s', delimiter=' ')
                
                #Model = voigt_absorption_line()
                wavegrid = np.array(plotwave)
                model=Model(voigt_absorption_line, independent_vars=['wavegrid'])
                model.set_param_hint('lambda0', value=7698.9, vary=False) #4232.548 #7698.9
                model.set_param_hint('f',  value= 3.393e-1, vary=False) #for K: 3.393e-1 for Ch+: 0.005450
                model.set_param_hint('gamma', value=3.8e7, vary=False) # for K: 3.8e7 for CH+ : 1.0e8
                model.set_param_hint('v_resolution', value=3.75, vary=False) #3.75, v_resolution = c/R = 3e5/8e4 for EDIBBLES
                model.set_param_hint('n_step', value=25, vary=False)
                model.set_param_hint('N', value=7e12)
                model.set_param_hint('b', value=3.36)
                model.set_param_hint('v_rad', value=0.6)
                params=model.make_params()
                params.pretty_print()
                print(' ')

                bool_keep = (plotwave > fitting_range[0]) & (plotwave < fitting_range[1])
                fitting_wave = plotwave[bool_keep]
                #fitting_flux= normflux[bool_keep]
                fitting_flux= (flux_without_continuum[bool_keep])

                


                result = model.fit(fitting_flux,params,wavegrid=fitting_wave)
                print(result.fit_report())
                result.params.pretty_print()

                result_dic = {}
                result_dic['Sightline'] = f'{sightline}'  
                result_dic['Red Chi-Squared'] = f'{result.redchi}'
                
                # Update result_dic with parameter values
                for name, param in result.params.items():
                    result_dic[name] = param.value
                
                # Add fit_report to result_dic
                result_dic['fit_report'] = f'{result.fit_report()}'
                
                fit_results_list.append(result_dic) 



                plt.figure(figsize=(10,8))
               
                plt.plot(plotwave, flux_without_continuum, 'b-', label='Data')
                plt.plot(fitting_wave, fitting_flux, color = 'green', label = 'This goes into fitting')
                plt.plot(fitting_wave,  result.best_fit, 'r-', label='Best Fit')
                plt.legend(loc='lower left', fontsize = 8)
                plt.xlabel('Plotwave')
                plt.ylabel('Continuum subtracted flux')
                plt.subplots_adjust(right=0.5)
                plt.figtext(0.55, 0.02, f'Sightline : {sightline} \n \n'  + result.fit_report(), wrap=True, horizontalalignment='left', fontsize=10)
                #plt.show()

                plt.savefig(save_here + f'{sightline}_fit_result_K_obs{i}.png' , format = 'png')

    results_dataframe = pd.DataFrame(fit_results_list)

            
    return plotwave, plotflux, results_dataframe



sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165', 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']

#sightlines = ['HD 24398']
fit_results_list = []

#K lines
save_here = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_K_lines_linewidth_check/Fits_results_K_lines/corrected_continuum_removal/'

plotwave, normflux, results_dataframe = fit_conitnuum(Wave=7698,  sightlines = sightlines, plotrange=(7698.1, 7699.9), 
                                   conitnuum_range_before = (7698.1, 7698.5),
                                    conitnuum_range_after = (7699.4, 7699.9),
                                    fitting_range = (7698.5, 7699.4),
                                     make_plot=True )

save_results_into_csv_as = save_here + 'K_voigt_fitting_results_all_obs_not_normalizing.csv'
results_dataframe.to_csv(save_results_into_csv_as)


#CH+
# save_here = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_CH_plus_lines_linewidth_check/fit_results_CH_plus/Correction-not_normalizing_before_fitting/'
# plotwave, normflux, results_dataframe= fit_conitnuum(Wave=4232,  sightlines = sightlines, plotrange=(4232.1, 4233), 
#                                                     conitnuum_range_before = (4232.1, 4232.4),
#                                                     conitnuum_range_after = (4232.75, 4233 ),
#                                                     fitting_range= (4232.2, 4232.9),
#                                                     make_plot=True )


# save_results_into_csv_as = save_here + 'CHplus_voigt_fitting_results_all_obs_not_normalizing.csv'
# results_dataframe.to_csv(save_results_into_csv_as)




                
                #Model = voigt_absorption_line()
                #wavegrid = np.array(plotwave)
                # model=Model(voigt_absorption_line, independent_vars=['wavegrid'])
                # model.set_param_hint('lambda0', value=4232.548, vary=False)
                # model.set_param_hint('f', value=6.393e-3, vary=False) #3.393e-1
                # model.set_param_hint('gamma', value=3.8e7, vary=False)
                # model.set_param_hint('v_resolution', value=3.75, vary=False) #3.75, v_resolution = c/R = 3e5/8e4 for EDIBBLES
                # model.set_param_hint('n_step', value=25, vary=False)
                # model.set_param_hint('N', value=7e11)
                # model.set_param_hint('b', value=2.36)
                # model.set_param_hint('v_rad', value=0.06)
                # params=model.make_params()
                # params.pretty_print()
                # print(' ')
                # result = model.fit(normflux,params,wavegrid=plotwave)
                # print(result.fit_report())
                # result.params.pretty_print()



                # plt.figure(figsize=(10,8))
                # plt.plot(plotwave, normflux, 'b-', label='Data')
                # plt.plot(plotwave, result.best_fit, 'r-', label='Best Fit')
                # plt.legend(loc='best')
                # plt.xlabel('Plotwave')
                # plt.ylabel('Normflux')
                # plt.subplots_adjust(right=0.5)
                # plt.figtext(0.55, 0.02, f'Sightline : {sightline} \n \n'  + result.fit_report(), wrap=True, horizontalalignment='left', fontsize=10)
                # plt.show()
                
                # plt.close()
                    

#  Args:
# wavegrid (float64): Wavelength grid (in Angstrom) on which the final result is desired.
# lambda0 (float64): Central (rest) wavelength for the absorption line, in Angstrom.
# b (float64): The b parameter (Gaussian width), in km/s.
# N (float64): The column density (in cm^{-2})
# f (float64): The oscillator strength (dimensionless)
# gamma (float64): Lorentzian gamma (=HWHM) component
# v_rad (float64): Radial velocity of absorption line (in km/s)
# v_resolution (float64): Instrument resolution in velocity space (in km/s)
# n_step (int): no. of point per FWHM length, governing sampling rate and efficiency
# debug (bool): If True, info on the calculation will be displayed

    
# result_dic = {}
# result_dic['Sightline'] = f'{sightline}'  
# result_dic['Chi-Squared']= f'{result.redchi}'
# # Update result_dic with parameter values
# for name, param in result.params.items():
#     result_dic[name] = param.value
# # Add fit_report to result_dic
# result_dic['fit_report'] = f'{result.fit_report()}'

# # Append result_dic to result_row
# result_row.append(result_dic)


# If you have more results, you can append their dictionaries to the list as well
# result_list.append(another_result_dic)

# Finally, convert the list of dictionaries to a dataframe
# results_dataframe = pd.DataFrame(result_row)



    

       

# print(results_dataframe)
# save_results_into_csv_as = workdir + 'CH_plus_voigt_fitting_results_of_all_sightlines_all_obs_with_chisquare.csv'
# results_dataframe.to_csv(save_results_into_csv_as)
    
# 
#plt.show()
# plt.close()

# plt.xlim(plotrange)
# plt.title('7698 K line in 12 EDIBLES single cloud sightlines')
    
# wavegrid = np.arange(1000) * 0.01 + 5000
# AbsorptionLine = voigt_absorption_line(
#     wavegrid,
#     lambda0=5003.0,
#     b=0.75,
#     N=1e11,
#     f=1,
#     gamma=1e7,
#     v_rad=-10.0,
#     v_resolution=0.26,
# )
# plt.plot(wavegrid, AbsorptionLine, marker="1")
# plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
# plt.show()
