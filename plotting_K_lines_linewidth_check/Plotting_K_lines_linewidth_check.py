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

workdir = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_K_lines_linewidth_check/'
pythia = EdiblesOracle()

vrad_filename = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_K_lines_linewidth_check/cloudVels_readable.csv"
V_rad_data = pd.read_csv(vrad_filename)

#object = 'HD 23180'

#sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165', 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']

sightlines = ['HD 149757']
result_row = []

for i, sightline in enumerate(sightlines):
    List = pythia.getFilteredObsList(object=[sightline], OrdersOnly=True, Wave=7698)
    test = List.values.tolist()
    print('=======')
    print(test)
   
    #filename = test[0]
    for filename in List: 
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

        plotrange=(7698.7, 7699.15)
        bool_keep = (wave_rest > plotrange[0]) & (wave_rest < plotrange[1])
        plotwave = wave_rest[bool_keep]
        plotflux = flux[bool_keep]

        print(wave_rest)
        print(plotwave)
        



        # Let's do a quick zeroth-order normalization
        # coef = np.polyfit(plotwave,plotflux,1)
        # poly1d_fn = np.poly1d(coef) 
        # linefit = poly1d_fn(plotwave)
        normflux = plotflux / np.median(plotflux)
        # normflux = plotflux / linefit

        wavegrid = np.array(plotwave)
        
        plt.plot(plotwave, normflux, 'b-', label='Data')

        
        #Model = voigt_absorption_line()

        model=Model(voigt_absorption_line, independent_vars=['wavegrid'])
        model.set_param_hint('lambda0', value=7698.974, vary=False)
        model.set_param_hint('f', value=3.393e-1, vary=False)
        model.set_param_hint('gamma', value=3.8e7, vary=False)
        model.set_param_hint('v_resolution', value=0.56, vary=False)
        model.set_param_hint('n_step', value=25, vary=False)
        model.set_param_hint('N', value=7e11)
        model.set_param_hint('b', value=2.36)
        model.set_param_hint('v_rad', value=0.06)
        params=model.make_params()
        params.pretty_print()
        print(' ')
        result = model.fit(normflux,params,wavegrid=plotwave)
        print(result.fit_report())
        result.params.pretty_print()

        import pandas as pd

    

        result_dic = {}
        result_dic['Sightline'] = f'{sightline}'  
        # Update result_dic with parameter values
        for name, param in result.params.items():
            result_dic[name] = param.value
        # Add fit_report to result_dic
        result_dic['fit_report'] = f'{result.fit_report()}'
        # Append result_dic to result_row
        result_row.append(result_dic)


        # If you have more results, you can append their dictionaries to the list as well
        # result_list.append(another_result_dic)

        # Finally, convert the list of dictionaries to a dataframe
        results_dataframe = pd.DataFrame(result_row)
        


        

        plt.figure(figsize=(10,8))
        plt.plot(plotwave, normflux, 'b-', label='Data')
        plt.plot(plotwave, result.best_fit, 'r-', label='Best Fit')
        plt.legend(loc='best')
        plt.xlabel('Plotwave')
        plt.ylabel('Normflux')
        plt.subplots_adjust(right=0.5)
        plt.figtext(0.55, 0.02, f'Sightline : {sightline} \n \n'  + result.fit_report(), wrap=True, horizontalalignment='left', fontsize=10)
        #plt.show()
        save_plot_as = workdir + f'{sightline}_K_lines_width_check_all_obs.png'
        plt.savefig(save_plot_as, format = 'png')
        plt.close()
        
print(results_dataframe)
save_results_into_csv_as = workdir + 'voigt_fitting_results_of_all_sightlines_all_obs.csv'
results_dataframe.to_csv(save_results_into_csv_as)
    
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
