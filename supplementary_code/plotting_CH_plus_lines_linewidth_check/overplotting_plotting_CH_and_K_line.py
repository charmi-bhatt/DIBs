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

    print(sp.datetime)
    return wave_rest, flux, sp.target, sp.datetime


pythia = EdiblesOracle()
vrad_filename = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_K_lines_linewidth_check/cloudVels_readable.csv"
V_rad_data = pd.read_csv(vrad_filename)

# Wave = 4232
# sightlines = ['HD 149757']
# for i, sightline in enumerate(sightlines):
#         List, sightline = load_EDIBLES_filenames (Wave, sightline)
        
#         print(List)
#         for i, filename in enumerate(List):  

           
#             if i == 0: 
#                 wave_rest, flux, target, datetime = load_EDIBLES_spectrum(filename, sightline, make_plot=False )
#                 plt.plot(wave_rest, flux)
#                 plt.show()


# Wave = [7698, 4232]
# rest_wave = [7698.9, 4232.5]
# label= ['K', 'CH+']
# sightline = 'HD 166937'

def define_continuum(start_wavelength, start_flux, end_wavelength, end_flux):
    slope = (end_flux - start_flux) / (end_wavelength - start_wavelength)
    intercept = start_flux - slope * start_wavelength
    
    equation = f"y = {slope:.2f} * x + {intercept:.2f}"
    #print(f">>> Defined continuum equation: {equation}")
    return lambda w: slope * w + intercept


def plot_K_and_CHplus_line (wave, sightline, plotrange, rest_wave, conitnuum_range_before, conitnuum_range_after):
    
        List, sightline = load_EDIBLES_filenames (wave, sightline)

        print('=======')
        print(List)

        print('=======')

        velocities = []
        fluxes = []
        for i, filename in enumerate(List):

                # print(filename)  

            
            if i == 0: 
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
                flux_without_continuum = plotflux / defined_continuum(plotwave)
                normflux = (flux_without_continuum - min(flux_without_continuum))/ (max(flux_without_continuum) - min(flux_without_continuum))
                    
            
                z = (plotwave - rest_wave)/rest_wave
                velocity = z*cst.c.to("km/s").value
                
                # plt.plot(velocity, flux, label = label)
                # plt.xlim(-20, 20)
                # plt.legend()

                # velocities.append(velocity)
                # fluxes.append(normflux)

                return velocity, normflux, datetime
        
            
        
sightlines = ['HD 23180']
sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165', 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']


save_plots_here = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_CH_plus_lines_linewidth_check/Overplotting_CHplus_and_K_in velocity_space/'
for sightline in sightlines : 
      
    

   
    velocity1, flux1, datetime1  = plot_K_and_CHplus_line(wave = 4232, rest_wave = 4232.548, sightline= sightline,
                                            plotrange=(4232.1, 4233),
                                        conitnuum_range_before = (4232.1, 4232.4),
                                        conitnuum_range_after = (4232.75, 4233 ) )

        
    
  

    velocity2, flux2, datetime2  = plot_K_and_CHplus_line(wave = 7698, rest_wave = 7698.965, sightline= sightline,
                                             plotrange=(7698.1, 7699.9), 
                                   conitnuum_range_before = (7698.1, 7698.5),
                                    conitnuum_range_after = (7699.4, 7699.9))
    
    
    plt.figure()
    plt.plot(velocity1, flux1, label = f'CH+ {datetime1}')
    plt.plot(velocity2, flux2, label = f'K   {datetime2}')
    plt.title(sightline)
    plt.xlim(-13, 13)
    plt.ylim(-0.3, 1.1)
    plt.legend(loc = 'lower left')
    #plt.show()


    plt.savefig(save_plots_here + f'{sightline}_obs0_CHplus_and_K.png' , format = 'png')