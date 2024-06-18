''' This code takes the co-added data o CHplus and K line and overplots them

'''

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
from edibles.utils.functions import make_grid


##################################

'''
Importing CHplus co-addded data and directory paths
'''

coadded_data_path = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting_CHplus_and_K_lines/"
savehere = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting_CHplus_and_K_lines/'
##################################

def wavelength_to_velocity(plotwave, rest_wave):

    z = (plotwave - rest_wave)/rest_wave
    velocity = z*cst.c.to("km/s").value

    return velocity




sightlines = ['HD 23180', 'HD 24398', 'HD 144470', 'HD 147165', 'HD 147683', 'HD 149757', 'HD 166937', 'HD 170740', 'HD 184915', 'HD 185418', 'HD 185859', 'HD 203532']
#sightlines = ['HD 24398'] 


##################################
''' All 12 sightlines in one plot'''
##################################
plt.figure(figsize = (15, 40))
start = 0
spacing = 1
count = 12

offsets = np.arange(start, start + spacing * count, spacing)

for offset, sightline in zip(offsets, sightlines): 

    CHplus_coadded_data = pd.read_csv(coadded_data_path + f'CHplus_coadded_data/{sightline}_coadded_data_CHplus.txt', sep = ' ')

    CHplus_plotwave = CHplus_coadded_data.iloc[:,0]
    CHplus_velocity = wavelength_to_velocity(CHplus_plotwave, rest_wave=4232.548)
    CHplus_plotflux = CHplus_coadded_data.iloc[:,1]


    K_coadded_data = pd.read_csv(coadded_data_path + f'K_line_coadded_data/{sightline}_coadded_data_K_line.txt', sep = ' ')

    K_plotwave = K_coadded_data.iloc[:,0]
    K_velocity = wavelength_to_velocity(K_plotwave, rest_wave=7698.965)
    K_plotflux = K_coadded_data.iloc[:,1]

    


    plt.plot(CHplus_velocity, CHplus_plotflux - offset, color = 'blue', 
             label='CH+ (4232 $\AA$)' if offset == offsets[0] else "")
    plt.plot(K_velocity, K_plotflux - offset, color = 'red',
             label='K line (7698 $\AA$)' if offset == offsets[0] else "")
    plt.axvline(x = 0, alpha = 0.3, linestyle = 'dashed')
    plt.axvline(x = 2, alpha = 0.3, linestyle = 'dashed')
    plt.axvline(x = -2, alpha = 0.3, linestyle = 'dashed')

    xy = (-10 , CHplus_plotflux[10] - offset)
    plt.annotate( f'{sightline}', xy = xy , xytext = (-10, CHplus_plotflux[10] - offset + 0.1), 
                 fontsize = 20 )
    
    
    plt.xlim(-15, 15)
    plt.legend(loc = 'lower left', fontsize = 20)
    plt.xlabel('Velocity (km/s)', labelpad = 14, fontsize = 22)
    plt.ylabel('Normalized Intensity', labelpad = 14, fontsize = 22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.tick_params(axis='both', which='minor', labelsize=22)

   
plt.savefig(savehere + 'CHplus_and_K_line_in_velocity_space.png', format = 'png')
#plt.show()

##################################
'''Individual sightline plots'''
##################################

# for sightline in sightlines: 

#     CHplus_coadded_data = pd.read_csv(coadded_data_path + f'CHplus_coadded_data/{sightline}_coadded_data_CHplus.txt', sep = ' ')

#     CHplus_plotwave = CHplus_coadded_data.iloc[:,0]
#     CHplus_velocity = wavelength_to_velocity(CHplus_plotwave, rest_wave=4232.548)
#     CHplus_plotflux = CHplus_coadded_data.iloc[:,1]


#     K_coadded_data = pd.read_csv(coadded_data_path + f'K_line_coadded_data/{sightline}_coadded_data_K_line.txt', sep = ' ')

#     K_plotwave = K_coadded_data.iloc[:,0]
#     K_velocity = wavelength_to_velocity(K_plotwave, rest_wave=7698.965)
#     K_plotflux = K_coadded_data.iloc[:,1]

    


#     plt.plot(CHplus_velocity, CHplus_plotflux, color = 'blue', 
#              label='CH+ (4232 $\AA$)' )
#     plt.plot(K_velocity, K_plotflux, color = 'red',
#              label='K line (7698 $\AA$)' )
#     plt.axvline(x = 0, alpha = 0.3, linestyle = 'dashed')
#     plt.axvline(x = 2, alpha = 0.3, linestyle = 'dashed')
#     plt.axvline(x = -2, alpha = 0.3, linestyle = 'dashed')

#     plt.legend(loc = 'lower left')
#     plt.xlabel('Velocity (km/s)' )
#     plt.ylabel('Normalized Intensity')
#     plt.xlim(-10, 10)
#     plt.title(f'{sightline} CH+ and K in velocity space')
#     #plt.savefig(savehere + f'overplotting_in_velocity_space/{sightline}CHplus_and_K_line_in_velocity_space.png', format = 'png')
#     #plt.close()
#     #plt.show()
