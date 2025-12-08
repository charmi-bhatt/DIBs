import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import math



#plt.figure(figsize = (15,8))

Our_model_filepath = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Figures/final/PGOPHER and calculated comparision/Jmax= 1000/model_data_kerr_condition_c.csv'
Our_model= pd.read_csv(Our_model_filepath, sep = ',')
Our_model_wavelength = Our_model.iloc[:,0]
Our_model_intensity  =  Our_model.iloc[:,1]

Our_model_intensity = Our_model_intensity/max(Our_model_intensity)



PGOPHER_filepath  = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Figures/final/PGOPHER and calculated comparision/Jmax= 1000/Mainplot_with_Kerr_consition_c.txt'
PGOPHER = pd.read_csv(PGOPHER_filepath,  sep = '\s+')
PGOPHER_wavelength = PGOPHER.iloc[:,0]
PGOPHER_intensity = PGOPHER.iloc[:,1]

PGOPHER_intensity = 1-0.1*(PGOPHER_intensity/max(PGOPHER_intensity))



# print(PGOPHER)
# plt.plot(Our_model_wavelength, Our_model_intensity, label = 'Our model')
# plt.plot(PGOPHER_wavelength, PGOPHER_intensity, label = 'PGOPHER')

# Our_model_interpolated = np.interp( PGOPHER_wavelength, Our_model_wavelength, Our_model_intensity)
# plt.plot(PGOPHER_wavelength, (PGOPHER_intensity - Our_model_interpolated) + 1)

# plt.legend()
# plt.xlim(-3, 3)




#fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))


plt.figure(figsize= (11,8))
plt.subplot(4, 1, (1, 3))
# Plot Calculated and PGOPHER data
plt.scatter(Our_model_wavelength, Our_model_intensity, linewidth=1, color='darkorange', alpha=0.7, label='Calculated')
plt.plot(PGOPHER_wavelength, PGOPHER_intensity, color='black', label='PGOPHER')
plt.ylabel('Normalized Intenisty', fontsize = 17, labelpad= 20)
plt.legend(loc = "lower left", fontsize = 13)
plt.xlim(2, -2.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize= 17)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.01))
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.2))


plt.subplot(4, 1, 4)
# Calculate and plot the residuals
Our_model_interpolated = np.interp( PGOPHER_wavelength, Our_model_wavelength, Our_model_intensity)
residuals = PGOPHER_intensity - Our_model_interpolated
plt.plot(PGOPHER_wavelength, residuals, color='red', label='Residuals')
plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize = 17,  labelpad= 10)
plt.ylabel('Residuals', fontsize = 17,  labelpad= 10)
plt.legend(loc = 'upper left', fontsize = 13)
plt.xlim(2, -2.5)
#plt.ylim(-0.007, 0.007)
plt.xticks(fontsize=12)
plt.yticks(fontsize= 15)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.01))
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.2))




#plt.savefig("PGOPHER_vs_calculated_flipped_x_axis.pdf", format = 'pdf', bbox_inches="tight")


# plt.tight_layout()
# plt.show()

