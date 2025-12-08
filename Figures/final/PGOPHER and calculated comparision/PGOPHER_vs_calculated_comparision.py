# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:42:50 2022

@author: Charmi Bhatt
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


plt.figure(figsize = (15,8))


Cal_smooth = pd.read_csv(r"/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Figures/final/PGOPHER and calculated comparision/Cal_conditon_c.txt", sep='\s+')
#plt.plot(Cal_smooth.iloc[:,0], 1-0.1*(Cal_smooth.iloc[:,1]/max(Cal_smooth.iloc[:,1])), linewidth = 4, color = 'orange', alpha = 0.7, label = 'Calculated')
#plt.plot(Cal_smooth.iloc[:,0], (Cal_smooth.iloc[:,1]/max(Cal_smooth.iloc[:,1])), linewidth = 4, color = 'orange', alpha = 0.7, label = 'Calculated')


PG_smooth = pd.read_csv(r"/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/Figures/final/PGOPHER and calculated comparision/PGO_conditon_c.txt", sep='\s+')
#PG_smooth = PG_smooth[(PG_smooth.iloc[:,0] <= (max(Cal_smooth.iloc[:,0])) ) & (PG_smooth.iloc[:,0] >= (min(Cal_smooth.iloc[:,0])) ) ]
#plt.plot(PG_smooth.iloc[:,0], (PG_smooth.iloc[:,1]/max(PG_smooth.iloc[:,1])), color = 'black', label = 'PGOPHER')


Cal_smooth_normalized = Cal_smooth.iloc[:,1] / max(Cal_smooth.iloc[:,1])
PG_smooth_normalized = PG_smooth.iloc[:,1] / max(PG_smooth.iloc[:,1])



print(len(Cal_smooth_normalized))
print(len(PG_smooth_normalized))

Cal_interpolated = np.interp(PG_smooth.iloc[:,0], Cal_smooth.iloc[:,0], Cal_smooth_normalized)

print(len(Cal_interpolated))


# Create subplots with a 3:1 height ratio
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                       gridspec_kw={'height_ratios': [3, 1]})
# Subplot 1
ax[0].plot(PG_smooth.iloc[:, 0], PG_smooth_normalized, color='black', label='PGOPHER')
#ax[0].scatter(Cal_smooth.iloc[:, 0], Cal_smooth_normalized, linewidth=2, color='orange', alpha=0.7, label='Calculated')

# Plot every 4th point for Cal_smooth using scatter
ax[0].scatter(Cal_smooth.iloc[::4, 0], Cal_smooth_normalized[::4], linewidth=2, 
              color='red', alpha=0.7, label='Calculated', s = 10)


ax[0].legend()
ax[0].set_ylabel('Normalized Intensity', labelpad = 13, fontsize = 13)


# Subplot 2
#markerline, stemlines, baseline = ax[1].stem(PG_smooth.iloc[:, 0], Cal_interpolated - PG_smooth_normalized, linefmt='r-', markerfmt='ro', basefmt='k-')
x_values = PG_smooth.iloc[::10, 0]
y_values = (Cal_interpolated - PG_smooth_normalized)[::10]
ax[1].scatter(x_values, y_values, color='blue', label='Residuals', s = 3)

#markerline, stemlines, baseline = ax[1].stem(x_values, y_values, linefmt='r-', markerfmt='ro', basefmt='k-', label = 'Residuals')
ax[1].set_xlabel('Wavenumbers ($cm^{-1}$)', fontsize = 12)
ax[1].set_ylabel('Residuals', fontsize = 13)
plt.ylim(-0.0002, 0.0005)
plt.axhline(y = 0, linestyle = 'dashed', color = 'gray')
plt.legend(loc = 'upper left')
plt.xlim(2, -2.5)
#plt.show()

plt.savefig("PGOPHER_vs_calculated_flipped_x_axis.pdf", format = 'pdf', bbox_inches="tight")
# #Plot Calculated and PGOPHER data
# ax[0].plot(Cal_smooth.iloc[:,0], Cal_smooth_normalized, linewidth=4, color='orange', alpha=0.7, label='Calculated')
# ax[0].plot(PG_smooth.iloc[:,0], PG_smooth_normalized, color='black', label='PGOPHER')
# ax[0].set_ylabel('Normalized Intensity')
# ax[0].legend()
# ax[0].set_title('Calculated vs PGOPHER')

# # Calculate and plot the residuals
# residuals = Cal_interpolated - PG_smooth_normalized
# ax[1].plot(Cal_smooth.iloc[:,0], residuals, color='red', label='Residuals')
# ax[1].set_xlabel('Wavelength')
# ax[1].set_ylabel('Residuals')
# ax[1].legend()

# plt.tight_layout()
# plt.show()


# # Calculated_normalized_intenisty = 1-0.1*(Cal_smooth.iloc[:,1]/max(Cal_smooth.iloc[:,1]))
# # PGO_Calculated_intenisty = 1-0.1*(PG_smooth.iloc[:,1]/max(PG_smooth.iloc[:,1]))
# # Calculated_wavenumber = Cal_smooth.iloc[:,0]
# # PGO_wavenumber = PG_smooth.iloc[:,0]


# # PGO_vs_cal = np.column_stack([PGO_wavenumber, PGO_Calculated_intenisty]) #, Calculated_wavenumber, Calculated_normalized_intenisty ])
# # np.savetxt('PGO_conditon_c.txt', PGO_vs_cal)

