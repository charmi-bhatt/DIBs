#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:23:23 2024

@author: charmibhatt
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Data for CH+ linewidth and uncertainty
K_linewidth = [
    1.6807, 0.6989, 2.4115, 0.2685, 4.3511,
    0.6178, 0.4370, 1.9868, 0.3194, 2.4008,
    0.7416, 1.3943
]

K_uncertainty = [
    0.1511, 1.1680, 0.3686, 2.1407, 0.4570,
    1.6554, 1.3360, 0.1019, 0.8892, 0.2565,
    0.0475, 0.2748
]


# Data for DIB 6614 linewidth and uncertainty
dib_6614_linewidth = [3.2315, 3.5148, 3.2714, 3.1982, 3.8639, 2.828, 3.0399, 3.4332, 3.5275, 3.8875, 4.2204, 3.3111]
dib_6614_uncertainty = [0.1415, 0.1055, 0.1306, 0.3366, 0.1623, 0.2436, 0.1204, 0.0653, 0.1246, 0.094, 0.0841, 0.2353]

# Plotting
plt.errorbar(dib_6614_linewidth, K_linewidth,
             xerr=dib_6614_uncertainty, 
             fmt='ko', ecolor='indianred', capsize=3, markersize = 4)

plt.errorbar(dib_6614_linewidth, K_linewidth,
            yerr=K_uncertainty,
             fmt='ko', ecolor='green', capsize=3,  markersize = 4)



# Adding a line of best fit (linear regression)

correlation_coefficient, p_value = pearsonr( dib_6614_linewidth, K_linewidth)

coefficients = np.polyfit(dib_6614_linewidth, K_linewidth,  1)
poly = np.poly1d(coefficients)
x_points = np.linspace(min(dib_6614_linewidth) -0.3, max(dib_6614_linewidth) +0.3)
plt.plot(x_points, poly(x_points), color= 'gray', 
         label=f'Pearson Correlation Coefficient: {correlation_coefficient:.2f}',
         linestyle = 'dashed', alpha = 0.5)



plt.ylabel('K I linewidth (km s$^{-1}$)', fontsize = 12, labelpad=10)
plt.xlabel(r' $\lambda$6614 linewidth (km s$^{-1}$)' , fontsize = 12, labelpad=10)

#plt.ylim(1, 7)
plt.legend(loc = 'upper left' , fontsize = 10)


# Customize tick labels and reduce the number of ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on x-axis
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis

# Increase tick label size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add minor ticks
plt.minorticks_on()
plt.tick_params(which='both', width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='gray')

# # Set the border (spine) linewidth
# for spine in plt.gca().spines.values():
#     spine.set_linewidth(1.5)  # Adjust the linewidth to your desired thickness





# plt.savefig('Linewidth_6614_vs_CHplus.pdf', format = 'pdf', bbox_inches = 'tight')

# # Display the plot
# plt.show()
