#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:36:38 2024

@author: charmibhatt
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import colors as mcolors

# Get a list of named colors
colors = list(mcolors.TABLEAU_COLORS.values())  # Tableau palette
# If more colors are needed, use a larger palette
# colors = list(mcolors.CSS4_COLORS.values())  # CSS4 palette
sightlines = ['23180', '24398', '144470', '147165' , '147683', '149757', '166937', '170740', '184915', '185418', '185859', '203532']


# Data for CH+ linewidth and uncertainty
ch_plus_linewidth = [2.8497, 2.371, 2.9273, 2.0664, 5.5569, 2.3963, 2.3283, 3.743, 2.7586, 4.9278, 3.3286, 2.8754]
ch_plus_uncertainty = [0.1215, 0.2464, 0.1296, 0.1437, 0.2709, 0.0405, 0.1195, 0.0782, 0.1459, 0.1614, 0.0517, 0.1136]

# Data for DIB 6614 linewidth and uncertainty
dib_6614_linewidth = [3.2315, 3.5148, 3.2714, 3.1982, 3.8639, 2.828, 3.0399, 3.4332, 3.5275, 3.8875, 4.2204, 3.3111]
dib_6614_uncertainty = [0.1415, 0.1055, 0.1306, 0.3366, 0.1623, 0.2436, 0.1204, 0.0653, 0.1246, 0.094, 0.0841, 0.2353]

# Plotting -------------------------------------------------------------------------------------
# plt.errorbar(dib_6614_linewidth, ch_plus_linewidth,
#              xerr=dib_6614_uncertainty, 
#              fmt='ko', ecolor='indianred', capsize=3, markersize = 4)

# plt.errorbar(dib_6614_linewidth, ch_plus_linewidth,
#             yerr=ch_plus_uncertainty,
#              fmt='ko', ecolor='steelblue', capsize=3,  markersize = 4)


# colors = plt.get_cmap('tab20', len(sightlines))

plt.figure(figsize=(5.5, 4))
colors = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (0.6, 0.6, 0.0),  # Olive (replacing Yellow with a darker shade)
    (1.0, 0.5, 0.0),  # Orange
    (0.5, 0.0, 0.5),  # Purple
    (0.0, 1.0, 1.0),  # Cyan
    (0.3, 0.3, 0.3),  # Dark Gray
    (0.0, 0.0, 0.0),  # Black
    (1.0, 0.0, 1.0),  # Magenta
    (0.4, 0.2, 0.1),  # Dark Brown
    (0.2, 0.2, 0.4),  # Navy Blue
    (0.2, 0.6, 0.2),  # Forest Green
    (0.4, 0.0, 0.2),  # Deep Plum
    (0.3, 0.0, 0.0),  # Dark Red
    (0.2, 0.2, 0.2),  # Charcoal
]


# Plot each point with its error bars
for i in range(len(dib_6614_linewidth)):
    plt.errorbar(
        dib_6614_linewidth[i], ch_plus_linewidth[i],
        xerr=dib_6614_uncertainty[i], yerr=ch_plus_uncertainty[i], label = 'HD ' + sightlines[i],
        fmt='o', color=colors[i], ecolor=colors[i], capsize=3, markersize=4
    )


# -------------------------------------------------------------------------------------



# Adding a line of best fit (linear regression)

correlation_coefficient, p_value = pearsonr( dib_6614_linewidth, ch_plus_linewidth)

coefficients = np.polyfit(dib_6614_linewidth, ch_plus_linewidth,  1)
poly = np.poly1d(coefficients)
x_points = np.linspace(min(dib_6614_linewidth) -0.1, max(dib_6614_linewidth))
plt.plot(x_points, poly(x_points), color= 'gray', 
         linestyle = 'dashed', alpha = 0.5)

plt.annotate(
    f'Pearson Correlation Coefficient: {correlation_coefficient:.2f}',  # Text to display
    xy=(0.7, 0.95),  # Coordinates in axes fraction (normalized between 0 and 1)
    xycoords='axes fraction',  # Place relative to the plot area
    fontsize=10,  # Font size of the annotation
    ha='right',  # Horizontal alignment
    va='top'  # Vertical alignment
)
plt.ylabel('CH$^+$ linewidth (km s$^{-1}$)', fontsize = 12, labelpad=10)
plt.xlabel(r' $\lambda$6614 linewidth (km s$^{-1}$)' , fontsize = 12, labelpad=10)

plt.ylim(1, 7)
plt.xlim(2.4, 4.4)
plt.legend(
    loc='upper right',  # Specify location relative to the bounding box
    bbox_to_anchor=(1.32, 1),  # Adjust position outside the plot
    fontsize=10 , # Font size of the legend
    frameon = False
)


# Customize tick labels and reduce the number of ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on x-axis
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  # Fewer ticks on y-axis

# Increase tick label size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add minor ticks
plt.minorticks_on()
plt.tick_params(which='both', width=1, direction  ='in')
plt.tick_params(which='major', length=7,  direction  ='in')
plt.tick_params(which='minor', length=4, color='gray',  direction  ='in')




plt.savefig('Linewidth_6614_vs_CHplus.pdf', format = 'pdf', bbox_inches = 'tight')

# # Display the plot
# plt.show()
