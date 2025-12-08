#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:21:00 2024

@author: charmibhatt
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import matplotlib
from scipy.stats import pearsonr

data = pd.read_csv('centraldepths_6614_CHplus_and_K.csv')

sightlines = data['Sightlines']


# colors = plt.get_cmap('tab20', len(sightlines))
colors = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.5, 0.0),  # Orange
    (0.5, 0.0, 0.5),  # Purple
    (0.0, 1.0, 1.0),  # Cyan
    (0.5, 0.5, 0.5),  # Gray
    (0.0, 0.0, 0.0),  # Black
    (1.0, 0.0, 1.0),  # Magenta
    (0.6, 0.4, 0.2),  # Brown
    (0.8, 0.8, 0.8)   # Light Gray
]



############################################################################
'''central depth of 6614 vs CH+'''
############################################################################

for i in range(len(sightlines)):


    plt.errorbar(1 - data['Centraldepth_6614'][i], 1 - data['Centraldepth_CHplus'][i], 
             xerr=data['stddev_6614'][i], yerr=data['stddev_CHplus'][i], 
             fmt='o', label=sightlines[i], color = colors[i],
             markersize=5, ecolor='black')  # Set the error bars color to black


# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
correlation_coefficient, p_value = pearsonr(1 - data['Centraldepth_6614'], 1- data['Centraldepth_CHplus'])
plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.075, 0.25))
plt.xlabel('$\lambda$6614 central depth')
plt.ylabel('CH$^+$ central depth')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()


############################################################################
'''central depth of 6614 vs K'''
############################################################################

for i in range(len(sightlines)):


    plt.errorbar(1 - data['Centraldepth_6614'][i], 1 - data['Centraldepth_K'][i], 
             xerr=data['stddev_6614'][i], yerr=data['stddev_K'][i], 
             fmt='o', label=sightlines[i], color = colors[i],
             markersize=5, ecolor='black')  # Set the error bars color to black


# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
correlation_coefficient, p_value = pearsonr(data['Centraldepth_6614'], data['Centraldepth_K'])
plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.075, 0.85))
plt.xlabel('$\lambda$6614 central depth')
plt.ylabel('KI central depth')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()




############################################################################
'''central depth of 6614 vs E(B-V)'''
############################################################################

for i in range(len(sightlines)):


    plt.errorbar(1 - data['Centraldepth_6614'][i],  data['E(B-V)'][i],  
             fmt='o', label=sightlines[i], color = colors[i],
             markersize=5, ecolor='black')  # Set the error bars color to black


# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
correlation_coefficient, p_value = pearsonr(1- data['Centraldepth_6614'], data['E(B-V)'])
plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.075, 0.55))
plt.xlabel('$\lambda$6614 central depth')
plt.ylabel('E(B-V)')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()



############################################################################
'''central depth of CHplus vs E(B-V)'''
############################################################################

for i in range(len(sightlines)):


    plt.errorbar(1 - data['Centraldepth_CHplus'][i],  data['E(B-V)'][i],  
             fmt='o', label=sightlines[i], color = colors[i],
             markersize=5, ecolor='black')  # Set the error bars color to black


# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
correlation_coefficient, p_value = pearsonr(1- data['Centraldepth_CHplus'], data['E(B-V)'])
plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.075, 0.5))
plt.xlabel('CH$^+$ central depth')
plt.ylabel('E(B-V)')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()

############################################################################
'''central depth of K vs E(B-V)'''
############################################################################

for i in range(len(sightlines)):


    plt.errorbar(1 - data['Centraldepth_K'][i],  data['E(B-V)'][i],  
             fmt='o', label=sightlines[i], color = colors[i],
             markersize=5, ecolor='black')  # Set the error bars color to black


# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
correlation_coefficient, p_value = pearsonr(data['Centraldepth_K'], data['E(B-V)'])
plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.3, 0.5))
plt.xlabel('KI central depth')
plt.ylabel('E(B-V)')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()

############################################################################
'''central depth of 6614 vs f(H2)'''
############################################################################

for i in range(len(sightlines)):

    if  data['fH2'][i] != 0:

        plt.errorbar(1 - data['Centraldepth_6614'][i], data['fH2'][i], 
                 fmt='o', label=sightlines[i], color = colors[i],
                 markersize=5, ecolor='black')  # Set the error bars color to black


# Place the legend outside the plot box
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# correlation_coefficient, p_value = pearsonr(data['Centraldepth_6614'], data['fH2'][i])
# plt.annotate(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}", xy = (0.075, 0.85))
plt.xlabel('$\lambda$6614 central depth')
plt.ylabel('f(H2)')
plt.tight_layout()  # Adjust layout to make space for the legend
plt.show()
