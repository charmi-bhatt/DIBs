#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:17:44 2024

@author: charmibhatt
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

linewidth_data_filepath = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/fitting_CHplus_and_K_lines/Linewdiths_of_6614_CHplus_and_K.csv'

linewidth_data = pd.read_csv(linewidth_data_filepath)

print(linewidth_data)
CHplus_linewidths = linewidth_data['CH+ plus linewidth (km/s)']
K_linewidths = linewidth_data['K line width (km/s)']
DIB_6614_linewidths = linewidth_data['DIB 6614 linewidth (km/s)']


# Calculate Pearson correlation coefficient and p-value for CH+ vs DIB 6614
r_ch_dib, _ = pearsonr(CHplus_linewidths, DIB_6614_linewidths)
print(f"Pearson correlation coefficient (CH+ vs DIB 6614): {r_ch_dib:.3f}")

# Calculate Pearson correlation coefficient and p-value for K vs DIB 6614
r_k_dib, _ = pearsonr(K_linewidths, DIB_6614_linewidths)
print(f"Pearson correlation coefficient (K vs DIB 6614): {r_k_dib:.3f}")


for i in CHplus_linewidths:
    square = i**2
    sum_all =+ square
    
print(sum_all / len(CHplus_linewidths))

def average_linewidth(linewidths):
    mean = np.mean(linewidths)
    std_dev = np.std(linewidths)
    
    print(std_dev)
    
    sem = std_dev / np.sqrt(len(linewidths))
    
    return mean, sem
    
mean, sem = average_linewidth(DIB_6614_linewidths)
print(f' Average linewidth of 6614 DIB is {mean:.2f} +/- {sem:.2f} km/s')
    

mean, sem = average_linewidth(CHplus_linewidths)
print(f' Average linewidth of CH+ is {mean:.2f} +/- {sem:.2f} km/s')
    

mean, sem = average_linewidth(K_linewidths)
print(f' Average linewidth of K is {mean:.2f} +/- {sem:.2f} km/s')
    
    


# # Data
# CH_plus_linewidth = np.array([2.84969648, 2.37103237, 2.92730933, 2.06643817, 5.5568805,
#                               2.39629501, 2.32833292, 3.74302174, 2.75858844, 4.92778509,
#                               3.32855634, 2.87535421])
# K_linewidth = np.array([1.68073751, 0.69894496, 2.41145166, 0.26847047, 4.35105824,
#                         0.61783659, 0.43697302, 1.98677139, 0.31944249, 2.40079731,
#                         0.74155698, 1.39425543])
# DIB_6614_linewidth = np.array([3.231536, 3.514822, 3.271392, 3.198238, 3.86385,
#                                2.828011, 3.039859, 3.433244, 3.527464, 3.887516,
#                                4.220369, 3.311124])


# import numpy as np

# # Example data
# values = CH_plus_linewidth
# #uncertainties = np.array([0.5, 0.4, 0.6, 0.5, 0.3])  # your uncertainties

# # Calculate the average of the values
# average_values = np.mean(values)

# # Calculate the standard deviation of the values
# std_dev = np.std(values, ddof=1)  # ddof=1 for sample standard deviation

# # Calculate the number of observations
# n = len(values)

# # Calculate the Standard Error of the Mean (SEM)
# sem = std_dev / np.sqrt(n)

# # Print results
# print(f"Average of values: {average_values}")
# print(f"Estimated uncertainty in the average: {sem}")



# # Calculate Pearson correlation coefficient and p-value for CH+ vs DIB 6614
# r_ch_dib, _ = pearsonr(CH_plus_linewidth, DIB_6614_linewidth)
# print(f"Pearson correlation coefficient (CH+ vs DIB 6614): {r_ch_dib:.3f}")

# # Calculate Pearson correlation coefficient and p-value for K vs DIB 6614
# r_k_dib, _ = pearsonr(K_linewidth, DIB_6614_linewidth)
# print(f"Pearson correlation coefficient (K vs DIB 6614): {r_k_dib:.3f}")

# print(f' Avg. of DIB 6614 linwidth: {np.mean(DIB_6614_linewidth)}')
# print(f' Avg. of CH+ linwidth: {np.mean(CH_plus_linewidth)}')
# print(f' Avg. of K linwidth: {np.mean(K_linewidth)}')

