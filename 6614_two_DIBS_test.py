#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 05:59:47 2023

@author: charmibhatt
"""

from LPS_functions import get_rotational_spectrum
from LPS_functions import allowed_perperndicular_transitions
import pandas as pd
import numpy as np
import pandas as pd
import astropy.constants as const
from matplotlib import pyplot as plt

B = 0.00320092
delta_B = -0.10171446
zeta = -0.28852371
sigma = 0.197
origin = 0.06

Jmax = 1000

combinations = allowed_perperndicular_transitions(Jmax)


low_T = 59.41
high_T = 70.5

right_origin = 0.06
left_origin = -0.05

linelist, model_data_low_T = get_rotational_spectrum(B, delta_B, zeta, low_T, sigma, right_origin, combinations)
linelist, model_data_high_T = get_rotational_spectrum(B, delta_B, zeta, high_T, sigma, left_origin, combinations)

plt.plot(model_data_low_T[:,0], model_data_low_T[:,1], label = low_T)
plt.plot(model_data_high_T[:,0], model_data_high_T[:,1], label = high_T)

combined_model_y_data = (model_data_high_T[:,1] + model_data_low_T[:,1])

# original_min = np.min(combined_model_y_data)
# original_max = np.max(combined_model_y_data)
# new_min = 0.9
# new_max = 1.0

# # Function to transform the data
# def transform_data(data, original_min, original_max, new_min, new_max):
#     original_range = original_max - original_min
#     new_range = new_max - new_min
#     normalized_data = (data - original_min) / original_range
#     return (normalized_data * new_range) + new_min


# transformed_y_data = transform_data(combined_model_y_data, original_min, original_max, new_min, new_max)


transformed_y_data = (combined_model_y_data - min(combined_model_y_data)) / (1 - min(combined_model_y_data)) * 0.1 + 0.9   

plt.plot(model_data_high_T[:,0], transformed_y_data)

plt.legend()
plt.xlim(-3, 3)









