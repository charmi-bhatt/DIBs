#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:30:46 2023

@author: charmibhatt
"""

report = """
# fitting method   = leastsq
# function evals   = 517
# data points      = 175
# variables        = 24
chi-square         = 3711.59282
reduced chi-square = 24.5800849
Akaike info crit   = 582.525324
Bayesian info crit = 658.480187
R-squared          = -135649.735
[[Variables]]
B:        0.00331770 (init = 0.005)
delta_B: -0.02653625 (init = -0.053)
zeta:    -0.11820260 (init = -0.197)
T1:       84.0040936 (init = 67)
T2:       82.7823542 (init = 67)
T3:       87.7684260 (init = 67)
T4:       91.5297879 (init = 67)
T5:       82.8827974 (init = 67)
T6:       74.8162735 (init = 67)
T7:       79.8058634 (init = 67)
sigma1:   0.12700000 (init = 0.127)
sigma2:   0.12700000 (init = 0.127)
sigma3:   0.12700000 (init = 0.127)
sigma4:   0.12700000 (init = 0.127)
sigma5:   0.12700000 (init = 0.127)
sigma6:   0.12700000 (init = 0.127)
sigma7:   0.12700000 (init = 0.127)
origin1:  0.03506386 (init = 0.061)
origin2:  0.00151502 (init = 0.061)
origin3: -0.06038301 (init = 0.061)
origin4: -0.03555046 (init = 0.061)
origin5:  0.06507949 (init = 0.061)
origin6:  0.07865156 (init = 0.061)
origin7:  0.01477863 (init = 0.061)


"""

#Save the report to a text file
with open('Cami_2004_fitting_7_sightlines_same_sigma.txt', 'w') as file:
    file.write(report)
    



   # Define a class to store the variables
class Variables:
 def __init__(self):
     self.variables = {}

 def __getattr__(self, name):
     return self.variables.get(name, None)

# Create an instance of the Variables class
v = Variables()

# Open the lmfit report file and read its content
# with open('Cami_2004_fitting_7_sightlines_same_sigma.txt', 'r') as file:
#     lines = file.readlines()
#     parsing_variables = False
#     for line in lines:
#         if line.startswith("[[Variables]]"):
#             parsing_variables = True
#         elif parsing_variables and not line.startswith("#") and line.strip() != "":
#             key_value = line.split(":")
#             key = key_value[0].strip()
#             if len(key_value) == 2:
#                 value = key_value[1].split("+/-")[0].strip()
#                 v.variables[key] = float(value)
#             else:
#                 v.variables[key] = None

variables = {}
import re
with open('Cami_2004_fitting_7_sightlines_same_sigma.txt', 'r') as file:
    lines = file.readlines()
    parsing_variables = False
    for line in lines:
        if line.startswith("[[Variables]]"):
            parsing_variables = True
        elif parsing_variables and not line.startswith("#") and line.strip() != "":
            key_value = line.split(":")
            key = key_value[0].strip()
            value = re.search(r'\d+\.\d+', key_value[1]).group()
            variables[key] = float(value)

B = variables.get("B")
delta_B = variables.get("delta_B")

print("B:", B)
print("delta_B:", delta_B)
# Access the variables later
print(v.T1)  # Access T1
print(v.B)  # Access B
print(v.delta_B)  # Access delta_B
print(v.zeta)  # Access zeta
# ... access other variables as needed
