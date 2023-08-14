#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:25:39 2023

@author: charmibhatt
"""

import numpy as np

def calculate_pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Both lists should have the same length")

    correlation = np.corrcoef(x, y)[0, 1]
    return correlation

# Example data
x =[77.6051193, 84.3844641, 87.2852205, 93.3360293, 83.8822524, 79.5465432, 87.9420139, 80.2861379, 86.0165888, 79.6913374, 86.5009653, 79.6918109]

y  = [0.17978137, 0.19617752, 0.18346550, 0.18017502, 0.21097759, 0.15886251, 0.17259367, 0.19082668, 0.20134184, 0.21524517, 0.23758045, 0.18463089]


print("Correlation coefficient:", calculate_pearson_correlation(x, y))



def equivalent_width(wavelength, flux, continuum_level):
    """
    Calculate the equivalent width of a spectral line.

    Parameters:
    - wavelength: Array of wavelength values.
    - flux: Array of flux values corresponding to the wavelengths.
    - continuum_level: The continuum level to subtract from the flux.

    Returns:
    - Equivalent width of the spectral line.
    """

    # Subtract the continuum
    line_flux = continuum_level - flux

    # Integrate using the trapezoidal rule
    area = np.trapz(line_flux, wavelength)

    # Calculate the equivalent width
    EW = area / continuum_level

    return EW

# Example data
wavelengths = np.array([1, 2, 3, 4, 5])
fluxes = np.array([1, 0.5, 0.1, 0.5, 1])
continuum = 1.0

print("Equivalent Width:", equivalent_width(wavelengths, fluxes, continuum), "units of wavelength")
