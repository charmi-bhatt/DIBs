import numpy as np
import pandas as pd
import astropy.constants as const
import matplotlib.pyplot as plt
import timeit
import scipy.stats as ss
from scipy.signal import argrelextrema
from matplotlib import cm
import numpy as np
import pandas as pd
import astropy.constants as const
import matplotlib.pyplot as plt
import timeit
import scipy as sp
import scipy.stats as ss
from lmfit import Model
import csv
import lmfit
from lmfit import minimize, Parameters, report_fit 
import uncertainties as unc
import uncertainties.umath as umath                                                                          

data = pd.read_csv('/Users/charmibhatt/Desktop/Local_GitHub/edibles/edibles/utils/simulations/Charmi/fitting methods/individual_best_fits.csv') #, encoding = 'latin-1')

PR_sep_value = list(data['PR_sep'])
PR_sep_unc = list(data['PR_sep_unc'])
PR_sep = [unc.ufloat(value, uncertainty) for value, uncertainty in zip(PR_sep_value, PR_sep_unc)]
print(PR_sep)

B = unc.ufloat(0.0222, 0.0089)

Temp = []
for i in range(len(PR_sep)):
    T = 0.180 * (PR_sep[i]** 2 / B)
    np.set_printoptions(precision=3)

    print(T)
    Temp.append(T)
    
#Temp = np.array(Temp)

#np.set_printoptions(formatter={'float': '{:0.3f}'.format})
print(['{:.3f}'.format(x) for x in Temp])
      
      
      
print(data['T'] * data['sigma'])      
      