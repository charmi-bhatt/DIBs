# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from astropy import constants as const
import matplotlib.pyplot as plt

#Input constants:
    
ground_B = 0.0111
ground_C = 0.005552
T = 2

h = const.h.cgs.value
c = const.c.to('cm/s').value
k = const.k_B.cgs.value

#creating empty arrays: 
    
HL_Factor_pP_Branch = []
HL_Factor_rP_Branch = []
HL_Factor_pQ_Branch = []
HL_Factor_rQ_Branch = []
HL_Factor_pR_Branch = []
HL_Factor_rR_Branch = []
            
BD_pP_Branch = []
BD_rP_Branch = []
BD_pQ_Branch = []
BD_rQ_Branch = []
BD_pR_Branch = []
BD_rR_Branch = []

Intensity_pP_Branch = []
Intensity_rP_Branch = []
Intensity_pQ_Branch = []
Intensity_rQ_Branch = []
Intensity_pR_Branch = []
Intensity_rR_Branch = []

deviation_pP = []
deviation_rP = []
deviation_pQ = []
deviation_rQ = []
deviation_pR = []
deviation_rR = []

Energy_pP = []
Energy_rP = []
Energy_pQ = []
Energy_rQ = []
Energy_pR = []
Energy_rR = []

Energy = (Energy_pP, Energy_rP, Energy_pQ, Energy_rQ, Energy_pR, Energy_rR)
HL_factor = [HL_Factor_pP_Branch, HL_Factor_rP_Branch, HL_Factor_pQ_Branch, HL_Factor_rQ_Branch, HL_Factor_pR_Branch, HL_Factor_rR_Branch]
BD_factor = (BD_pP_Branch, BD_rP_Branch, BD_pQ_Branch, BD_rQ_Branch, BD_pR_Branch, BD_rR_Branch)
Intensities = (Intensity_pP_Branch, Intensity_rP_Branch, Intensity_pQ_Branch, Intensity_rQ_Branch, Intensity_pR_Branch, Intensity_rR_Branch)
deviation = (deviation_pP, deviation_rP, deviation_pQ, deviation_rQ, deviation_pR, deviation_rR)
label = ['pP', 'rP', 'pQ', 'rQ', 'pR', 'rR']
number_of_branches = [0,1,2,3,4,5]




#reading in PGopher file

#kmax = 1
#coronene = pd.read_csv(r"C:\Users\Charmi Bhatt\Desktop\Pgopher practice plots\Coronene P,Q and R Branches\coronene line lists pgopher\coronene_strengthtest1.txt", delim_whitespace= True)

#kmax = 5
coronene = pd.read_csv(r"C:\Users\Charmi Bhatt\Desktop\Pgopher practice plots\Coronene P,Q and R Branches\coronene line lists pgopher\coronene_strengthtest2_k5.txt", delim_whitespace= True)





'''>>>>>>>>>>>>>>>>>> Defining P, Q and R Branches <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'''

#sorting by value of J in ascending order
coronene = coronene.sort_values(by=['J'], ascending=True)

coronene = coronene[(coronene['K'] != 0)] 

P_Branch = coronene[(coronene['label'].str[1] == "P")]
Q_Branch = coronene[(coronene['label'].str[1] == "Q")]
R_Branch = coronene[(coronene['label'].str[1] == "R")]


pP_Branch = coronene[(coronene['label'].str[1] == "P") & (coronene['label'].str[0] == "p")]
rP_Branch = coronene[(coronene['label'].str[1] == "P") & (coronene['label'].str[0] == "r")]

pQ_Branch = coronene[(coronene['label'].str[1] == "Q") & (coronene['label'].str[0] == "p")]
rQ_Branch = coronene[(coronene['label'].str[1] == "Q") & (coronene['label'].str[0] == "r")]

pR_Branch = coronene[(coronene['label'].str[1] == "R") & (coronene['label'].str[0] == "p")]
rR_Branch = coronene[(coronene['label'].str[1] == "R") & (coronene['label'].str[0] == "r")]
rR_Branch = rR_Branch.iloc[1: , :]

Branches = (pP_Branch, rP_Branch, pQ_Branch, rQ_Branch, pR_Branch, rR_Branch)

lines = []
index_of_lines = []
for n in number_of_branches:
    length = len(Branches[n])
    lines.append(length)
    

'''Honl-London Factor'''

for n in number_of_branches:
        for J, K in zip(Branches[n]['J'], Branches[n]['K']): 
            #if K != 0:
                HL_equation_for_pP = ((J - 1 + K)*(J + K))/(2*J*(J+1))
                HL_equation_for_rP = ((J - 1 - K)*(J - K))/(2*J*(J+1))
                HL_equation_for_pQ = (J+1-K)*(J+K)/ (2*J*(J + 1))
                HL_equation_for_rQ = (J+1+K)*(J-K) / (2*J*(J + 1))
                HL_equation_for_pR = ( J + 2 - K)*( J + 1 - K)/ (2*(J + 1)*(2*J +1))
                HL_equation_for_rR = ( J + 2 + K)*( J + 1 + K)/ (2*(J + 1)*(2*J +1))
                
                HL_equations = (HL_equation_for_pP, HL_equation_for_rP, HL_equation_for_pQ, HL_equation_for_rQ, HL_equation_for_pR, HL_equation_for_rR)
    
                HL_factor[n] = np.append(HL_factor[n], HL_equations[n])
            
            

'''BD_factor'''

for n in number_of_branches:
    for J, K in zip(Branches[n]['J'], Branches[n]['K']):
                E = ground_B*J*(J + 1) + (ground_C - ground_B)*(K**2)
                Energy[n].append(E)
                boltzmann_equation = ((2*J) + 1)*(np.exp((-h * c * E ) / (k*T)))
                BD_factor[n].append(boltzmann_equation)

'''Intensity'''

for n,l in zip(number_of_branches, lines):   
    #if n == 5:
        index = list(range(l))
        for i in index:
            Intensity_equaltion =  HL_factor[n][i] * BD_factor[n][i]
            Intensities[n].append(Intensity_equaltion) # / (max(Intensities[n])))

        
'''Deviation'''
        
for n,l in zip(number_of_branches, lines):   
    #if n ==0:
        index = list(range(l))
        for i in index:
            #ratio
            ratio = ((Intensities[n][i]) / Branches[n]['strength'][i])
            
            # difference
            # Normalized_Intenisty = (Intensities[n][i] /(max(Intensities[n]))) 
            # Normalized_Pgopher_strength = (Branches[n]['strength'][i]/(max(Branches[n]['strength'])))
            # difference = (Normalized_Intenisty - Normalized_Pgopher_strength)
            
            deviation[n].append(ratio)
            
'''Plotting Deviation'''

for n in number_of_branches:
   #if n == 5:
        plt.scatter(Branches[n]['J'], deviation[n], label = label[n])
        plt.xlabel('J')
        plt.ylabel('deviation')
        plt.legend()
        # for i, l in enumerate(Branches[n]['label']):
        #         plt.annotate(Branches[n]['K'][i], (Branches[n]['J'][i], deviation[n][i]+ 0.05))
        #         plt.legend() 
            
    
'''Linelist'''

for n in number_of_branches:
    if n == 0:
            table_contents = { 'J' : Branches[n]['J'],
                              'K' : Branches[n]['K'],
                  'HL_factor' : HL_factor[n],
                  'BD_factor' : BD_factor[n],
                  'Energy' : Energy[n],
                  'Intenisty' : Intensities[n],
                  "PGopher strength" : Branches[n]['strength'],
                  "deviation": deviation[n]}
        
            linelist = pd.DataFrame(data = table_contents)
            
            print('--------------------------------')
            print(label[n])
            print('--------------------------------')
            print(linelist.to_string())
            print('--------------------------------')
 
    
''' Stem Plots'''

for n in number_of_branches:
    #if n == 0:    
        fig, ax = plt.subplots(figsize=(25,6))
        ax.set(title = "Coronene  " + label[n] + "  Branch Calcuated",
                xlabel = "Wavenumber",
                ylabel = "Normalized Inetensity")
            
        ax.stem(Branches[n]['position'], Intensities[n]/(max(Intensities[n])),linefmt=('red'), markerfmt= 'ro', label = [n])
        ax.stem(Branches[n]['position'], Branches[n]['strength']/(max(Branches[n]['strength'])),linefmt=('blue'), markerfmt= 'bo', label = [n])

        plt.show()
        


        
        

                                                   
                       
                           