# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:35:31 2022

@author: Charmi Bhatt
"""

import numpy as np
import matplotlib.pyplot as plt


T_for_B_ofive = np.linspace(10,100,10)
peak_sep_B_ofive = [2.1202752752752687, 3.017314814814803, 3.751256256256241, 4.322099599599582, 4.811393893893879, 5.300688188188175, 5.708433433433413, 6.116178678678658, 6.442374874874858, 6.850120120120096]

peak_sep_pq_B_ofive = np.array([1.0601376376376308, 1.5494319319319274, 1.7940790790790757, 2.1202752752752687, 2.364922422422417, 2.6095695695695653, 2.772667667667662, 3.01731481481481, 3.1804129129129066, 3.343511011011003])
peak_sep_qr_B_ofive = np.array([1.060137637637638, 1.4678828828828756, 1.957177177177165, 2.2018243243243134, 2.4464714714714617, 2.69111861861861, 2.935765765765751, 3.0988638638638477, 3.2619619619619513, 3.5066091091090925])

'====================================='
T_for_B_oone = np.linspace(20,100,9)
peak_sep_B_oone = [1.253798798798785, 1.629938438438419, 1.9165210210209995, 2.149369369369345, 2.364306306306278, 2.543420420420391, 2.7225345345345024, 2.883737237237204, 3.044939939939904]

peak_sep_pq_B_oone = np.array([0.6268993993993917, 0.8239249249249152, 0.9672162162162046, 1.074684684684672, 1.1821531531531386, 1.2717102102101947, 1.3612672672672508, 1.4508243243243069, 1.5224699699699515])
peak_sep_qr_B_oone = np.array([0.6268993993993934, 0.8060135135135038, 0.949304804804795, 1.074684684684673, 1.1821531531531395, 1.2717102102101965, 1.3612672672672517, 1.4329129129128972, 1.5224699699699524])

'======================================'
T_for_B_oofive = np.linspace(40,100,7)

peak_sep_B_oofive = [1.254545045045032, 1.4735925925925768, 1.6328998998998836, 1.7822505005004823, 1.9116876876876683, 2.041124874874853, 2.1606053553553326]
peak_sep_pq_B_oofive = np.array([0.6272725225225155, 0.736796296296288, 0.8164499499499414, 0.8961036036035939, 0.9558438438438337, 1.0255407907907794, 1.0852810310310192])
peak_sep_qr_B_oofive = np.array([0.6272725225225164, 0.7367962962962888, 0.8164499499499422, 0.8861468968968884, 0.9558438438438346, 1.0155840840840735, 1.0753243243243134])



a = peak_sep_qr_B_oofive - peak_sep_pq_B_oofive
b = peak_sep_qr_B_oone - peak_sep_pq_B_oone
c = peak_sep_qr_B_ofive - peak_sep_pq_B_ofive



#plt.plot(T_for_B_ofive, c, label = 'B = 0.05')
plt.plot(T_for_B_oone, b, label = 'B = 0.01')
plt.plot(T_for_B_oofive, a, label = 'B = 0.005')

# plt.plot(T_for_B_oone, peak_sep_pq_B_oone, label = 'PQ separation')
# plt.plot(T_for_B_oone, peak_sep_qr_B_oone, label = 'QR separation')

plt.xlabel('Temperature (K)')
#plt.ylabel('(Vr -Vq) - (Vq-Vp) in cm-1')
plt.ylabel('Peak Separation (cm-1)')
plt.title('B = 0.05 cm-1')
plt.legend()

print(max(c))
print(min(c))
# print(a)
# print(b)
print(c)





'''data'''
# T_for_B_ofive = np.linspace(10,100,10)
# peak_sep_B_ofive = [2.1202752752752687, 3.017314814814803, 3.751256256256241, 4.322099599599582, 4.811393893893879, 5.300688188188175, 5.708433433433413, 6.116178678678658, 6.442374874874858, 6.850120120120096]

# T_for_B_oone = np.linspace(20,100,9)
# peak_sep_B_oone = [1.253798798798785, 1.629938438438419, 1.9165210210209995, 2.149369369369345, 2.364306306306278, 2.543420420420391, 2.7225345345345024, 2.883737237237204, 3.044939939939904]

# peak_sep_pq_B_oone = [0.6268993993993917, 0.8239249249249152, 0.9672162162162046, 1.074684684684672, 1.1821531531531386, 1.2717102102101947, 1.3612672672672508, 1.4508243243243069, 1.5224699699699515]
# peak_sep_qr_B_oone = [0.6268993993993934, 0.8060135135135038, 0.949304804804795, 1.074684684684673, 1.1821531531531395, 1.2717102102101965, 1.3612672672672517, 1.4329129129128972, 1.5224699699699524]

# T_for_B_oofive = np.linspace(40,100,7)

# peak_sep_B_oofive = [1.254545045045032, 1.4735925925925768, 1.6328998998998836, 1.7822505005004823, 1.9116876876876683, 2.041124874874853, 2.1606053553553326]
# peak_sep_pq_B_oofive = [0.6272725225225155, 0.736796296296288, 0.8164499499499414, 0.8961036036035939, 0.9558438438438337, 1.0255407907907794, 1.0852810310310192]
# peak_sep_qr_B_oofive = [0.6272725225225164, 0.7367962962962888, 0.8164499499499422, 0.8861468968968884, 0.9558438438438346, 1.0155840840840735, 1.0753243243243134]

# plt.figure(figsize=(15,6))
# plt.plot(T_for_B_ofive, peak_sep_B_ofive, marker = 'o', label = 'B = 0.05')
# plt.plot(T_for_B_oone, peak_sep_B_oone, marker = 'o', label = 'B = 0.01')
# plt.plot(T_for_B_oofive, peak_sep_B_oofive, marker = 'o', label = 'B = 0.005')
# plt.axhline(y = 1.46, color = 'black', linestyle = '-')
# plt.axhline(y = 1.27, color = 'black', linestyle = '-')
# plt.ylim(1,2)
# plt.xlabel('Peak Separation (cm-1)')
# plt.ylabel('Temperature (K)')
# plt.title('Peak Seperation PR vs Temperature')

# plt.legend()