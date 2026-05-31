# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:14:49 2023

@author: Pep Rubi
"""

import numpy as np
import mpmath as mp
import math as m
from Model_delta_t_E import *
from calculation_angles import *
from Change_dimensions import *
from xsection_jacobians_stuff import *
from f_function import *
from phases import *
from Anisotropy import *
from setup import SetUp
from constants import *
from Spectrum import *
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import scienceplots
import os

SetUp()

debug = False
parallalize = True
n_jobs = 10

mp.mp.dps =  50  # number of digits for internal calculation

Eunit = u.keV     # energy unit to which the results are getting referred to

R0 =  0.9  # FIXME 0.9  # Initial radii from which the positrons start to accelerate
RLI = 10   # Upper integracion limit
Rf =  2    # Final radius up to which the positrons are getting accelerated (in units of RLC)

delta_R = 0.1   # Step width for the integral (in units of RLC)

R_arr  = np.arange(R0, RLI, delta_R)  # array of distances w.r.t . the NS, in units of RLC
a_arr  = np.arcsin(1/R_arr)           # array angles obtained from the simplification for R>>RLC
w_arr  = 1.-np.cos(a_arr)             # weights for the efficiency of the ICS 

alpha = 1         # power-law evolution index of Gammas along the current sheet WE WANT TO VARY BETWEEN 0.5 and 10
gamma_w = 6*10**7 # corresponds to Gamma_w * m_e * c^2 = 30 TeV, the maximum possible with HESS Vela data
gamma_0 = 300     # Initial gamma factor of the  positrons when the enter the current sheet
E0 = 1*Eunit      # pivot energy, 1 keV

epsilon_pulse_arr = [0.5, 1.3, 3.0, 7.0, 12.0, 27.0, 65.0, 170.0]*(u.keV) # Energies for the pulse profiles

log_epsilon_max = 8     # Maximum of CR emission, according to Cao and Yang, in terms of log10(epsilon/E0)
log_epsilon_min = -5    # Minimum of CR emission, according to Cao and Yang, in terms of log10(epsilon/E0)
log_epsilon_bins= 100   # 100  # was 30 number of logarithmically-space bins of epsilon
epsilon_arr = np.logspace(log_epsilon_min,log_epsilon_max,log_epsilon_bins)*(Eunit) # Array of initial photon energies, logarithmically spaced, BUT LINEAR

epsilon_mean = np.sqrt(epsilon_arr[1:]*epsilon_arr[:-1]) # Mean energies of incident photons, logarithmically spaced, BUT LINEAR!!
Delta_epsilon = epsilon_arr[1:] - epsilon_arr[:-1]       # Energy spacing of the incident photons, logarithmically spaced, BUT LINEAR!!
Delta_epsilon_log = np.log((epsilon_arr[1:])/E0) - np.log((epsilon_arr[:-1])/E0) # Spacing of natural logarithm of incident photon energies

log_steps = 100 # Number of final photon energies 

log_E_min   = 6   # 10^6 keV --> 1 GeV
log_E_max   = 11  # 10^11 keV --> 100 TeV
log_E_bins  = 100
E_arr       = np.logspace(log_E_min,log_E_max, log_E_bins)*(Eunit) # Array of scattered photon energies, logarithmically spaced, BUT LINEAR!!

E_mean     = np.sqrt(E_arr[1:]*E_arr[:-1])   # Mean energies of scattered photons, logarithmically spaced, BUT LINEAR!!
Delta_E    = E_arr[1:]-E_arr[:-1]            # Energy spacing of scattered photons, logarithmically spaced, BUT LINEAR!!
Delta_E_log = np.log((E_arr[1:])/E0) - np.log((E_arr[:-1])/E0) # Spacing of natural logarithm of scattered photon energies

epsilon_mean_3d = add_dim_e_fi(E_mean, epsilon_mean, R_arr)   # Energia inicials dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f
E_mean_3d       = add_dim_e_ff(E_mean, epsilon_mean, R_arr)   # Energia final dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f

Gamma_arr = gammaw(R_arr, R0, Rf, gamma_0, gamma_w, alpha)  # array of gamma factors for each distance, following the wind model
M_arr     = M(R_arr, R0, Rf, gamma_w, alpha)                # array of angular moments que s'emporten els electrons
Gamma_2d  = add_dimension_R(Gamma_arr, epsilon_mean)        # New array dimension: len(Gamma_arr), len(epsilon_mean)
Gamma_3d  = add_dimension_R(Gamma_2d, E_mean)               # New array dimension: len(Gamma_arr), len(epsilon_mean), len(E_mean)
beta_3d   = beta_f(Gamma_3d, dps=None)                      # Value of beta of the positrons in 3 dimensions
beta_arr  = beta_f(Gamma_arr, dps=None)                     # Value of beta of the positrons in 1 dimension (R)

if debug: 
    print ('R: ', R,'\n')
    print ('Gamma_arr: ', Gamma_arr,'\n')
    print ('M: ', M_arr,'\n')
    print ('Gamma_2d: ', Gamma_2d,'\n')
    print ('Gamma_3d: ', Gamma_3d,'\n')
    print ('beta_3d: ', beta_3d,'\n')
    print ('beta_arr: ', beta_arr,'\n')

#theta_1d = np.arcsin(M_i*c/(Gamma*m*R*RLC))    # Array d'angles de la colisio entre electrons i fotons
#print ('theta_1d: ', theta_1d,'\n')

theta_arr = theta_from_R(R_arr,R0,Rf,RLC,gamma_w,Gamma_arr,beta_arr, alpha)*u.rad  # Array of collision angles, Eq. 2.5
theta_2d  = add_dimension_R(theta_arr, epsilon_mean)       # New array dimension: len(Gamma_arr), len(epsilon_mean)
theta_3d  = add_dimension_R(theta_2d, E_mean)              # New array dimension: len(Gamma_arr), len(epsilon_mean), len(E_mean)

theta_3d = theta_3d*u.rad                                  # Array in 3 dimensions and units, THIS IS THE theta_L in the paper! 

theta_init = theta_init(beta_3d, theta_3d, epsilon_mean_3d, E_mean_3d) # Primera approximacio del que val el valor final de l'angle de dispersió del foto

if debug:
    print ('theta_arr: ', theta_arr,'\n')
    print ('theta_init:', theta_init)


plt.figure()
plt.plot(R_arr, theta_arr, label = r"$\theta_{L} (R)$")
plt.ylabel(r"$\theta_{L}$ (rad)")
plt.xlabel(r"$R (R_{LC})$")
#plt.xscale("log")
#plt.yscale("log")
plt.legend()
plt.savefig('theta_L.png')
if plt.isinteractive():
    plt.show()


    
#I try to compute the exact maximum and minimum of the final energy
def Efotof(Ein, Gamma_arr, beta_3d, thetaL, theta_mm, me):
    # theta_mm is theta_L'   --> need Taylor expansion for theta_mm very small 
    num = Ein*me*Gamma_arr*(1-beta_3d*np.cos(thetaL))
    den = me*Gamma_arr*(1-beta_3d*np.cos(theta_mm)) + Ein*(1-np.cos(thetaL + theta_mm))
    return num/den

def derEfotof(Ein, Gamma_arr, beta_3d, theta, theta_L, me):
    num = -(Gamma_arr*me*beta_3d*np.sin(theta_L) + (np.sin(theta + theta_L))*Ein)
    num2 = Ein*me*Gamma_arr*(1-beta_3d*np.cos(theta))
    den = (beta_3d*np.cos(theta_L)-1)*Gamma_arr*me + (np.cos(theta+theta_L) -1)*Ein
    return num

ind1 = -1

angleplot = np.linspace(0, 2*np.pi, 2000)*u.rad
derfinal = derEfotof(epsilon_mean_3d[ind1][ind1][ind1], Gamma_3d[ind1][ind1][ind1], beta_3d[ind1][ind1][ind1], theta_3d[ind1][ind1][ind1], angleplot, m_keV)
Ef = Efotof(epsilon_mean_3d[ind1][ind1][ind1], Gamma_3d[ind1][ind1][ind1], beta_3d[ind1][ind1][ind1], theta_3d[ind1][ind1][ind1], angleplot, m_keV)

'''   TESTS
gamma_test = Gamma_3d[13,0,0]
beta_test = beta_f(gamma_test, dps= None)
E_in_test = epsilon_mean_3d[13,0,0]
theta_test = theta_3d[13,0,0]
Eout_test = E_mean_3d[13,0,0]
Etest = Eout_test - Efotof(E_in_test,gamma_test, beta_test, theta_test, angleplot ,m_keV)
plt.plot(angleplot, Etest)
plt.yscale("log")

ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.show()

plt.plot(angleplot, derfinal)
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.show()
''' 
theta_fs = np.arctan(-epsilon_mean_3d*np.sin(theta_3d)/(epsilon_mean_3d*np.cos(theta_3d) + m_keV*(Gamma_3d**2 -1)))
theta_ss = theta_fs + np.pi*u.rad

#print ('HERERERE', theta_fs)

E_log_max = Efotof(epsilon_mean_3d, Gamma_3d, beta_3d, theta_3d, theta_fs, m_keV)
E_log_min = Efotof(epsilon_mean_3d, Gamma_3d, beta_3d, theta_3d, theta_ss, m_keV)

#Poso Gamma_3d[0] ja que no em depen de l'energia final del fotó, i per taant no em canvia el resultat quina triï
E_log_max2 = epsilon_mean_3d[0]*m_keV*Gamma_3d[-1]*(1-beta_3d[0]*np.cos(theta_3d[0]))/(m_keV*Gamma_3d[-1]*(1-beta_3d[0]) + epsilon_mean_3d[0]*(1-np.cos(theta_3d[0])))#Energia maxima que els fotons poden assolir, primera aproximacio
E_log_min2 = epsilon_mean_3d[0]*m_keV*Gamma_3d[0]*(1+beta_3d[0]*np.cos(theta_3d[0]))/(m_keV*Gamma_3d[0]*(1+beta_3d[0]) + epsilon_mean_3d[0]*(1+np.cos(theta_3d[0])))#Energia minima que els fotons poden assolir, primera aproximacio

print ('E_f,max: ',E_log_max,' E_f,min: ',E_log_min)

#A partir d'aqui comença un calcul numeric del valor de l'angle de dispersió dels fotons

#theta_f_e = []
#for j in range(len(E_mean)):
#    print(j)
#    auxiliar = []
#    for k in range(len(epsilon_mean)):
#        auxiliar2 = []
#        for i in range(len(R)):
#            if (E_log_max[k][i] > E_mean_3d[j][k][i] > E_log_min[k][i]):
#                theta_fexacte1 = solver(theta_init[j][k][i],theta[j][k][i], Gamma_3d[j][k][i], beta_3d[j][k][i], E_mean_3d[j][k][i], epsilon_mean_3d[j][k][i])
#                theta_fexacte2 = solver(theta_fexacte1,theta_3d[j][k][i], Gamma_3d[j][k][i], beta_3d[j][k][i], E_mean_3d[j][k][i], epsilon_mean_3d[j][k][i])                
#                theta_fexacte3 = solver(theta_fexacte2,theta_3d[j][k][i], Gamma_3d[j][k][i], beta_3d[j][k][i], E_mean_3d[j][k][i], epsilon_mean_3d[j][k][i])                
#                auxiliar2.append(float(theta_fexacte3))
#            else:
#                auxiliar2.append(10000)
#        auxiliar.append(np.array(auxiliar2))
#    theta_f_e.append(np.array(auxiliar))
#theta_f_e = np.array(theta_f_e)*u.rad  #Obtinc un valor 

if parallalize:
    theta_f_e = compute_theta_f_exact_parallel(theta_init, theta_3d, Gamma_3d, beta_3d, E_mean_3d, epsilon_mean_3d, E_log_min, E_log_max, n_jobs=n_jobs)
else: 
    theta_f_e = compute_theta_f_exact(theta_init, theta_3d, Gamma_3d, beta_3d, E_mean_3d, epsilon_mean_3d, E_log_min, E_log_max, fill_value=np.nan)

print ('theta_f_e: ',theta_f_e)

comprovacions = equation_solve(theta_f_e, theta_3d, Gamma_3d, beta_3d, E_mean_3d, epsilon_mean_3d, m)

# Broadcast min/max from (n_Ei, n_R) to (n_Ef, n_Ei, n_R)
valid = (
    (E_mean_3d > E_log_min[:, :, :]) &
    (E_mean_3d < E_log_max[:, :, :])
)
contador = np.count_nonzero(valid)

good = valid & (np.abs(comprovacions / m_keV**2) < 0.1)
contador2 = np.count_nonzero(good)

# Set invalid entries
comprovacions = comprovacions.copy()
comprovacions[~valid] = 99999999 * u.keV * u.keV

#contador = 0
#contador2 = 0

##Aquí comprovo com de precís és el resultat
#for j in range(len(E_mean)):
#    for k in range(len(epsilon)):
#        for i in range(len(R)):
#            if (E_log_max[k][i] > E_mean_3d[j][k][i] > E_log_min[k][i]):
#                contador += 1 
#
#                if abs(comprovacions[j][k][i]/m_keV**2) < 0.1 :
#                    contador2 += 1
#            else:
#                comprovacions[j][k][i] = 99999999*u.keV*u.keV           
print('contadors: ',contador, contador2)
#Contador em dona el numero de bins que tinc entre el range de Energia maxima i Energia minima del foto
#Mentres que contador2 em dona el numero de bins que tinc amb un error menor al 10%
#AQUEST CALCUL S'HAURIA DE MILLORAR JA QUE L'ENERGIA FINAL DELS FOTONS CONVERGEIX POC

#Aquí ha acabt el calcul de l'angle de dispersió dels fotons

#Obtinc la secció eficaç segons el Peskin,
#i li trec les unitats per poder-ho visualitzar millor en el compilador
cross_section_exact = exact_xsection(theta_3d, theta_f_e, beta_3d, Gamma_3d, epsilon_mean_3d, E_mean_3d)
#Declaro les variables de l'espectre que necessito per una energia major a 0.2 keV
#Aquests valors s'obtenen del codi: "phase_a_spectra"
E0 = 1*u.keV
K = 0.488
a = 0.618
b = 0.133


#Declaro les variables de l'espectre que necessito per una energia menor a 0.2 keV
#Aquests valors també s'obtenen del codi: "phase_a_spectra"
K1 = 0.524
a1 = 0.452
b1 = 0.057

#Aquest espectre es més precís
mask = epsilon_mean > 0.2 * u.keV

print ('epsilon: ', epsilon_mean)

print ('mask: ', mask)
###Experimental data 1
#Points that range from 2e-4 MeV to 2e-2 MeV
Interval_1_2 = np.array([0.00018529204335398188, 0.00026826936312485507, 0.000404709199115943, 0.0006628706949378706, 0.0011787679216418198, 0.0017782806652028994, 0.0027952995832933696, 0.004970836558590657, 0.008483426131078615, 0.014478204824082668, 0.02682698680657344, 0.049708318805032674, 0.08483450082550081, 0.13894990437515836, 0.22758518925665774])
Data_1_2 = np.array([0.00019306977288832496, 0.000249359061432569, 0.00031050117634809897, 0.00040102813760005114, 0.0005179474679231213, 0.0006218004555057681, 0.0006689545056200522, 0.0008329802002184533, 0.000896150501946605, 0.000896150501946605, 0.0008639889313244301, 0.0008639889313244301, 0.0008030857221391513, 0.0008030857221391513, 0.0007742641144826989])
#Points that range from 1e-7 MeV to 2e-4 MeV
Interval_1_3 = np.array([1.1787695460435082e-7, 1.9306991565211053e-7, 2.9126328996497e-7, 4.3939680501067657e-7, 6.628707506370985e-7, 0.0000010857117945006027, 0.0000017782804560086247, 0.0000027952989960892988, 0.00000404708702618929, 0.000006361673467383023, 0.000010857102618453848, 0.00001637894422619578, 0.000025746246154698523, 0.00004393972185302389, 0.00006906934166047259, 0.00011312823624679939, 0.0002096177914042753])
Data_1_3 = np.array([6.011005284041015e-7, 8.966678019375663e-7, 0.0000012437607565150496, 0.0000017890865809929578, 0.0000026687987387302526, 0.0000041284737078301905, 0.000005726572952667369, 0.000009186661857373139, 0.000012742745392311902, 0.00001767536050137334, 0.00002636651591589646, 0.000035267021797634925, 0.000052608089317734, 0.00007036692442105561, 0.00010496706036384912, 0.00014040040241817682, 0.00020943671528903202])

#The same, but in keV
Interval_1_2 = Interval_1_2*1e3
Data_1_2 = Data_1_2*1e3

Interval_1_3 = Interval_1_3*1e3
Data_1_3 = Data_1_3*1e3
#Points that range from 1e-4 MeV to 2e1 MeV
Data_tot = np.concatenate((Data_1_3, Data_1_2))
Interval_tot = np.concatenate((Interval_1_3, Interval_1_2))

### Experimental data obtained with plot digitalizer
Interval_x = np.array([0.0003162276172147746, 0.0006628700711058477, 0.001279801052582922, 0.002470912307632171, 0.004970822524341731, 0.010419739772213055, 0.02011739496035263, 0.035774307157273, 0.07498939446834635, 0.15719128418077574, 0.3162285100282012, 0.6105401886168376, 1.1312834556437819, 1.8529221773372084, 4.970827202420301, 12.798058703114059, 20.117413892984658, 32.95018783048523, 66.28706949378679, 127.98046658775885, 227.58476089298526, 388.4066097985313, 749.8967676108324, 1279.803461444897, 2096.1821765480395, 3727.6072514469956, 6628.719426036748])
Interval_y = np.array([0.000719370045023045, 0.0008129133633486966, 0.0008674801800921799, 0.0009142518392550057, 0.0009532283408371741, 0.0009844093279992444, 0.0009922048424193426, 0.0009844093279992444, 0.0009688190128379296, 0.0009532283408371741, 0.0009220473536751038, 0.0008908660096735928, 0.0008362991929301096, 0.0007817323761866263, 0.0007271652026037024, 0.0007583465466052134, 0.0006024408971159803, 0.0005790550675345674, 0.0006336218842780508, 0.0006725983858602192, 0.0006803935434408766, 0.0006881890578609747, 0.0006881890578609747, 0.0006570077138594637, 0.0006258267266973932, 0.000540078565952399, 0.0005088975787903287])

Interval_x = Interval_x*1e3
Interval_y = Interval_y*1e3

### Experimental deviation
Inc_x = np.array([0.0003162276172147746, 0.0003162276172147746, 0.8141694802703412, 0.7498960618779956, 6.361675872880828, 9.59717769319248, 8.83954254202545, 6.628713187704805, 12.282489672705855, 12.282489672705855, 22.758497507472, 20.117413892984658, 35.774340824743255, 35.774340824743255, 2912.6431298426255, 2682.698680657325, 848.3458066388913, 562.3425159442847, 61.05413377853293, 61.05413377853293, 6361.681859901954, 6105.4248695598335])
Inc_y = np.array([0.0006258267266973932, 0.0007661417041858708, 0.0007895275337672837, 0.0008596850225115225, 0.0006414173986981487, 0.0005790550675345674, 0.0007505510321851153, 0.0007973226913479412, 0.000719370045023045, 0.0008207085209293541, 0.0007037793730222895, 0.0005166927363709861, 0.0006725983858602192, 0.00047771659162825837, 0.0006725983858602192, 0.0007661417041858708, 0.000719370045023045, 0.0006414173986981487, 0.0005634643955338119, 0.0006881890578609747, 0.00029842511255761243, 0.0005946457395353229])

Inc_x = Inc_x*1e3
Inc_y = Inc_y*1e3

sigma2 = 0
for i in range(int(len(Inc_y)/2)):
    a = (Inc_y[i] - Inc_y[i+1])/2
    sigma2 += a**2
sigma = np.sqrt(sigma2)

(Sedfit, xi, poptspectrum) = Xi_Fit(Interval_y, Interval_x, 6, sigma)  

plt.figure()
plt.plot(Interval_x, Sedfit, label = "Fitting quadratic")
plt.plot(Interval_tot, Data_tot, '.', label = "Interval 4")
plt.plot(Interval_x, Interval_y, '.', label = "Interval 5")

plt.ylabel(r"$E^2F (keVcm^{-2}s^{-1})$")
plt.xlabel(r"E (keV)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
if plt.isinteractive():
    plt.show()

spec1d2 = SEDfromFIT(np.log10(epsilon_mean/E0), *poptspectrum)

""" I will need to change this with the new polynomial"""
spec_1d = np.empty(len(epsilon_mean)) * (1 / E0).unit
spec_1d[mask] = (
    K * (epsilon_mean[mask] / E0) ** (-a - 1 - b * np.log10(epsilon_mean[mask] / E0)) / (E0)
)
spec_1d[~mask] = (
    K1 * (epsilon_mean[~mask] / E0) ** (-a1 - 1 - b1 * np.log10(epsilon_mean[~mask] / E0)) / E0
)

# Broadcast to shape (len(E_mean), len(epsilon), len(R))
#I change it to the new fiting
spectra = np.broadcast_to(
    spec1d2.reshape(1, len(epsilon_mean), 1),
    (len(E_mean), len(epsilon_mean), len(R_arr))
)*(1/E0).unit


### As this does not have the  right units (which should be MeV/(cm^2 s))
### we expect the final result, i.e., the second integral to have the same units

plt.clf()
plt.plot(epsilon_mean, spec_1d, '.', label = "Original fit")
plt.plot(epsilon_mean, spec1d2, '.', label = "Polynomial fit")
plt.plot(Interval_x, Interval_y, '.', label = "Data")

plt.ylabel(r"$E^2F (keVcm^{-2}s^{-1})$")
plt.xlabel(r"E (keV)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
if plt.isinteractive():
    plt.show()


print ('spectra: ', spectra)

#spectra = []
#for j in range(len(E_mean)):
#    auxiliar1 = []
#    for k in range(len(epsilon)):
#        auxiliar2 = []
#        for i in range(len(R)):
#            if epsilon[k] > 0.2*u.keV:
#                auxiliar = K*(epsilon[k]/E0)**(-a-1-b*np.log10(epsilon[k]/E0))/E0.value
#            elif epsilon[k] < 0.2*u.keV:
#               auxiliar = K1*(epsilon[k]/E0)**(-a1-1-b1*np.log10(epsilon[k]/E0))/E0.value
#           auxiliar2.append(auxiliar)
#        auxiliar1.append(np.array(auxiliar2))
#    spectra.append(np.array(auxiliar1))
#spectra = np.array(spectra)*(1/E0).unit

#Inicialtzo la variable de la fase
delta_phase = 0.004
phase = np.arange(0.2,0.5,delta_phase)
phase_3d = add_dim_phase(phase, E_mean, R_arr) #El poso en les dimensions que em conve
time = phase_3d*P


#Canvio les variables segons les dimensions que em conve
R_2d = add_dimension_R(R_arr, E_mean)
R_3d = add_dimension_R(R_2d, phase)

#Busco els parametres dels pulse profiles
resultsLor = adjust_as_lor(phases, fs)

X0s = resultsLor["x0"]
sigmes1 = resultsLor["sigma_left"]
sigmes2 = resultsLor["sigma_right"]
As = resultsLor["A"]
Cs = resultsLor["C"]
pcov = resultsLor["cov"]

print ('X0s: ',X0s, ' sigmes1: ', sigmes1,' sigmes2: ', sigmes2, ' As: ', As, ' Cs: ',Cs, ' pcov: ', pcov)


#funció que em treu un valor i el seu error directament per passar-ho a latex
def display_sigfig(x, xerr, sigfigs=2) -> str:
    '''
    Suppose we want to show 2 significant figures. Implicitly we want to show 3 bits of information:
    - The order of magnitude
    - Significant digit #1
    - Significant digit #2
    '''

    if sigfigs < 1:
        raise Exception('Cannot have fewer than 1 significant figures. ({} given)'.format(sigfigs))

    order_of_magnitude = np.floor(np.log10(np.abs(xerr)))

    # Because we get one sigfig for free, to the left of the decimal
    decimals = (sigfigs - 1)

    xerr /= np.power(10, order_of_magnitude)
    xerr = np.round(xerr, decimals)
    xerr *= np.power(10, order_of_magnitude)

    # Subtract from decimals the sigfigs we get from the order of magnitude
    decimals -= order_of_magnitude
    # But we can't have a negative number of decimals
    decimals = int(max(0, decimals))

    return '{:.{dec}f}'.format(x, dec=decimals)+'$\pm${:.{dec}f}'.format(xerr, dec=decimals)
"""
#Bucle per passar coses a Latex
for i in range(len(X0s)):
    print("GRAFICA"+str(i+1))
    cov = pcov[i]
    print("$x_0$ = ", display_sigfig(X0s[i], np.sqrt(cov[0][0]))+",")
    print("$\sigma_1$ = ", display_sigfig(sigmes1[i], np.sqrt(cov[1][1]))+",")
    print("$\sigma_2$ = ", display_sigfig(sigmes2[i], np.sqrt(cov[2][2]))+",")
    print("$A = $", display_sigfig(As[i], np.sqrt(cov[3][3]))+",")
    print("$C = $", display_sigfig(Cs[i], np.sqrt(cov[4][4]))+".")
"""

        
#Declaro una funcio pel temps dels Pulse Profiles
def Time_d(time, R, theta_3d):
    return time - R*(1-np.cos(theta_3d))*RLC/c + theta_3d.value/Omega

#A partir d'aquí re faig les dues primeres integrals però considerant que els pulse profiles depenen de l'energia


n_phase = len(phase)
n_ef = len(E_mean)
n_ei = len(epsilon_mean)
n_r = len(R_arr)

# Output without the redundant E_mean dimension first
base = np.empty((n_phase, n_ei, n_r), dtype=float)

# Precompute which fitted parameter set corresponds to each epsilon[i]
# idx[i] = first index n such that epsilon[i] < epsilon_pulse_arr[n]
idx = np.searchsorted(epsilon_pulse_arr, epsilon_mean, side="right")

# Clamp in case some epsilon are above all epsilon_pulse_arr values
idx = np.clip(idx, 0, len(epsilon_pulse_arr) - 1)

for l, ph in enumerate(phase):
    # Time_d depends only on phase[l] and R
    temps = Time_d(ph * P, R_arr, theta_arr)    
    #temps = np.array([Time_d(ph * P, r, th) for r, th in zip(R, theta_arr)])

    for i in range(n_ei):
        n = idx[i]
        base[l, i, :] = funct_f(
            temps,
            X0s[n],
            sigmes1[n],
            sigmes2[n],
            As[n],
            Cs[n]
        )

# Broadcast over E_mean, since the result does not depend on j
Funct = np.broadcast_to(base[:, None, :, :], (n_phase, n_ef, n_ei, n_r)).copy()

print ('Funct: ', Funct)

#Els pulse profiles mes precisos en funcio de l'energia
#Funct = []
#for l in range(len(phase)):
#    print(l)
#    auxiliar3 = []
#    for j in range(len(E_mean)):
#        auxiliar2 = []
#        for i in range(len(epsilon_mean)):
#            auxiliar = []
##            for k in range(len(R)):
#                temps_pt = Time_d(phase[l]*P, R[k], theta_arr[k])
#                for ñ in range(len(epsilon_pulse_arr)):
#                    if epsilon[i]<epsilon_pulse_arr[ñ]:
#                        suport = funct_f(temps_pt, X0s[ñ], sigmes1[ñ], sigmes2[ñ], As[ñ], Cs[ñ])
#                        break
#                auxiliar.append(suport)
#            auxiliar2.append(np.array(auxiliar))
#        auxiliar3.append(np.array(auxiliar2))
#    Funct.append(np.array(auxiliar3))
#Funct = np.array(Funct)


#L'angle d'interacció entre positrons i fotons amb les dimensions que vull
Theta2D = add_dimension_R(theta_arr, epsilon_mean)*u.rad
Theta3D = add_dimension_R(Theta2D, E_mean)*u.rad
Theta4D = add_dimension_R(Theta3D, phase)*u.rad

#L'espectre i la x-sec amb les dimensions que vull (afegeixo la dimensió de la fase)
xsec4D = add_dimension_R(cross_section_exact, phase)*cross_section_exact.unit
spectra4D = add_dimension_R(spectra, phase)*spectra.unit

#Primer producte
first = Funct*(1-np.cos(Theta4D))*xsec4D*spectra4D

print ('first: ', first)

# Validity mask: shape (n_Efotof, n_Efotoi, n_R)
valid = (
    (E_mean_3d >= E_log_min[None, :, :]) &
    (E_mean_3d <= E_log_max[None, :, :])
)

print ('Valid: ', valid)

weights = Delta_epsilon_log * epsilon_mean     # epsilon_mean is the Jacobian of the integration in terms of log(epsilon)

# Reshape weights for broadcasting over (phase, E_mean, epsilon_mean, R)
weights_4d = weights[None, None, :, None]

# Broadcast valid mask to include phase dimension
valid_4d = valid  #[None, :, :, :]

# Masked weighted sum over epsilon axis (axis=2)
#first_int = np.sum(first * weights_4d * valid_4d, axis=2)

# Use np.where to avoid nan * 0 = nan issue
masked = np.where(valid, first * weights_4d, 0.0)
first_int = np.sum(masked, axis=2)

# With units and without units, now I do not need
#to multiply by the units as this conserves the units 
#during the whole proces
#first_int = np.squeeze(first_int, axis=0)
first_int = first_int
first_intu = first_int.value

print ('first_intu: ', first_intu)


#Calculo la primera integral
#first_int = []
#for l in range(len(phase)):
#    print(l)
#    auxiliar3 = []
#    for j in range(len(E_mean)):
#        auxiliar2 = []
#        for i in range(len(R)):
#            auxiliar = 0
#            for k in range(len(epsilon_mean)):
#                if (E_log_max[k][i] > E_mean_3d[j][k][i] > E_log_min[k][i]):
#                    auxiliar += first[l][j][k][i]*Delta_log[k]*epsilon_mean[k]
#            auxiliar2.append(np.array(auxiliar))
#        auxiliar3.append(np.array(auxiliar2))
#    first_int.append(np.array(auxiliar3))
        
#Amb unitats i sense, per segons em convingui
#first_int = np.array(first_int)*first.unit*epsilon_mean.unit
#first_intu = first_int.value

folder_name = f"Data_Alpha_{alpha}_Rf_{int(Rf)}"

if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")

#first_intu = np.squeeze(first_intu, axis=0)
    
for i in range(len(phase)):
    file_name = "First_int_" + str(i)
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "w") as file:
        for j in range(len(E_mean)):
            
            line = ' '.join(map(str, first_intu[i][j]))
            file.write(line + '\n')
    file.close()



dNdV = Lsd/(4*np.pi*c**3*gamma_w*m_e)

sigma = 0.015

theta0 = phase_r(R_3d*RLC, freq_r)

#rho = anisotropy(phase_3d, R_3d*RLC, sigma, theta0 )


rho = 1

#torno a integrar, tranposo i poso les unitats que toquen
second_int = np.sum(first_int*RLC**2/(R_3d*RLC)**2*rho, axis = 2)*delta_R
secondi_tr = np.transpose(second_int)
secitr_units = secondi_tr*dNdV*e2/(kevs_m**2*RLC)

#This has the same units as the fit I previusly did
#to the sed, and thus, this would be keV/(cm^2 s)
#which we want to obtain

"""
for j in range(len(secitr_units)):
    name =  "Second_integral_" + str(j)
    with open(name, "w") as file:
        for i in range(len(phase)):
            file.write(str(secitr_units[j][i]/secitr_units[j][i].unit))
            file.write("\n")
        file.close()
"""

#Això em servirà per comparar els SEDs
"""
x = [100000000, 345511.48368313984, 492388.6470441612, 766683.7914064317, 1045274.7822721634, 1778278.058325248, 3025302.9216365786, 5146809.142031739, 11420677.265467953, 14896233.027498951, 20309180.95905017, 22189798.33029695, 26489698.039047025, 28942627.400982425, 30253059.877816208, 33054474.176812675, 34551078.33323037, 34551078.33323037, 34551078.33323037, 34551078.33323037, 34551078.33323037, 33054474.176812675, 253423.29584824733, 162756.47818676993, 119377.94337766837, 73347.36026464758, 51468.19574620963]
y = [-0.000006614946304329913, 6.409472670790406e-7, 0.0000014296313509278393, 0.000002218315434776638, 0.000002849262701855678, 0.0000036379467857044774, 0.000004268894052783516, 0.000004794685046597926, 0.000004794685046597926, 0.000004163737779518886, 0.0000032173168789003275, 0.0000022708959782817682, 0.0000012718945341580796, 3.780541770446496e-7, -5.157909938144078e-7, -0.000001462211894432968, -0.000002356052251546398, -0.0000030921557918900665, -0.000004038576692508627, -0.000004827260776357425, -0.000005668525403711354, -0.0000063520532142955235, -2.0031736027488833e-7, -9.890014441236888e-7, -0.0000017251049844673573, -0.0000026189453415807873, -0.000003670527329209606]
"""

#Radiació de curvatura
x2 = [91.38064031403975, 143.41124711707917, 325.42897873361665, 835.0421424203932, 2973.7908453421987, 7630.668566358315, 12476.298173503372, 18039.65018323842, 24031.663174227924, 32013.850754225157, 44431.17228141686, 56813.21278921525, 66930.28098621606, 69729.53796180403, 85582.35356986578, 100822.89271839082, 118777.47306640366, 134312.01677685397]
y2 = [0.000007545003984212426, 0.00001573289719840635, 0.00003530793742217019, 0.00008528077408914708, 0.00012775659273060922, 0.00007362436798086763, 0.00003530793742217019, 0.00001573289719840635, 0.000006513732499309725, 0.0000022442107754792045, 8.02149475022789e-7, 2.3859397545156373e-7, 9.521881042259686e-8, 3.800021287685452e-8, 8.424246985134803e-9, 2.5995202050740402e-9, 8.633158520511537e-10, 2.5678750360758687e-10]

#Dades experimentals
x = [108.57105939203441, 218.41711189283046, 439.3975286702991, 610.5401639918168, 1228.2501438265476, 2371.3741520050517, 3433.3209414598523, 9210.557634784978, 12798.061370079453, 22758.50332098324, 34333.31281068948, 71968.81499932126, 78137.24604821566, 117877.28186263177, 209618.50651936233, 329501.72436413093, 477058.91099048086, 636169.1203944023, 1085715.4984275685, 2096183.4870063816, 2912645.029041295]
y = [0.0002137189267770382, 0.00022159005881784106, 0.00022159005881784106, 0.00022159005881784106, 0.00018493367035466994, 0.00015434114063286706, 0.00011146035613731978, 0.00005215277739701573, 0.000024402507622277154, 0.000012274542945270864, 0.00001319529752225786, 0.000009190742948669466, 0.0000016196597796793656, 0.0000012127458765828707, 5.091038668057519e-7, 2.3821187691854446e-7, 2.061273863385585e-7, 1.5434114063286723e-7, 1.3847178134144655e-7, 1.335531054735925e-7, 1.8493336418610452e-7]

ns = np.array(y)
xns2 = np.array(x)

cr = np.array(x2)
cry = np.array(y2)

spec = np.sum(secitr_units, axis = 1)*delta_phase

SED = (spec*E_mean**2).to('MeV')




#Plotejo la SED
plt.clf()
plt.plot(E_mean.to('MeV') , SED , label = "Theoretical SED")
plt.plot(E_mean.to('MeV') , SED*0.015 , label = r"Theoretical SED, $\eta$")
plt.plot(xns2, ns,'.', label = " Data of the SED")
plt.plot(cr, cry, label ="Model of CR")
plt.ylim((1e-8, 1e0))
plt.xlim((1e1, 1e8))
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$E_{\gamma}^2 \dfrac{dN^{(0)}}{dSdtd\epsilon}$ (MeVcm$^{-2}$s$^{-1}$)", fontsize = 20)
plt.xlabel(r"$E_{\gamma}$(MeV)", fontsize = 20)
plt.title(
    rf"SED with $\epsilon_{{max}} = 10^{{{log_epsilon_max}}}$ and $E_{{\gamma,max}} = 10^{{{log_E_max}}}$",
    fontsize=20)
plt.legend(fontsize = 20)
plt.savefig("SED_theoretical.png")
if plt.isinteractive():
    plt.show()
#Ara ja he comparat els SEDs

#Això servira per la normalitació
phase_averaged = np.sum(secitr_units, axis = 1)*delta_phase

#Torno a plotejar, pero coses diferents, ara plotjo els Pulse profiles en funcio de l'energia
maxs = []
max_value = []
Es = []

adjusts = []
inc_adjusts = []

r_mesh, phase_mesh = np.meshgrid(R_arr/RLC, phase)

#If I put rho = 1, then this rho2 = 1 also.
rho2 = 1
#rho2 = np.transpose(rho, axes = [1,0,2])

plt.clf()
for i in range(len(secitr_units)):
    
    #plt.contourf(r_mesh, phase_mesh, rho2[i])
    #plt.show()
    
    x_initial = 0.3
    gamma_initial1 = 0.3
    gamma_initial2 = 0.1
    A_initial = 1
    C_initial = 1
    
    popt, pcov = curve_fit(asym_lorentz_C, phase, secitr_units[i], [x_initial, gamma_initial1, gamma_initial2, A_initial, C_initial], maxfev = 800000)
    
    adjusts.append(popt)
    inc_adjusts.append(pcov)
    
    #plt.plot(phase, secitr_units[i], label = "excate"+str(i))
    plt.plot(phase, secitr_units[i], label = "_nolegend_")
    
    max_value.append(np.max(asym_lorentz_C(phase,  *popt)))
    
    plt.plot(phase, asym_lorentz_C(phase, *popt))
    
    ids = pd.Series(secitr_units[i]).idxmax()
    maxs.append(phase[ids])
    Es.append(E_mean[i]/(E_mean[i].unit*1e6))#En GeV
    plt.axvline(phase[ids])
    print(phase[ids], E_mean[i].to('GeV'),i)

    if (i+1)%4 == 0:
        plt.xlabel("Phase(rad)", fontsize = 25)
        plt.ylabel(r"$\frac{dN_{\gamma}}{dE_{\gamma}dSdt}$", fontsize = 25)
        plt.legend()
        if plt.isinteractive():
            plt.show()
plt.xlabel("Phase(rad)", fontsize = 25)
plt.ylabel(r"$\frac{dN_{\gamma}}{dE_{\gamma}dSdt}$", fontsize = 25)
plt.legend()
if plt.isinteractive():
    plt.show()    

#Ho torno a transposar per conveniencia
seci_units = np.transpose(secitr_units)

#Calculo el flux per comparar-ho desprès
flux = []
for j in range(len(phase)):
    aux = 0
    for i in range(len(E_mean)):
        aux += seci_units[j][i]*Delta_E[i]
    flux.append(aux/aux.unit)
flux = np.array(flux)*aux.unit

phase_left = []
phase_right = []

for j in range(len(Es)):
    out = adjusts[j]
    maxim = max_value[j]
    left = out[0]-np.sqrt(out[-2]*out[1]/(np.pi*(maxim/2-out[-1]))-out[1]**2)
    right = out[0]+np.sqrt(2*out[2]**2*(out[-2]+np.pi*out[1]*out[-1])/(np.pi*out[1]*maxim)-out[2]**2)
    phase_left.append(left)
    phase_right.append(right)
    
    

#Grafico la posiscio dels peaks per comparar amb resultats
plt.plot(maxs, Es, 'x', color = "k")
plt.plot(phase_left, Es, '.', color = "g")
plt.plot(phase_right, Es, '.', color = "g")
plt.ylabel(r'$E_{\gamma}$ (GeV)', fontsize = 20)
plt.xlabel('Phase', fontsize = 20)
plt.yscale("log")
plt.savefig("Theoretical_Displacement_peak")
if plt.isinteractive():
    plt.show()

#A partir d'aqui ho faig per veure quina és l'energia inicial més baixa dels fotons que em permet
#tenir una energia finals dels fotons de 1 GeV
E_i_0 = 0.02*1e-3*u.keV

print(E_i_0*1000/E_i_0.unit, "eV")

def angle_max(tf, G, B, ti, Ei):
    eq = m*G*B*np.sin(tf)+Ei*np.sin(tf+ti)
    return eq

def second_der(tf, G, B, ti, Ei):
    eq = -m*G*B*np.cos(tf)-Ei*np.cos(tf+ti)
    return eq

#Aquí trobo quin es l'angle maxim
tmax = fsolve(lambda x: angle_max(x*u.rad, Gamma_arr, beta_arr, theta_arr, E_i_0), -theta_arr, xtol = 1e-8, maxfev = 3000)

tmax_com = second_der(tmax*u.rad, Gamma_arr, beta_arr, theta_arr, E_i_0)

#Grafico l'energia final dels fotons per aquest angle maxim
E_log_max2 = E_i_0*m*Gamma_arr*(1-beta_arr*np.cos(theta_arr))/(m*Gamma_arr*(1-beta_arr*np.cos(tmax*u.rad)) + E_i_0*(1-np.cos(theta_arr+tmax*u.rad)))

#plt.plot(R[R <= 2*RLC]/RLC, E_log_max2[R <= 2*RLC])
plt.clf()
plt.plot(R_arr/RLC, E_log_max2)
plt.xlabel(r"R/R$_{L}$", fontsize = 20)
plt.ylabel(r"$E_{\gamma}^{max}$(keV)", fontsize = 20)
plt.savefig("Energia_max_0_02eV")
if plt.isinteractive():
    plt.show()
   

#Resultats dels valors experimentals
A = 6.54617e+00
x0 = 4.05391e-01
sigma_2 = 5.86374e-03
sigma_1 = (2.11878e+00 + 1)*sigma_2
offset =   1.75916e+03 

#Aquí simplement garfico els resultats i els comparo amb els experimentals
result = as_lorentz(phase, x0, sigma_1, sigma_2, 1)

norm = np.sum(flux)*delta_phase

norm2 = np.sum(result)*delta_phase

max_f_1 = np.max(flux)
max_f_2 = np.max(result)

plt.clf()
plt.plot(phase, flux/norm, label = "theoretical result")
plt.plot(phase, result/norm2, label = "experimental result")
plt.xlabel("phase", fontsize = 20)
plt.ylabel("Normalized flux", fontsize = 20)
plt.legend(fontsize = 20)
plt.savefig("Flux_obtained")
if plt.isinteractive():
    plt.show()


#Dades Aharonian
x1 = [0.14364259957498768, 0.1876288679254484, 0.2206185534575859, 0.24536081760668915, 0.2632302096289865, 0.27835056606165837, 0.2934707966486671, 0.30034360585264125, 0.31821306079777023, 0.32920958069326167, 0.33470790356383906, 0.3429553249468734, 0.36082477989200235, 0.36769758909597655, 0.38006872117052815, 0.3841924318620454, 0.3910652410660197, 0.3910652410660197, 0.3979381761156571, 0.40756008383208847, 0.4158075052151229, 0.44604809223480346, 0.4817868762793981, 0.42955324946873463]
y1 = [0.10421694023096655, 0.12409639949634975, 0.15843379773357388, 0.20542168575026074, 0.2487951717523466, 0.2993976272403529, 0.34457839697809856, 0.39879525448070585, 0.5036144847429602, 0.5596385432528681, 0.6084337150052147, 0.6698795419937424, 0.6481927989926995, 0.5975903435046932, 0.4981927989926995, 0.43674705473253095, 0.38975900125912566, 0.33734942749217806, 0.28493985372523045, 0.2325301145015644, 0.17108445296975508, 0.07530122774400316, 0.03915671122783776, 0.12048199748174875]

x2 = [0.16151205452011663, 0.21237113207455152, 0.252233689733495, 0.28109966457411545, 0.30859102723567566, 0.32783509435986474, 0.34707903563839065, 0.35945016771294214, 0.36494849058351947, 0.37869410899146805, 0.3814433333495883, 0.3883161425535626, 0.40618559749869154, 0.4089346960111486, 0.4116837945236057, 0.4144330188817259, 0.417182117394183, 0.41993121590664007, 0.42542953877721745, 0.42955324946873463, 0.43780067085176905, 0.44467348005574336, 0.46116832282181214, 0.47766316558788097]
y2 = [0.08975916671584408, 0.09879525448070582, 0.13674705473253096, 0.1783132569989571, 0.23975908398748486, 0.30843371500521466, 0.3753012277440032, 0.4457831424973927, 0.5072289694859204, 0.5596385432528681, 0.6174698854984357, 0.6734939440083435, 0.6463855152570398, 0.57771088423931, 0.5144578562434817, 0.46746988549843566, 0.39337356873044516, 0.33373502547757705, 0.29578322522575196, 0.24337348600208586, 0.2090362532215802, 0.14939770996871216, 0.10783134224556755, 0.06807242371480116]

x3 = [0.12852236898797897, 0.1560137316495393, 0.1917525786169656, 0.2206185534575859, 0.24261165617140049, 0.25773201260407236, 0.27972505239505535, 0.30034360585264125, 0.314089350106253, 0.3429553249468734, 0.35670106920048517, 0.3718212997874938, 0.3828178196829852, 0.3979381761156571, 0.4006872746281142, 0.4048109853196314, 0.40756008383208847, 0.40756008383208847, 0.4130584067026658, 0.4158075052151229, 0.41993121590664007, 0.4213058280857002, 0.42680415095627755, 0.43505157233931196, 0.4419243815432863, 0.45704461213029485, 0.48453610063751834]
y3 = [0.12409639949634975, 0.1439760242184514, 0.17289157124869636, 0.2108433715005215, 0.25421685750260736, 0.2939759414900922, 0.34457839697809856, 0.41144574426016856, 0.4873494274921781, 0.5849397709968713, 0.6554216857502608, 0.7277108015109508, 0.8180722582580828, 0.7710842875130366, 0.7132530279958282, 0.6391566284994785, 0.5939759414900923, 0.5415662849947853, 0.48915662849947855, 0.43674705473253095, 0.38795188298018435, 0.33373502547757705, 0.27228919848904926, 0.20180728373565973, 0.1457831424973927, 0.08614459924452463, 0.03915671122783776]

A_1_x = np.array(x1)
A_1_flux = np.array(y1)
max_A1 = np.max(A_1_flux)

A_2_x = np.array(x2)
A_2_flux = np.array(y2)
max_A2 = np.max(A_2_flux)

A_3_x = np.array(x3)
A_3_flux = np.array(y3)
max_A3 = np.max(A_3_flux)

#Això és sense experimentals
plt.plot(phase, flux/max_f_1, color = "c", label = "our model")
plt.plot(A_1_x, A_1_flux/max_A1, "--", color = "k", label = r"$\gamma$-ray $R_w = R_L$")
plt.plot(A_2_x, A_2_flux/max_A2, color = "b", label = r"$\gamma$-ray $R_w = 30R_L$")
plt.plot(A_3_x, A_3_flux/max_A3, color = "r", label = r"$\gamma$-ray anisot. wind")
plt.ylim((0,1.2))
plt.xlim((0.1,0.5))
plt.xlabel("phase", fontsize = 20)
plt.ylabel("Normalized flux", fontsize = 20)
plt.legend(fontsize = 17)
plt.savefig("Flux_theory")
if plt.isinteractive():
    plt.show()

"""
the_f = np.arange(0,2*np.pi,0.01)

g = 300
algo = m*g*(1-np.sqrt(1-1/g**2)*np.cos(the_f*u.rad))+E_i_0*(1-np.cos(the_f*u.rad+theta_arr[0]))

plt.plot(the_f, algo)
"""
