# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:03:08 2024

@author: Pep Rubi
"""

import numpy as np
import math as m
from Model_delta_t_E import *
from calculation_angles import *
from Change_dimensions import *
from xsection_jacobians_stuff import *
from f_function import *
from Anisotropy import *
import astropy.units as u
from astropy.constants import c
from astropy.constants import m_e
from astropy.constants import hbar
from astropy.constants import e
from astropy.constants import eps0
from astropy.constants import h
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import os
from setup import SetUp
from constants import *


SetUp()

plt.style.use(["science","no-latex"])
plt.rcParams["figure.figsize"] = (7,7)
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5

SetUp()

Rf =  2    # Final radius up to which the positrons are getting accelerated (in units of RLC)
alpha = 1         # power-law evolution index of Gammas along the current sheet WE WANT TO VARY BETWEEN 0.5 and 10

folder_name = f"Data_Alpha_{alpha}_Rf_{int(Rf)}"
if os.path.exists(folder_name):
    print("The data has already been computed")
else:
    print("The data has to be computed")#An error wil pop up
    
Data_file_name = "Data_file"
Data_path = os.path.join(folder_name, Data_file_name)


data = {}

with open(Data_path, "r") as file:
    lines = file.readlines()

# First line
parts = lines[0].strip().split('\t')
R0 = float(parts[0].split(':')[1])
RLI = float(parts[1].split(':')[1])
delta_R = float(parts[2].split(':')[1])

# Second line
parts = lines[1].strip().split('\t')
log_epsilon_max = float(parts[0].split(':')[1])
log_epsilon_min = float(parts[1].split(':')[1])
log_epsilon_bins = int(parts[2].split(':')[1])

# Third line
parts = lines[2].strip().split('\t')
log_E_max = float(parts[0].split(':')[1])
log_E_min = float(parts[1].split(':')[1])
log_E_bins = int(parts[2].split(':')[1])


debug = False
parallalize = True
n_jobs = 4

mp.mp.dps =  50  # number of digits for internal calculation

Eunit = u.keV     # energy unit to which the results are getting referred to

R_arr  = np.arange(R0, RLI, delta_R)  # array of distances w.r.t . the NS, in units of RLC

gamma_w = 6*10**7 # corresponds to Gamma_w * m_e * c^2 = 30 TeV, the maximum possible with HESS Vela data
gamma_0 = 300     # Initial gamma factor of the  positrons when the enter the current sheet
E0 = 1*Eunit      # pivot energy, 1 keV

epsilon_pulse_arr = [0.5, 1.3, 3.0, 7.0, 12.0, 27.0, 65.0, 170.0]*(u.keV) # Energies for the pulse profiles

epsilon_arr = np.logspace(log_epsilon_min,log_epsilon_max,log_epsilon_bins)*(Eunit) # Array of initial photon energies, logarithmically spaced, BUT LINEAR

epsilon_mean = np.sqrt(epsilon_arr[1:]*epsilon_arr[:-1]) # Mean energies of incident photons, logarithmically spaced, BUT LINEAR!!
Delta_epsilon = epsilon_arr[1:] - epsilon_arr[:-1]       # Energy spacing of the incident photons, logarithmically spaced, BUT LINEAR!!
Delta_epsilon_log = np.log((epsilon_arr[1:])/E0) - np.log((epsilon_arr[:-1])/E0) # Spacing of natural logarithm of incident photon energies

E_arr       = np.logspace(log_E_min,log_E_max, log_E_bins)*(Eunit) # Array of scattered photon energies, logarithmically spaced, BUT LINEAR!!

E_mean     = np.sqrt(E_arr[1:]*E_arr[:-1])   # Mean energies of scattered photons, logarithmically spaced, BUT LINEAR!!
Delta_E    = E_arr[1:]-E_arr[:-1]            # Energy spacing of scattered photons, logarithmically spaced, BUT LINEAR!!
Delta_E_log = np.log((E_arr[1:])/E0) - np.log((E_arr[:-1])/E0) # Spacing of natural logarithm of scattered photon energies

epsilon_mean_3d = add_dim_e_fi(E_mean, epsilon_mean, R_arr)   # Energia inicials dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f
E_mean_3d       = add_dim_e_ff(E_mean, epsilon_mean, R_arr)   # Energia final dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f

Gamma_arr = gammaw(R_arr, R0, Rf, gamma_0, gamma_w, alpha)  # array of gamma factors for each distance, following the wind model
M_arr     = M(R_arr, R0, Rf, gamma_w, alpha)                # array of angular moments que s'emporten els electrons
Gamma_2d  = add_dimension_R(Gamma_arr, epsilon_mean)        # New array dimension: len(epsilon_mean), len(Gamma_arr)
Gamma_3d  = add_dimension_R(Gamma_2d, E_mean)               # New array dimension: len(E_mean), len(epsilon_mean), len(Gamma_arr)
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

theta_init = theta_init_funct(beta_3d, theta_3d, epsilon_mean_3d, E_mean_3d) # Primera approximacio del que val el valor final de l'angle de dispersió del foto

if debug:
    print ('theta_arr: ', theta_arr,'\n')
    print ('theta_init:', theta_init)

plt.figure()
plt.plot(R_arr, np.rad2deg(theta_arr), label = r"$\theta_{L} (R)$")
plt.ylabel(r"$\theta_{L}$ (deg)")
plt.xlabel(r"$R (R_{LC})$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig('theta_L.png')
if plt.isinteractive():
    plt.show()


if debug:
    ind1 = -1
    angleplot = np.linspace(0, 2*np.pi, 2000)*u.rad
    derfinal = derEfotof(epsilon_mean_3d[ind1][ind1][ind1], Gamma_3d[ind1][ind1][ind1], beta_3d[ind1][ind1][ind1], theta_3d[ind1][ind1][ind1], angleplot, m_keV)
    Ef = Eout_from_Ein_theta_thetaL(epsilon_mean_3d[ind1][ind1][ind1], Gamma_3d[ind1][ind1][ind1], beta_3d[ind1][ind1][ind1], theta_3d[ind1][ind1][ind1], angleplot, m_keV)
    
    '''   TESTS
    gamma_test = Gamma_3d[13,0,0]
    beta_test = beta_f(gamma_test, dps= None)
    E_in_test = epsilon_mean_3d[13,0,0]
    theta_test = theta_3d[13,0,0]
    Eout_test = E_mean_3d[13,0,0]
    Etest = Eout_test - Eout_from_Ein_theta_thetaL(E_in_test,gamma_test, beta_test, theta_test, angleplot ,m_keV)
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


# Solutions of Eq. 3.22
theta_fs = np.arctan(-epsilon_mean_3d*np.sin(theta_3d)/(epsilon_mean_3d*np.cos(theta_3d) + m_keV*(Gamma_3d**2 -1)))
theta_ss = theta_fs + np.pi*u.rad

if debug:
    print ('theta_fs:', theta_fs)

E_log_max = Eout_from_Ein_theta_thetaL(epsilon_mean_3d, Gamma_3d, beta_3d, theta_3d, theta_fs, m_keV)
E_log_min = Eout_from_Ein_theta_thetaL(epsilon_mean_3d, Gamma_3d, beta_3d, theta_3d, theta_ss, m_keV)



#Inicialtzo la variable de la fase
delta_phase = 0.004
phase = np.arange(0.2,0.5,delta_phase)
phase_3d = add_dim_phase(phase, E_mean, R_arr) #El poso en les dimensions que em conve
time = phase_3d*P


#Canvio les variables segons les dimensions que em conve
R_2d = add_dimension_R(R_arr, E_mean)
R_3d = add_dimension_R(R_2d, phase)

#Això són diferents coses que s'hauràn d'afegir per la segona integral
Lsd = 4.6e31*u.J/u.s

dNdV = Lsd/(4*np.pi*c**3*gamma_w*m_e)

sigma = 0.015

freq_r = 0.1*u.rad/(u.m)

theta0 = phase_r(R_3d*RLC, freq_r)

rho = anisotropy(phase_3d, R_3d*RLC, sigma, theta0 )

#rho = 1


first_int = []
for i in range(len(phase)):
    file_name = "First_int_" + str(i)
    file_path = os.path.join(folder_name, file_name)
    aux = []
    with open(file_path, "r") as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            row = np.array(row)
            aux.append(row)
        aux = np.array(aux)
    first_int.append(aux)
first_int = np.array(first_int)*(1/u.keV)**3


#torno a integrar, tranposo i poso les unitats que toquen
second_int = np.sum(first_int*RLC**2/(R_3d*RLC)**2*rho, axis = 2)*delta_R
secondi_tr = np.transpose(second_int)
secitr_units = secondi_tr*dNdV*e2/(kevs_m**2*RLC)

"""   
#Això es per si directament vull llegir la segona integral en comptes de la 
primera 
secitr_units = []         
for j in range(len(E_fotof)):
    name =  "Second_integral_" + str(j)
    with open(name, "r") as file:
        auxiliar = []
        for line in file:
            auxiliar.append(float(line))
        secitr_units.append(np.array(auxiliar))

secitr_units = np.array(secitr_units)/u.keV
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
plt.ylim((1e-8, 1e-1))
plt.xlim((1e1, 1e8))
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$E_{\gamma}^2 \dfrac{dN^{(0)}}{dSdtd\epsilon}$ (MeVcm$^{-2}$s$^{-1}$)", fontsize = 20)
plt.xlabel(r"$E_{\gamma}$(MeV)", fontsize = 20)
plt.title(
    rf"SED with $\epsilon_{{max}} = 10^{{{log_epsilon_max}}}$, $E_{{\gamma,max}} = 10^{{{log_E_max}}}$ and $\alpha = {{{alpha}}}$",
    fontsize=20)
plt.legend(fontsize = 20)
plt.savefig("SED_theoretical.png")
if plt.isinteractive():
    plt.show()
#Ara ja he comparat els SEDs

#M'ho imprimeixo per mi
#print(Ri/Rl, Rf/Rl, R[0]/Rl, R[-1]/Rl)

#Això servira per la normalitació
phase_averaged = np.sum(secitr_units, axis = 1)*delta_phase

#Torno a plotejar, pero coses diferents, ara plotjo els Pulse profiles en funcio de l'energia
maxs = []
max_value = []
Es = []

adjusts = []
inc_adjusts = []

r_mesh, phase_mesh = np.meshgrid(R_arr/RLC, phase)

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
E_i_0 = 0.04*1e-3*u.keV

print(E_i_0*1000/E_i_0.unit, "eV")


def angle_max(tf, G, B, ti, Ei):
    eq = m_keV*G*B*np.sin(tf)+Ei*np.sin(tf+ti)
    return eq

def second_der(tf, G, B, ti, Ei):
    eq = -m_keV*G*B*np.cos(tf)-Ei*np.cos(tf+ti)
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


delta_phase2 = 0.001
phase2 = np.arange(0.2,0.5,delta_phase2)

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
algo = m*g*(1-np.sqrt(1-1/g**2)*np.cos(the_f*u.rad))+E_i_0*(1-np.cos(the_f*u.rad+theta_1d[0]))

plt.plot(the_f, algo)
"""