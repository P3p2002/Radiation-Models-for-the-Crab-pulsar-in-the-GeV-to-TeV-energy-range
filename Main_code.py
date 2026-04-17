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
import pandas as pd
import scienceplots
import os

SetUp()

mp.dps =  50  # number of digits for internal calculation

R0 =  0.9  # Radi inicial on els positrons comencen a accelerar
RLI = 10   # Limit d'integració superior
Ri =  0.9  # Radi inicial on els positrons comencen a accelerar (en unitats de RLC)
Rf =  2    # Radi final on els positrons deixen d'accelerar (en unitats de RLC)

delta_R = 0.05          # Step width for the integral (in units of RLC)

R  = np.arange(1., RLI, delta_R)   # array de distancies respecte l'estrella de neutrons, en unitats de RLC
a  = np.arcsin(1/R)                # array d'angles que s'obté a partir de la simplificació per a R>>RLC
w  = 1.-np.cos(a)                  # pesos per a l'eficiencia de la dispersió IC

Eunit = u.MeV

#Si nomes vull graficar una alpha necessito posarla com una llista o array
#alpha = np.array([1,3,10])
alpha = 1
#gamma_w = np.arange(55*10**4, 1*10**6, 0.5*10**4)      # factors gamma final que utilitza el model
gamma_w = 5.5*10**5

#Vigilar, perquè si només vull graficar un conjunt de Ri i Rf necessito posarles com a llistes
#Ri    = np.array([1, 20, 25])*RLC        # radi on es comencen a accelerar els electronsque anirà a les funcions de gamma i del moment 
#Rf    = np.array([30, 50, 70])*RLC         # radi final que diu el mateix model

gamma_0 = 300  # Factor gamma inicials dels positrons

E0 = 1*u.keV

E_fotoi2 = [0.5, 1.3, 3.0, 7.0, 12.0, 27.0, 65.0, 170.0]*(u.keV)#Energies en les quals s'ha dividit els pulse profiles
#Delta_E_fotoi = [0.2, 1.4, 2.0, 6.0, 4.0, 26.0, 50.0, 160.0]*(u.keV)

E_fotoi_uplim = 4
E_fotoi_lowlim = -5
ebins = 30   # was 30
#E_fotoi = np.logspace(-1,2, steps)*(u.keV)
E_fotoi1 = np.logspace(E_fotoi_lowlim,E_fotoi_uplim,ebins)*(u.keV) # Energies inicials dels fotons

logE = np.log(E_fotoi1/E0)

E_fotoi = (E_fotoi1[1:] + E_fotoi1[:-1])/2  #Bins d'energies dels fotons

Delta_E = E_fotoi1[1:] - E_fotoi1[:-1]      #Espaiat d'energies dels fotons

Delta_log = np.log((E_fotoi1[1:])/E0) -np.log((E_fotoi1[:-1])/E0) #Espaiat logaritmic

steps = 30 #Numero de energia final dels fotons que tindrè
#El logspace em genera un conjunt de 10**n sent n valors entre 1 i 6 espaiats
#pels steps que li doni

E_foto_lowlim = 6  # 10^6 keV --> 1 GeV
E_foto_uplim  = 11  # 10^9 keV --> 1 TeV
E_fotof1 = np.logspace(E_foto_lowlim,E_foto_uplim, steps)*(u.keV)      # Energies finals dels fotons

E_fotof = (E_fotof1[1:]+E_fotof1[:-1])/2        # Bins de les energies finals dels fotons

Delta_Ef = E_fotof1[1:]-E_fotof1[:-1]           # Espaiat d'aquesta energia

E_fotoi_3d = add_dim_e_fi(E_fotof, E_fotoi, R)  # Energia inicials dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f

E_fotof_3d = add_dim_e_ff(E_fotof, E_fotoi, R)  # Energia final dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f

Gamma = gammaw(R, Ri, Rf,
               gamma_0, gamma_w, alpha)         # array de factors gamma per a cada distància, segons el model de vent

#print ('R: ', R,'\n')
#print ('Gamma: ', Gamma,'\n')

M_i   = M(R, Ri, Rf,
          gamma_w, alpha)                       # array de moments angulars que s'emporten els electrons

#print ('M: ', M_i,'\n')

Gamma_2d = add_dimension_R(Gamma, E_fotoi)      # Factor gamma dels positrons en 2 dimensions

#print ('Gamma_2d: ', Gamma_2d,'\n')

Gamma_3d = add_dimension_R(Gamma_2d, E_fotof)   # Factor gamma dels positrons en 3 dimensions

#print ('Gamma_3d: ', Gamma_3d,'\n')

beta = beta_f(Gamma_3d, dps=None)               # Valor de la beta dels positrons en 3 dimensions

#print ('beta_3d: ', beta,'\n')

beta1d = beta_f(Gamma, dps=None)                # Valor de la beta dels positrons en 1 dimensio (la de R, que es de l'unic parametre que depen)

#theta_1d = np.arcsin(M_i*c/(Gamma*m*R*RLC))    # Array d'angles de la colisio entre electrons i fotons
#print ('theta_1d: ', theta_1d,'\n')
theta_1d = thetafunct(R,R0,Rf,RLC,gamma_w,Gamma,alpha)*u.rad
#print ('theta_1d: ', theta_1d,'\n')

theta_2d = add_dimension_R(theta_1d, E_fotoi)  # Array d'angles de la colisio de positrons i fotons en 2 dimensions

thetau = add_dimension_R(theta_2d, E_fotof)    # Array d'angles de la colisio de positrons i fotons en 3 dimensions

theta = thetau*u.rad                           # Array en 3 dimensions i amb unitats

theta_init = theta_init(beta, theta, E_fotoi_3d, E_fotof_3d) # Primera approximacio del que val el valor final de l'angle de dispersió del foto

print ('theta_init:', theta_init)

#Poso Gamma_3d[0] ja que no em depen de l'energia final del fotó, i per taant no em canvia el resultat quina triï
E_fotof_max = E_fotoi_3d[0]*m_keV*Gamma_3d[-1]*(1-beta[0]*np.cos(theta[0]))/(m_keV*Gamma_3d[-1]*(1-beta[0]) + E_fotoi_3d[0]*(1-np.cos(theta[0])))#Energia maxima que els fotons poden assolir, primera aproximacio
E_fotof_min = E_fotoi_3d[0]*m_keV*Gamma_3d[0]*(1+beta[0]*np.cos(theta[0]))/(m_keV*Gamma_3d[0]*(1+beta[0]) + E_fotoi_3d[0]*(1+np.cos(theta[0])))#Energia minima que els fotons poden assolir, primera aproximacio

print ('E_f,max: ',E_fotof_max,' E_f,min: ',E_fotof_min)

#A partir d'aqui comença un calcul numeric del valor de l'angle de dispersió dels fotons

#theta_f_e = []
#for j in range(len(E_fotof)):
#    print(j)
#    auxiliar = []
#    for k in range(len(E_fotoi)):
#        auxiliar2 = []
#        for i in range(len(R)):
#            if (E_fotof_max[k][i] > E_fotof_3d[j][k][i] > E_fotof_min[k][i]):
#                theta_fexacte1 = solver(theta_init[j][k][i],theta[j][k][i], Gamma_3d[j][k][i], beta[j][k][i], E_fotof_3d[j][k][i], E_fotoi_3d[j][k][i])
#                theta_fexacte2 = solver(theta_fexacte1,theta[j][k][i], Gamma_3d[j][k][i], beta[j][k][i], E_fotof_3d[j][k][i], E_fotoi_3d[j][k][i])                
#                theta_fexacte3 = solver(theta_fexacte2,theta[j][k][i], Gamma_3d[j][k][i], beta[j][k][i], E_fotof_3d[j][k][i], E_fotoi_3d[j][k][i])                
#                auxiliar2.append(float(theta_fexacte3))
#            else:
#                auxiliar2.append(10000)
#        auxiliar.append(np.array(auxiliar2))
#    theta_f_e.append(np.array(auxiliar))
#theta_f_e = np.array(theta_f_e)*u.rad  #Obtinc un valor 

theta_f_e = compute_theta_f_exact(theta_init, theta, Gamma_3d, beta, E_fotof_3d, E_fotoi_3d, E_fotof_min, E_fotof_max, fill_value=np.nan)

print ('theta_f_e: ',theta_f_e)

comprovacions = equation_solve(theta_f_e, theta, Gamma_3d, beta, E_fotof_3d, E_fotoi_3d, m)

# Broadcast min/max from (n_Ei, n_R) to (n_Ef, n_Ei, n_R)
valid = (
    (E_fotof_3d > E_fotof_min[None, :, :]) &
    (E_fotof_3d < E_fotof_max[None, :, :])
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
#for j in range(len(E_fotof)):
#    for k in range(len(E_fotoi)):
#        for i in range(len(R)):
#            if (E_fotof_max[k][i] > E_fotof_3d[j][k][i] > E_fotof_min[k][i]):
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
cross_section_exact = exact_xsection(theta, theta_f_e, beta, Gamma_3d, E_fotoi_3d, E_fotof_3d)
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
mask = E_fotoi > 0.2 * u.keV

print ('E_fotoi: ', E_fotoi)

print ('mask: ', mask)
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

(Sedfit, xi, poptspectrum) = Xi_Fit(Interval_y, Interval_x, 7, sigma)  

plt.plot(Interval_x, Sedfit, label = "Fitting quadratic")
plt.plot(Interval_x, Interval_y, '.', label = "Interval 4")

plt.ylabel(r"$E^2F (keVcm^{-2}s^{-1})$")
plt.xlabel(r"E (keV)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

spec1d2 = SEDfromFIT(np.log10(E_fotoi/E0), *poptspectrum)
""" I will need to change this with the new polynomial"""
spec_1d = np.empty(len(E_fotoi)) * (1 / E0).unit
spec_1d[mask] = (
    K * (E_fotoi[mask] / E0) ** (-a - 1 - b * np.log10(E_fotoi[mask] / E0)) / (E0)
)
spec_1d[~mask] = (
    K1 * (E_fotoi[~mask] / E0) ** (-a1 - 1 - b1 * np.log10(E_fotoi[~mask] / E0)) / E0
)

# Broadcast to shape (len(E_fotof), len(E_fotoi), len(R))
#I change it to the new fiting
spectra = np.broadcast_to(
    spec1d2.reshape(1, len(E_fotoi), 1),
    (len(E_fotof), len(E_fotoi), len(R))
)*(1/E0).unit
### As this does not have the  right units (which should be MeV/(cm^2 s))
### we expect the final result, i.e., the second integral to have the same units

plt.plot(E_fotoi, spec_1d, '.', label = "Original fit")
plt.plot(E_fotoi, spec1d2, '.', label = "Polynomial fit")
plt.plot(Interval_x, Interval_y, '.', label = "Data")

plt.ylabel(r"$E^2F (keVcm^{-2}s^{-1})$")
plt.xlabel(r"E (keV)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()


print ('spectra: ', spectra)

#spectra = []
#for j in range(len(E_fotof)):
#    auxiliar1 = []
#    for k in range(len(E_fotoi)):
#        auxiliar2 = []
#        for i in range(len(R)):
#            if E_fotoi[k] > 0.2*u.keV:
#                auxiliar = K*(E_fotoi[k]/E0)**(-a-1-b*np.log10(E_fotoi[k]/E0))/E0.value
#            elif E_fotoi[k] < 0.2*u.keV:
#               auxiliar = K1*(E_fotoi[k]/E0)**(-a1-1-b1*np.log10(E_fotoi[k]/E0))/E0.value
#           auxiliar2.append(auxiliar)
#        auxiliar1.append(np.array(auxiliar2))
#    spectra.append(np.array(auxiliar1))
#spectra = np.array(spectra)*(1/E0).unit

#Inicialtzo la variable de la fase
delta_phase = 0.004
phase = np.arange(0.2,0.5,delta_phase)
phase_3d = add_dim_phase(phase, E_fotof, R) #El poso en les dimensions que em conve
time = phase_3d*P


#Canvio les variables segons les dimensions que em conve
R_2d = add_dimension_R(R, E_fotof)
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
def Time_d(time, R, theta):
    return time - R*(1-np.cos(theta))*RLC/c + theta.value/Omega

#A partir d'aquí re faig les dues primeres integrals però considerant que els pulse profiles depenen de l'energia


n_phase = len(phase)
n_ef = len(E_fotof)
n_ei = len(E_fotoi)
n_r = len(R)

# Output without the redundant E_fotof dimension first
base = np.empty((n_phase, n_ei, n_r), dtype=float)

# Precompute which fitted parameter set corresponds to each E_fotoi[i]
# idx[i] = first index n such that E_fotoi[i] < E_fotoi2[n]
idx = np.searchsorted(E_fotoi2, E_fotoi, side="right")

# Clamp in case some E_fotoi are above all E_fotoi2 values
idx = np.clip(idx, 0, len(E_fotoi2) - 1)

for l, ph in enumerate(phase):
    # Time_d depends only on phase[l] and R
    temps = Time_d(ph * P, R, theta_1d)    
    #temps = np.array([Time_d(ph * P, r, th) for r, th in zip(R, theta_1d)])

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

# Broadcast over E_fotof, since the result does not depend on j
Funct = np.broadcast_to(base[:, None, :, :], (n_phase, n_ef, n_ei, n_r)).copy()

print ('Funct: ', Funct)

#Els pulse profiles mes precisos en funcio de l'energia
#Funct = []
#for l in range(len(phase)):
#    print(l)
#    auxiliar3 = []
#    for j in range(len(E_fotof)):
#        auxiliar2 = []
#        for i in range(len(E_fotoi)):
#            auxiliar = []
##            for k in range(len(R)):
#                temps_pt = Time_d(phase[l]*P, R[k], theta_1d[k])
#                for ñ in range(len(E_fotoi2)):
#                    if E_fotoi[i]<E_fotoi2[ñ]:
#                        suport = funct_f(temps_pt, X0s[ñ], sigmes1[ñ], sigmes2[ñ], As[ñ], Cs[ñ])
#                        break
#                auxiliar.append(suport)
#            auxiliar2.append(np.array(auxiliar))
#        auxiliar3.append(np.array(auxiliar2))
#    Funct.append(np.array(auxiliar3))
#Funct = np.array(Funct)


#L'angle d'interacció entre positrons i fotons amb les dimensions que vull
Theta2D = add_dimension_R(theta_1d, E_fotoi)*u.rad
Theta3D = add_dimension_R(Theta2D, E_fotof)*u.rad
Theta4D = add_dimension_R(Theta3D, phase)*u.rad

#L'espectre i la x-sec amb les dimensions que vull (afegeixo la dimensió de la fase)
xsec4D = add_dimension_R(cross_section_exact, phase)*cross_section_exact.unit
spectra4D = add_dimension_R(spectra, phase)*spectra.unit

#Primer producte
first = Funct*(1-np.cos(Theta4D))*xsec4D*spectra4D

print ('first: ', first)

# Validity mask: shape (n_Efotof, n_Efotoi, n_R)
valid = (
    (E_fotof_3d > E_fotof_min[None, :, :]) &
    (E_fotof_3d < E_fotof_max[None, :, :])
)

print ('Valid: ', valid)

# Weight for integration over E_fotoi: shape (n_Efotoi,)
weights = Delta_log * E_fotoi

# Reshape weights for broadcasting over (phase, E_fotof, E_fotoi, R)
weights_4d = weights[None, None, :, None]

# Broadcast valid mask to include phase dimension
valid_4d = valid[None, :, :, :]

# Masked weighted sum over E_fotoi axis (axis=2)
#first_int = np.sum(first * weights_4d * valid_4d, axis=2)

# Use np.where to avoid nan * 0 = nan issue
masked = np.where(valid_4d, first * weights_4d, 0.0)
first_int = np.sum(masked, axis=2)

# With units and without units, now I do not need
#to multiply by the units as this conserves the units 
#during the whole proces
first_int = first_int
first_intu = first_int.value

print ('first_intu: ', first_intu)


#Calculo la primera integral
#first_int = []
#for l in range(len(phase)):
#    print(l)
#    auxiliar3 = []
#    for j in range(len(E_fotof)):
#        auxiliar2 = []
#        for i in range(len(R)):
#            auxiliar = 0
#            for k in range(len(E_fotoi)):
#                if (E_fotof_max[k][i] > E_fotof_3d[j][k][i] > E_fotof_min[k][i]):
#                    auxiliar += first[l][j][k][i]*Delta_log[k]*E_fotoi[k]
#            auxiliar2.append(np.array(auxiliar))
#        auxiliar3.append(np.array(auxiliar2))
#    first_int.append(np.array(auxiliar3))
        
#Amb unitats i sense, per segons em convingui
#first_int = np.array(first_int)*first.unit*E_fotoi.unit
#first_intu = first_int.value

folder_name = f"Data_Alpha_{alpha}_Rf_{int(Rf)}"

if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")
    
for i in range(len(phase)):
    file_name = "First_int_" + str(i)
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "w") as file:
        for j in range(len(E_fotof)):
            
            line = ' '.join(map(str, first_intu[i][j]))
            file.write(line + '\n')
    file.close()



dNdV = Lsd/(4*np.pi*c**3*gamma_w*m_e)

sigma = 0.015

theta0 = phase_r(R_3d*RLC, freq_r)

rho = anisotropy(phase_3d, R_3d*RLC, sigma, theta0 )


#rho = 1

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

SED = (spec*E_fotof**2).to('MeV')




#Plotejo la SED
plt.plot(E_fotof.to('MeV') , SED , label = "Theoretical SED")
plt.plot(E_fotof.to('MeV') , SED*0.015 , label = r"Theoretical SED, $\eta$")
plt.plot(xns2, ns,'.', label = " Data of the SED")
plt.plot(cr, cry, label ="Model of CR")
plt.ylim((1e-10, 1e-2))
plt.xlim((1e1, 1e7))
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$E_{\gamma}^2 \dfrac{dN^{(0)}}{dSdtd\epsilon}$ (MeVcm$^{-2}$s$^{-1}$)", fontsize = 20)
plt.xlabel(r"$E_{\gamma}$(MeV)", fontsize = 20)
plt.title(
    rf"SED with $\epsilon_{{max}} = 10^{{{E_fotoi_uplim}}}$ and $E_{{\gamma,max}} = 10^{{{E_foto_uplim}}}$",
    fontsize=20)
plt.legend(fontsize = 20)
plt.savefig("SED_theoretical")
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

r_mesh, phase_mesh = np.meshgrid(R/RLC, phase)

#If I put rho = 1, then this rho2 = 1 also.
#rho2 = 1
rho2 = np.transpose(rho, axes = [1,0,2])

for i in range(24):
    
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
    
    plt.plot(phase, secitr_units[i], label = "excate"+str(i))
    
    max_value.append(np.max(asym_lorentz_C(phase,  *popt)))
    
    plt.plot(phase, asym_lorentz_C(phase, *popt))
    
    ids = pd.Series(secitr_units[i]).idxmax()
    maxs.append(phase[ids])
    Es.append(E_fotof[i]/(E_fotof[i].unit*1e6))#En GeV
    plt.axvline(phase[ids])
    print(phase[ids], E_fotof[i].to('GeV'),i)

    if (i+1)%4 == 0:
        plt.xlabel("Phase(rad)", fontsize = 25)
        plt.ylabel(r"$\frac{dN_{\gamma}}{dE_{\gamma}dSdt}$", fontsize = 25)
        plt.legend()
        plt.show()
plt.xlabel("Phase(rad)", fontsize = 25)
plt.ylabel(r"$\frac{dN_{\gamma}}{dE_{\gamma}dSdt}$", fontsize = 25)
plt.legend()
plt.show()    

#Ho torno a transposar per conveniencia
seci_units = np.transpose(secitr_units)

#Calculo el flux per comparar-ho desprès
flux = []
for j in range(len(phase)):
    aux = 0
    for i in range(len(E_fotof)):
        aux += seci_units[j][i]*Delta_Ef[i]
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
tmax = fsolve(lambda x: angle_max(x*u.rad, Gamma, beta1d, theta_1d, E_i_0), -theta_1d, xtol = 1e-8, maxfev = 3000)

tmax_com = second_der(tmax*u.rad, Gamma, beta1d, theta_1d, E_i_0)

#Grafico l'energia final dels fotons per aquest angle maxim
E_fotof_max2 = E_i_0*m*Gamma*(1-beta1d*np.cos(theta_1d))/(m*Gamma*(1-beta1d*np.cos(tmax*u.rad)) + E_i_0*(1-np.cos(theta_1d+tmax*u.rad)))

#plt.plot(R[R <= 2*RLC]/RLC, E_fotof_max2[R <= 2*RLC])
plt.plot(R/RLC, E_fotof_max2)
plt.xlabel(r"R/R$_{L}$", fontsize = 20)
plt.ylabel(r"$E_{\gamma}^{max}$(keV)", fontsize = 20)
plt.savefig("Energia_max_0_02eV")
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

plt.plot(phase, flux/norm, label = "theoretical result")
plt.plot(phase, result/norm2, label = "experimental result")
plt.xlabel("phase", fontsize = 20)
plt.ylabel("Normalized flux", fontsize = 20)
plt.legend(fontsize = 20)
plt.savefig("Flux_obtained")
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
plt.show()

"""
the_f = np.arange(0,2*np.pi,0.01)

g = 300
algo = m*g*(1-np.sqrt(1-1/g**2)*np.cos(the_f*u.rad))+E_i_0*(1-np.cos(the_f*u.rad+theta_1d[0]))

plt.plot(the_f, algo)
"""
