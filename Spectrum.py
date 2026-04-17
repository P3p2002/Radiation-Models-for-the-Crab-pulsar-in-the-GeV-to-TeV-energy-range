# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:34:10 2024

@author: Pep Rubi

popt4, pcov4 = curve_fit(EF, E4, E1F, [K_in, alpha_in, beta_in])
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots
from sklearn.metrics import r2_score
from scipy.stats import chisquare

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

def EsquaredF(E, K, alpha, beta):
    E0 = 1
    #F = K*(E)**(-alpha+2-beta*np.log10(E/E0))/(E0**(-alpha-beta*np.log10(E/E0)))
    F = (E0)*K*(E/E0)**(-alpha+1-beta*np.log10(E/E0))
    return F


"""New data from new intervals"
See the Images in Phase-averaged data points to see which are the points I picked
"""

Interval_x = np.array([0.0003162276172147746, 0.0006628700711058477, 0.001279801052582922, 0.002470912307632171, 0.004970822524341731, 0.010419739772213055, 0.02011739496035263, 0.035774307157273, 0.07498939446834635, 0.15719128418077574, 0.3162285100282012, 0.6105401886168376, 1.1312834556437819, 1.8529221773372084, 4.970827202420301, 12.798058703114059, 20.117413892984658, 32.95018783048523, 66.28706949378679, 127.98046658775885, 227.58476089298526, 388.4066097985313, 749.8967676108324, 1279.803461444897, 2096.1821765480395, 3727.6072514469956, 6628.719426036748])
Interval_y = np.array([0.000719370045023045, 0.0008129133633486966, 0.0008674801800921799, 0.0009142518392550057, 0.0009532283408371741, 0.0009844093279992444, 0.0009922048424193426, 0.0009844093279992444, 0.0009688190128379296, 0.0009532283408371741, 0.0009220473536751038, 0.0008908660096735928, 0.0008362991929301096, 0.0007817323761866263, 0.0007271652026037024, 0.0007583465466052134, 0.0006024408971159803, 0.0005790550675345674, 0.0006336218842780508, 0.0006725983858602192, 0.0006803935434408766, 0.0006881890578609747, 0.0006881890578609747, 0.0006570077138594637, 0.0006258267266973932, 0.000540078565952399, 0.0005088975787903287])

Interval_x = Interval_x*1e3
Interval_y = Interval_y*1e3

Intervallog_x = np.log10(Interval_x)
Intervallog_y = np.log10(Interval_y)

Inc_x = np.array([0.0003162276172147746, 0.0003162276172147746, 0.8141694802703412, 0.7498960618779956, 6.361675872880828, 9.59717769319248, 8.83954254202545, 6.628713187704805, 12.282489672705855, 12.282489672705855, 22.758497507472, 20.117413892984658, 35.774340824743255, 35.774340824743255, 2912.6431298426255, 2682.698680657325, 848.3458066388913, 562.3425159442847, 61.05413377853293, 61.05413377853293, 6361.681859901954, 6105.4248695598335])
Inc_y = np.array([0.0006258267266973932, 0.0007661417041858708, 0.0007895275337672837, 0.0008596850225115225, 0.0006414173986981487, 0.0005790550675345674, 0.0007505510321851153, 0.0007973226913479412, 0.000719370045023045, 0.0008207085209293541, 0.0007037793730222895, 0.0005166927363709861, 0.0006725983858602192, 0.00047771659162825837, 0.0006725983858602192, 0.0007661417041858708, 0.000719370045023045, 0.0006414173986981487, 0.0005634643955338119, 0.0006881890578609747, 0.00029842511255761243, 0.0005946457395353229])

Inc_x = Inc_x*1e3
Inc_y = Inc_y*1e3

sigma2 = 0
for i in range(int(len(Inc_y)/2)):
    a = (Inc_y[i] - Inc_y[i+1])/2
    sigma2 += a**2
sigma = np.sqrt(sigma2)

    
#Parametres inicials
K_in = 1
E0_in = 0.1
alpha_in = 1
beta_in = 0.5


a0in = np.log10(E0_in/K_in)
a1in = 1 - alpha_in
a2in = beta_in
a3in = beta_in*alpha_in
a4in = beta_in
a5in = beta_in
a6in = beta_in
a7in = beta_in
ains = np.array([a0in, a1in, a2in, a3in, a4in, 
                 a5in, a6in, a7in])


def Xi2(experimental, theoretical, sigmaxi, dof):
    a = (experimental - theoretical)**2
    xi2 = np.sum(a)/sigmaxi**2
    xi2 = xi2/(len(experimental) - dof)
    return xi2

def SEDfromFIT(LogE, *a):
    return 10**(polN(LogE, *a))

def polN(x, *a):
    SedLog = sum(a[k] * x**k for k in range(len(a)))
    return SedLog

def Xi_Fit(experimentaly, experimentalx, N, Sigma):
    ain = beta_in*np.ones(N)
    poptfit, pcovfit = curve_fit(polN, np.log10(experimentalx), np.log10(experimentaly), ain)
    SEDFit = SEDfromFIT(np.log10(experimentalx), *poptfit)
    xi = Xi2(experimentaly, SEDFit, Sigma, N)
    
    plt.plot(experimentalx, SEDFit)
    plt.plot(experimentalx, experimentaly, '.')
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return (SEDFit, xi, poptfit)

xilist = []
Nlist = []
for j in range(1, 15):  
    (Sed, xi, popt) = Xi_Fit(Interval_y, Interval_x, j, sigma)  
    xilist.append(xi)
    Nlist.append(j)
plt.hlines(1, 1, 10)
plt.ylim((0, 0.2))
plt.plot(Nlist, xilist)
plt.show()

popt, pcov = curve_fit(polN, Intervallog_x, Intervallog_y, ains)


logE = np.logspace(-1, 8, 100)


SED = SEDfromFIT(np.log10(logE), *popt)

#xitry = Xi2(Interval_y, SED, sigma, 8)

plt.plot(logE, SED, label = "Fitting quadratic")
plt.plot(Interval_x, Interval_y, '.', label = "Interval 4")

plt.ylabel(r"$E^2F (keVcm^{-2}s^{-1})$")
plt.xlabel(r"E (keV)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()
