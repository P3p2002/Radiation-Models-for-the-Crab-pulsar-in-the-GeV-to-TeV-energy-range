# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:46 2023

@author: Pep Rubi
"""


import numpy as np
from calculation_angles import *
from Change_dimensions import *
from Power_law_X_ray import *
from astropy.constants import c
from astropy.constants import m_e
from astropy.constants import hbar
from astropy.constants import e
from astropy.constants import eps0
import matplotlib.pyplot as plt
import scienceplots
import astropy.units as u

from scipy.optimize import fsolve

m = m_e*c**2
m = m.to('keV')

plt.style.use(["science","no-latex"])
plt.rcParams["figure.figsize"] = (7,7)
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5

#Defineixo la funció gamma de la mateixa forma que ho fa a l'article
#On li demano les diferents variables que necesito
def gammaw(R, R_0, R_f, gamma_0, gamma_w, alpha):

    gamma = np.ones(len(R))
    
    gamma = gamma * gamma_0
    gamma[R>=R_0] += (gamma_w-gamma_0)*((R[R>=R_0]-R_0)/(R_f-R_0))**(alpha)
    gamma[R>R_f]  = gamma_w

    return np.array(gamma)

#Defineixo la funció del moment de la mateixa forma que fa a l'article
#On li demano les diferents variables que necesito
def M(R, R_0, R_f, gamma_w, alpha, Omega):
    
    M_w = gamma_w*m/Omega
    
    M_j = np.zeros(len(R))*M_w
    
    M_j[R>=R_0] = ((R[R>=R_0]-R_0)/(R_f-R_0))**(alpha)*M_w
    M_j[R>R_f]  = M_w
 
    return M_j

def beta_f(gamma):
    
    return np.sqrt(1-1/(gamma**2))

#Aquesta es la seccio eficaç que vam obtenir nosaltres

def display_sigfig(x, xerr, sigfigs=2) -> str:
    '''
    Suppose we want to show 2 significant figures. Implicitly we want to show 3 bits of information:
    - The order of magnitude
    - Significant digit #1
    - Significant digit #2
    '''

    if sigfigs < 1:
        raise Exception('Cannot have fewer than 1 significant figures. ({} given)'.format(sigfigs))

    order_of_magnitude = np.floor(np.log10(np.abs(xerr.value)))

    # Because we get one sigfig for free, to the left of the decimal
    decimals = (sigfigs - 1)

    xerr /= np.power(10, order_of_magnitude)
    xerr = np.round(xerr, decimals)
    xerr *= np.power(10, order_of_magnitude)

    # Subtract from decimals the sigfigs we get from the order of magnitude
    decimals -= order_of_magnitude
    # But we can't have a negative number of decimals
    decimals = int(max(0, decimals))

    return '{:.{dec}f}'.format(x.value, dec=decimals)+'+-{:.{dec}f}'.format(xerr.value, dec=decimals)


"""      
a = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],[[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
b = [1,2,3]
c = three_D(a,b)
print(c)
#Tot aixo son altres coses que les guardo per si vull fer alguna proba en aquest document

T     = 33*10**(-3)*u.s # període de Crab (s)
Omega = 2*np.pi/T       # velocitat angular

Rl    = c*T/(2*np.pi)   # radi de llum
delta_R = 0.01          # magnitud de cada pas per a l'integral (en unitats de Rl)


R  = np.arange(1, 200, 0.01)*Rl  # array de distancies respecte l'estrella de neutrons
a  = np.arcsin(Rl/R)             # array d'angles que s'obté a partir de la simplificació per a R>>Rl
w  = (1.-np.cos(a))/(R/Rl)                # pesos per a l'eficiencia de la dispersió IC


alpha = np.array([1])             # factors d'acceleració de l'article 
gamma_0 = 300           # factor gamma inicial que utilitza el model
gamma_w = np.arange(55*10**4, 1*10**6, 1*10**4)      # factors gamma final que utilitza el model
Ri    = np.array([1, 20, 25])*Rl         # radi on es comencen a accelerar els electronsque anirà a les funcions de gamma i del moment 
Rw    = np.array([30, 50, 70])*Rl           # radi final que diu el mateix model


Rf = Rw[0]
Ri2 = Ri[0]
for a in alpha:
    meandelta_ta, sddelta_ta = Delta_t(Omega, R, Ri2, Rf, gamma_0, gamma_w, a) 
"""
"""
def energia_foto_3D( E_fotoi, gamma):
    #Aquí estic creant un array en 3D de tal forma que sigui energia foto inicial en la primera capa
    #en la segona capa serà les diferents gammes, i en la tercera capa serà la distància
    #de tal forma que per la distribució d'energia necessito que per cada energia inicial del foto
    # i per cada gamma, existeixi un array de longitud R amb aquell valor concret de l'energia inicial
    #I la gamma que li passo seguirà el mateix argument, hi ha un exemple en la funció three_D que està definida a sota
    E_foto_3D = []
    ones_R = np.ones(len(gamma[0]))
    
    for i in range(len(E_fotoi)):
        rand_array = []
        for k in range(len(gamma)):
            rand_array.append(E_fotoi[i]*ones_R)
        E_foto_3D.append(np.array(rand_array))
    E_foto_3D = np.array(E_foto_3D)*u.J
    
    return E_foto_3D
"""