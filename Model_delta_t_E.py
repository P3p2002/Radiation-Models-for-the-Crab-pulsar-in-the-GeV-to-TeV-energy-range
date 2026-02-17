# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:46 2023

@author: Pep Rubi
"""


import numpy as np
from calculation_angles import *
from Change_dimensions import *
from Power_law_X_ray import *
from constants import m_keV, Omega, P
from astropy.constants import c
from astropy.constants import m_e
from astropy.constants import hbar
from astropy.constants import e
from astropy.constants import eps0
import matplotlib.pyplot as plt
import scienceplots
import astropy.units as u
import mpmath as mp

from scipy.optimize import fsolve

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
def gammaw(R, R0, Rf, gamma0, gammaw, alpha):

    R = np.asarray(R)

    # Handle degenerate interval safely
    if Rf <= R0:
        # If there's no ramp region, step at R0
        return np.where(R < R0, gamma0, gammaw).astype(float)

    # Normalized ramp coordinate, clipped to [0, 1]
    x = np.clip((R - R0) / (Rf - R0), 0.0, 1.0)

    return (gamma0 + (gammaw - gamma0) * x**alpha).astype(float)

    #gamma = np.ones(len(R))    
    #gamma = gamma * gamma_0
    #gamma[R>=R_0] += (gamma_w-gamma_0)*((R[R>=R_0]-R_0)/(R_f-R_0))**(alpha)
    #gamma[R>R_f]  = gamma_w
    #return np.array(gamma)

#Defineixo la funció del moment de la mateixa forma que fa a l'article
#On li demano les diferents variables que necesito
def M(R, R0, Rf, gamma_w, alpha):
    """
    Piecewise profile:
      - M = 0                                for R < R0
      - M = M_w * ((R-R0)/(Rf-R0))^alpha     for R0 <= R <= Rf
      - M = M_w                              for R > Rf
    where M_w = gamma_w * m / Omega
    """
    R = np.asarray(R)

    M_w = gamma_w * m_keV / Omega   # m is in reality m*c^2, then Eq. 2.5 is recovered

    # Handle degenerate interval safely
    if Rf <= R0:
        return np.where(R < R0, 0.0, M_w).astype(float)

    # Normalized ramp coordinate, clipped to [0, 1]
    x = np.clip((R - R0) / (Rf - R0), 0.0, 1.0)
    return (M_w * x**alpha).astype(float)    

    #M_w = gamma_w*m/Omega
    #M_j = np.zeros(len(R))*M_w    
    #M_j[R>=R_0] = ((R[R>=R_0]-R_0)/(R_f-R_0))**(alpha)*M_w
    #M_j[R>R_f]  = M_w 
    #return M_j

def theta(R, R0, Rf, RLC, gamma_w, Gamma, alpha):

    R = np.asarray(R)
    
    # Normalized ramp coordinate, clipped to [0, 1]
    x = np.clip((R - R0) / (Rf - R0), 0.0, 1.0)

    # M = gamma_w * x**alpha * m / Omega
    # Gamma = (gamma_0 + (gamma_w - gamma_0) * x**alpha)
    # RLC = c*P/(2*np.pi)   # Radius of the light cylinder
    # theta = arcsin(M*c/(Gamma*m*R*RLC)    

    arg = gamma_w * x**alpha / (Gamma * R)
    #print ('arg of theta: ', arg)
    #print ('x: ', x)
    #print ('Gamma: ', Gamma)
    #print ('gamma_w: ', gamma_w)
    
    return np.arcsin(arg)

def theta_init(beta, theta, E_fotoi, E_fotof):
    """
    An approximation of Eq. 3.19, assuming that epsilon-->0, 
    used for initialization only
    """
    arg = 1/beta - E_fotoi * (1/beta-np.cos(theta))/E_fotof
    #print ('arg=',arg)

    arg = np.where(arg > 1, 1., arg)
    
    return np.arccos(arg)
    
def theta_init_mp(beta, theta, E_fotoi, E_fotof, dps=50):
    """
    mpmath version of:
        arg = 1/beta - E_fotoi * (1/beta - cos(theta)) / E_fotof
        return arccos(arg)

    beta, theta, E_fotoi, E_fotof: 3D arrays with same shape
    Returns: array of mp.mpf with same shape
    """
    beta = np.asarray(beta)
    theta = np.asarray(theta)
    E_fotoi = np.asarray(E_fotoi)
    E_fotof = np.asarray(E_fotof)

    out = np.empty(beta.shape, dtype=object)

    with mp.workdps(dps):
        it = np.nditer(beta, flags=["multi_index"])
        for b in it:
            idx = it.multi_index

            b_mp  = mp.mpf(beta[idx])
            th_mp = mp.mpf(theta[idx])
            Ei_mp = mp.mpf(E_fotoi[idx])
            Ef_mp = mp.mpf(E_fotof[idx])

            arg = 1/b_mp - Ei_mp * (1/b_mp - mp.cos(th_mp)) / Ef_mp

            #print (arg)
            
            # Optional safety: clamp to [-1, 1] to avoid acos domain errors
            if arg > 1:
                #print ('beta: ',b_mp,' E_i: ',Ei_mp,' E_f: ',Ef_mp,' theta: ',th_mp, ' arg=',arg)
                arg = mp.mpf(1)
            if arg < -1: arg = mp.mpf(-1)

            out[idx] = mp.acos(arg)

    return out

def beta_f(gamma, dps=None):
    """
    beta = sqrt(1 - 1/gamma^2)

    - If gamma is array-like -> uses NumPy (fast)
    - If gamma is scalar and dps is set -> uses mpmath at given precision
    """
    if np.isscalar(gamma):
        if dps is None:
            return float(np.sqrt(1.0 - 1.0/(gamma*gamma)))
        with mp.workdps(dps):
            g = mp.mpf(gamma)
            return mp.sqrt(1 - 1/(g*g))
    else:
        g = np.asarray(gamma, dtype=float)
        return np.sqrt(1.0 - 1.0/(g*g))   


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
