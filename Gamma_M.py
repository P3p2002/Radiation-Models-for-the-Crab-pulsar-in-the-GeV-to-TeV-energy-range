# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:46 2023

@author: Pep Rubi
"""


import numpy as np
from calculation_angles import *
from Change_dimensions import *
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

    gamma = gamma_0 * np.ones(len(R))
    
    gamma[R>=R_0] += (gamma_w-gamma_0)*((R[R>=R_0]-R_0)/(R_f-R_0))**alpha
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

def weighted_mean(a,w):
    return np.sum(a*w)/np.sum(w)

# see https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
def weighted_sd(a,w,mean):
    N = np.count_nonzero(w)
    return np.sqrt(np.sum((a-mean)*(a-mean)*w)/np.sum(w)*N/(N-1)) 

#Aqui he separat en el cas de mes d'una ja que la funció axis = 1 em dona problemes
#I no he trobat cap altre forma de ferho mes simple
def weighted_means(a,w):
    
    #Defineixo el smt per comoditat, no se si es mes rapid sense definir-lo
    smt = a*w
    
    mean = np.sum(smt, axis = 2)/np.sum(w, axis = 2)#Axis = 1 indica que fa les mitjes per files
    #Si fos axis = 0 les mitges serien per columnes, pero per una dimensio no funciona
    
    return mean


def weighted_sds(a,w,mean):
    
    b = a#Si no ho defineixo així els valors de a canvien
    
    N = np.count_nonzero(w, axis = 2)
    #faig aquest bucle per poder restar element a element les llistes
    for j in range(len(a)):
        b[j] -= mean[j]
    sd = np.sqrt(np.sum((b)*(b)*w, axis = 2)/np.sum(w, axis = 2)*N/(N-1)) 
    return sd

#La formula que s'utilitza de l'energia final del fotó en llenguatge latex és
#E_{\gamma}^{f} = m_e*c^2*E_{\gamma}^{i}*\frac{(1-\beta*\cos(\theta))}{m_e*c^2*\gamma*(1-\beta)+E_{\gamma}^{i}*(1+\cos(\theta))}
def beta_f(gamma):
    
    return np.sqrt(1-1/(gamma**2))

#Aquesta es la seccio eficaç que vam obtenir nosaltres
def dsigma_dEff(E_fotoi, E_fotof, gamma, beta, theta_i, thetaf):
    
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    
    cos_f = np.cos(thetaf)
    sin_f = np.sin(thetaf)
    
    primera_part = 1-((cos_i-beta)*(cos_f-beta) - sin_i*sin_f/gamma**2)**2*1/((1-beta*cos_i)*(1-beta*cos_f))**2
        
    segona_part = (sin_i*(cos_f-beta) + sin_f*(cos_i-beta))/(gamma**2*(1-beta*cos_i)*(1-beta*cos_f)**2)
    
    term1 = E_fotoi*m*gamma*(1-beta*np.cos(theta_i))
    term2 = beta*m*gamma + E_fotoi*(np.cos(theta_i)-np.sin(theta_i)*np.cos(thetaf))
    term3 = m*gamma*(1-beta*np.cos(thetaf)) + E_fotoi*(1-np.cos(thetaf+theta_i))
    
    jacobian = term1*term2/(term3**2)
    
    #segona_part = (sin_i + (cos_i-beta))*E_fotoi_3d/(Gamma_3d**2*(1-beta*cos_f)**2*beta*E_fotof_3d**2)

    return primera_part*segona_part/jacobian

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

def Delta_t(Omega, R, Ri, Rf, Rl, gamma_0, gamma_w, alpha, E_fotoi, E_foto_f):
    
    Gamma = gammaw(R.value, Ri.value, Rf.value,
                   gamma_0, gamma_w, alpha)       # array de factors gamma per a cada distància, segons el model de vent

    M_i   = M(R.value, Ri.value, Rf.value,
                   gamma_w, alpha, Omega)                # array de moments angulars que s'emporten els electrons
   
    theta = np.arcsin(M_i*c/(Gamma*m*R))        # array d'angles de dispersió entre electrons i fotons
    
    Gamma_2d = add_dimension_R(Gamma, E_fotoi)

    Gamma_3d = add_dimension_R(Gamma_2d, E_foto_f)
    
    Beta = beta_f(Gamma_3d)
    
    theta_2d = add_dimension_R(theta, E_fotoi)
    
    theta_3d = add_dimension_R(theta_2d, E_foto_f)

    w2    = (1.-np.cos(theta_3d))
    
    #Tot i que R i theta no tinguin les mateixes dimensions, al passar que theta és una llista( o llista de llistes) que conté arrays de la dimensió de R 
    #El que fa el codi es multpilicar els elements més interns de theta, es a dir, la llista amb la mateixa longitud que R, per R
    
    delta_t2 = 1/Omega+R*(1-np.cos(theta_3d))/c
    
    E_fotoi_3d = add_dim_e_fi(E_foto_f, E_fotoi, R)
    
    E_fotof_3d = add_dim_e_ff(E_foto_f, E_fotoi, R)
    
    thetaf, cos_theta_ff = theta_f(E_fotoi_3d, E_fotof_3d, Beta, Gamma_3d, theta_3d)
   
    theta_3d = theta_3d*(u.rad)

    x_section = dsigma_dEff(E_fotoi_3d, E_fotof_3d, Gamma_3d, Beta, theta_3d, thetaf)
        
    return theta_3d, Gamma_3d, delta_t2, E_fotoi_3d, E_fotof_3d, thetaf, x_section, cos_theta_ff

def integrals(x_section, thetaf, spectrum):
    
    a = x_section*spectrum*(1-np.cos(thetaf))
    
    return np.sum(a, axis = 1)
    
#Aquesta funció simplement serveix per graficar el que vull
def Graficar(alpha, Omega, R, Ri, Rf, Rl, gamma_0, gamma_w, E_fotoi, E_foto_f):
    
    for k in range(len(Ri)):
        for j in range(len(alpha)):
            theta, Gamma_3d, delta_t2, E_fotoi_3d, E_fotof_3d, thetaf = Delta_t(Omega, R, Ri[k], Rf[k], Rl, gamma_0, gamma_w, alpha[j], E_fotoi, E_foto_f)
            
            """
            for l in range(len(E_fotoi)):
                plt.plot(gamma_w, delta_ta[l], label = 'alpha, E_gamma = ' + str(alpha[j])+ ',' + str(E_fotoi[l]))
            plt.xlabel(r"$\Gamma$", fontsize = 17)
            plt.ylabel(r"$\Delta t(s)$", fontsize = 17)
            plt.title('Cas amb: Ri(Rl) = ' + str(Ri[k]/Rl) + ',Rf(Rl) = ' + str(Rf[k]/Rl) + ' i Gamma_w = ' + str(gamma_w), fontsize = 17)
            plt.legend(fontsize = 17)
            plt.show()
            """
    return theta, Gamma_3d, delta_t2, E_fotoi_3d, E_fotof_3d, thetaf

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
