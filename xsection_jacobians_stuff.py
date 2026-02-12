# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:34:27 2024

@author: Pep Rubi
"""

import numpy as np
from astropy.constants import c
from astropy.constants import m_e
import astropy.units as u

m = m_e*c**2
m = m.to('keV')
m_unitless = m/(u.keV)


#Aquesta funció em retorna d(cos(theta_L'))/(d(cos(theta_ER')))
def jacobian_obs_LR(theta_f, theta, beta, gamma):
    
    term1 = np.sin(theta_f)*(np.cos(theta)-beta)+np.sin(theta)*(np.cos(theta_f)-beta)
    term2 = 1/((1-beta*np.cos(theta))*(1-beta*np.cos(theta_f))**2*gamma**2)
    
    jacobian = (term1*term2)
    
    return jacobian

#Aquesta funció em retorna dE_foto_f/(dcos(theta_f))
def jacobian_energy_angle(theta, theta_f, beta, gamma, E_foto_i):

    term1 = E_foto_i*m*gamma*(1-beta*np.cos(theta))
    #I changed the sign before E_foto_i of the term2
    term2 = beta*m*gamma + E_foto_i*(np.cos(theta)-np.sin(theta)/np.tan(theta_f))
    #I changed the sign inside the E_foto_i of the term3, 
    #before we had E_foto_I(1-...)
    term3 = m*gamma*(1-beta*np.cos(theta_f)) + E_foto_i*(1-np.cos(theta_f+theta))
    
    jacobian = term1*term2/(term3**2)
    
    return jacobian

#Aquesta és la secció eficaç que surt del llibre Peskin
def exact_xsection(theta, theta_f, beta, gamma, E_foto_i, E_foto_f):
    
    pki = E_foto_i*m*gamma*(1-beta*np.cos(theta))
    pkf = E_foto_f*m*gamma*(1-beta*np.cos(theta_f))
    
    jacobian1 = jacobian_obs_LR(theta_f, theta, beta, gamma)
    jacobian2 = jacobian_energy_angle(theta, theta_f, beta, gamma, E_foto_i)

    factor = 2*(pkf/pki + pki/pkf + 2*m**2*(1/pki-1/pkf) + m**4*(1/pki-1/pkf)**2)
    
    x_section_LR =  factor*(pkf)**2/(4*pki**2*m**2*8*np.pi)
    
    x_section_obs = x_section_LR*(abs(jacobian1/jacobian2))
    
    return x_section_obs


"""
#Aqui simplement calculo quan val beta del segon boost
def second_boost(E_foto_i, theta, beta):
    
    return E_foto_i*(1-beta*np.cos(theta))/(E_foto_i*(1-beta*np.cos(theta)) + m)

#Aquesta funció em retorna el valor del cosinus de l'angle del centre de masses
#en funció de variables que ja conec
def theta_f_cm(theta_f, theta, beta, gamma, beta2, gamma2):
    
    term1 = (np.cos(theta_f)-beta)*(np.cos(theta)-beta) - np.sin(theta_f)*np.sin(theta)/(gamma**2)
    term2 = (1-beta*np.cos(theta_f))*(1-beta*np.cos(theta))
    
    cos_cm = (term1 - beta2*term2)/(term2 - beta2*term1)
    
    return cos_cm

#Aquesta funció em retorna d(cos(theta_f))/(d(cos(theta_cm')))
def jacobian_obs_cm(theta_f, theta, beta, gamma, beta2, gamma2):
    
    term1 = (np.cos(theta_f)-beta)*(np.cos(theta)-beta) - np.sin(theta_f)*np.sin(theta)/(gamma**2)
    term2 = np.cos(theta)-beta-np.sin(theta)*(np.cos(theta_f)-beta)/(gamma*(1-beta*np.cos(theta_f)))
   
    jacobian = gamma**2*gamma2**2*(1-beta*np.cos(theta_f)-beta2*term1/(1-beta*np.cos(theta)))**2*(1-beta*np.cos(theta))/term2
    
    return jacobian

#Aquesta és la secció eficaç en el High Energy limit que surt del llibre Peskin
def HEL_xsection(theta, theta_f, beta, gamma,beta2, gamma2, E_foto_i):
    
    s = 2*E_foto_i*gamma*m*(1-beta*np.cos(theta)) + m**2
    
    cos_theta_f_cm = theta_f_cm(theta_f, theta, beta, gamma, beta2, gamma2)
   
    jacobian1 = jacobian_obs_cm(theta_f, theta, beta, gamma, beta2, gamma2)
    jacobian2 = jacobian_energy_angle(theta, theta_f, beta, gamma,beta2, gamma2, E_foto_i)
    
    x_section_cm = 1/(2*m**2 +s*(1+cos_theta_f_cm))
    
    x_section_obs = x_section_cm*(jacobian1/jacobian2)
    
    return x_section_obs
"""