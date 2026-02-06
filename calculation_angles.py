# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:52:42 2024

@author: Pep Rubi
"""

import numpy as np
from astropy.constants import c
from astropy.constants import m_e
import astropy.units as u
m = m_e*c**2
m = m.to('keV')
m_unitless = m/(u.keV)

from scipy.optimize import fsolve
from scipy.optimize import bisect

#Per solucionar això ho haurè de fer elements per element de la funció, 
# és a dir, que haurè de fer un bucle tridimensional
def equation_solve(theta_f, theta_i, gamma, beta, E_foto_f, E_foto_i, m):

    # this solves Eq. (3.19) for theta_f, given the input values gamma, beta, theta_i, and the ingoing and outgoing photon energies
    # considering that: theta_f --> theta_L'
    #                   theta_i --> theta_L
    #                   E_foto_f --> E_gamma
    #                   E_foto_i --> epsilon
    eq = E_foto_f*(m*gamma*(1-beta*np.cos(theta_f)) + E_foto_i*(1 - np.cos(theta_f + theta_i))) - E_foto_i*m*gamma*(1-beta*np.cos(theta_i))
    
    return eq

#Funcio que em soluciona l'equacio que li pasi en funció d'un parametre
def solver(initial_value, theta_i, gamma, Beta, E_foto_f, E_fotoi):
    #x_tol és la toleràcnia que vull d'error
    #maxev  és el numèro d'iteracions màximes que agafa, de normal son 200, no ho hauria de canviar
    #initial_value valor pel qual comença a buscar la solució

    p0 = fsolve(lambda x: equation_solve(x*u.rad, theta_i, gamma, Beta, E_foto_f, E_fotoi, m), initial_value, xtol = 1e-8, maxfev = 4000)

    return p0

#primera aproximacio de l'angle final
def theta_f(E_fotoi, E_fotof, Beta, gamma, theta_i):
    
    cos_thetaf = 1/Beta - (1-Beta*np.cos(theta_i))*E_fotoi/(E_fotof*Beta) + (1-np.cos(theta_i))*E_fotoi/(m*gamma*Beta)

    cos_theta_f = np.where(cos_thetaf < 1, cos_thetaf, 1)
    #La funció np.where el que fa és que al passar la condició  et retorna
    #El primer valor que li dones si la condicio es certa, o el segon valor 
    #Si la condició es falsa
    
    thetaf = np.arccos(cos_theta_f)
    
    return thetaf, cos_theta_f

#Un altre tipus de solucionadro numeric, pero mes lent
def solve_bisect(initial_point, final_point, theta_i, gamma, Beta, E_foto_f, E_fotoi):

    p1 = bisect(lambda x: equation_solve(x, theta_i, gamma, Beta, E_foto_f, E_fotoi, m_unitless), initial_point, final_point, xtol = 1e-10, maxiter = 2000)
    
    return initial_point
