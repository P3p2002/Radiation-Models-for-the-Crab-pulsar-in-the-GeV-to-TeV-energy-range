# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:17:27 2024

@author: Pep Rubi
Prova
"""

import numpy as np
import astropy.units as u
from astropy.constants import c

T     = 33*10**(-3)*u.s # període de Crab (s)
Omega = 2*np.pi/T       # velocitat angular
Rl    = c*T/(2*np.pi)   # radi de llum

import matplotlib.pyplot as plt


def gaussian(theta, theta_0, sigma):
    
    return np.exp(-(theta-theta_0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def phase_r(r, freq_r):
    
    #return 0.4
    return np.cos(r*freq_r)/(r*freq_r/(u.rad)) + 0.4

def anisotropy(phase, r, sigma, theta_0):
        
    
    return np.where(r > Rl , gaussian(phase, theta_0, sigma), 0)


"""
r_list = np.arange(0, 4.01, 0.01 )

theta_list = np.radians(np.arange(-90, 270, 1))

r_mesh, theta_mesh = np.meshgrid(r_list, theta_list)

sigma = 0.05
freq_w = 1
freq_r = 1
t = 0

theta0 = phase_r(r_mesh, freq_r)

function = anisotropy(theta_mesh,r_mesh, sigma, theta0)

fig, ax = plt.subplots(dpi = 120, subplot_kw=dict(projection = "polar"))
ax.contourf(theta_mesh, r_mesh, function ,1000, cmap = "plasma")
plt.show()

"""