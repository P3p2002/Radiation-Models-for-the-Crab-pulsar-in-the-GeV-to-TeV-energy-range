# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:32:51 2023

@author: Jan Corredor
"""

import numpy as np
from astropy.constants import c
from astropy.constants import m_e
from astropy.constants import hbar
from astropy.constants import e
from astropy.constants import eps0
from astropy.constants import h
import matplotlib.pyplot as plt
import scienceplots
import astropy.units as u

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


dE = 0.01
E = np.arange(3, 500, dE)#En keV
 
def Earth_spectrum(E_fotoi):
    b = 0.14
    a = 1.6
    F = E_fotoi*(-(a+b*np.log(E_fotoi)))
    return F
"""
plt.plot(E, F/norm)

plt.yscale("log")
plt.xscale("log")

plt.xlabel( "E(keV)", fontsize = 17)
plt.ylabel("F(E)" , fontsize = 17)
plt.savefig("X ray power law star")
"""