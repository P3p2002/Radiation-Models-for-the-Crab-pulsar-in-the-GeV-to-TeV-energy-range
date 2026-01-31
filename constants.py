import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.constants import m_e
from astropy.constants import hbar
from astropy.constants import e
from astropy.constants import eps0
from astropy.constants import h

m = m_e*c**2             # electron mass in J 
m = m.to('keV')          # electron mass in keV
m_unitless = m/(u.keV)   # electron mass in keV unitless

# Bohr radius in m 
r_0 = (e.value)**2/(4*np.pi*eps0.value*m_e.value*c.value**2)*u.m
# Electron charge in natural units  alpha = e^2/(4pi) \approx 1/137
e2 = 4*np.pi/137

kevs_m = 8.07e8/(u.m*u.keV)  # ???  
freq_r = 1*u.rad/(u.m)       # Frequency in radians / m

# Crab pulsar constants
P     = 33*10**(-3)*u.s # Crab pulsar period (s)
Omega = 2*np.pi/P       # Pulsar angular velocity
RLC   = c*P/(2*np.pi)   # Radius of the light cylinder
Lsd   = 4.6e31*u.J/u.s  # Spin-down luminosity



