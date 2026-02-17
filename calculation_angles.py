# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:52:42 2024

@author: Pep Rubi
"""

import numpy as np
from astropy.constants import c
from astropy.constants import m_e
import astropy.units as u
from constants import m_keV
from scipy.optimize import fsolve, bisect, root_scalar

def eq_319(theta_f, theta_i, gamma, beta, E_out, E_in, m):
    """
    Eq. (3.19) rearranged as f(theta_f)=0.
    All angles must be in radians (floats), energies and m in consistent units.
    """
    ctf = np.cos(theta_f)
    cti = np.cos(theta_i)
    ctif = np.cos(theta_f + theta_i)

    A = m * gamma  # common factor
    return E_out * (A * (1 - beta * ctf) + E_in * (1 - ctif)) \
           - E_in * A * (1 - beta * cti)


def solve_theta_f(theta_i, gamma, beta, E_out, E_in, m, theta0,
                  xtol=1e-10, maxfev=4000):
    """
    Solve Eq. (3.19) for theta_f given parameters.

    Parameters
    ----------
    theta_i : float
        Incoming angle (rad)
    gamma, beta : float
    E_out, E_in : float
        Photon energies (same units)
    m : float
        Electron mass energy (same units as E)
    theta0 : float
        Initial guess for theta_f (rad)

    Returns
    -------
    theta_f : float
        Solution for theta_f (rad)
    """
    f = lambda x: eq_319(x, theta_i, gamma, beta, E_out, E_in, m)
    theta_f = fsolve(f, x0=float(theta0), xtol=xtol, maxfev=maxfev)[0]
    return theta_f

def solve_theta_f_bracketed(theta_i, gamma, beta, E_out, E_in, m,
                            theta0,
                            domain=(0.0, np.pi/2),   # forward scattering only 
                            step0=1e-3,
                            max_expand=60,
                            method="brentq",
                            xtol=1e-12, rtol=1e-12, maxiter=200):
    """
    Solve eq_319(theta_f)=0 in theta_f using a bracketing method.

    - Searches for a sign-change bracket around theta0 by expanding symmetrically.
    - Then uses brentq (or another bracketed method) to find the root.

    Returns
    -------
    theta_f : float (rad)
    """

    a_dom, b_dom = domain

    # Quick check if a solution exists: 
    xs = np.linspace(a_dom, b_dom, 2000)
    ys = [eq_319(x, theta_i, gamma, beta, E_out, E_in, m) for x in xs]

    if (min(ys) > 0 and max(ys) > 0) or (min(ys) < 0 and max(ys) < 0):
        raise ValueError(f"No root exists, min(Eq.3.19): {min(ys)}, max(Eq.3.19): {max(ys)}.")    
    
    f = lambda x: eq_319(x, theta_i, gamma, beta, E_out, E_in, m)

    # Clamp initial guess to domain
    theta0 = float(np.clip(theta0, a_dom, b_dom))

    # Evaluate at initial guess
    f0 = f(theta0)
    if f0 == 0.0:
        return theta0

    # Expand symmetric bracket around theta0 until sign change
    step = step0
    a = theta0
    b = theta0
    fa = f0
    fb = f0

    for _ in range(max_expand):
        a = max(a_dom, theta0 - step)
        b = min(b_dom, theta0 + step)
        fa = f(a)
        fb = f(b)

        if np.isfinite(fa) and np.isfinite(fb) and (fa == 0.0 or fb == 0.0 or fa * fb < 0.0):
            break

        # increase step exponentially
        step *= 1.6

        # If we've covered the whole domain and still no sign change, give up
        if a <= a_dom + 1e-15 and b >= b_dom - 1e-15:
            raise ValueError("Could not bracket a root in the specified domain.")

    # If exact root at endpoint
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if not (fa * fb < 0.0):
        raise ValueError("Bracket search ended without a sign change.")

    sol = root_scalar(f, bracket=(a, b), method=method, xtol=xtol, rtol=rtol, maxiter=maxiter)
    if not sol.converged:
        raise RuntimeError(f"root_scalar did not converge: {sol.flag}")
    return sol.root

def solve_theta_f_quantity(theta_i, gamma, beta, E_out, E_in, theta0, **kw):
    """
    Use solve_theta_f with astropy units
    """
    
    theta_i_rad = theta_i.to_value(u.rad)
    theta0_rad  = theta0.to_value(u.rad)

    # Convert energies/m to plain floats in the same unit (example: eV)
    unitE     = E_out.unit
    E_out_val = E_out.to_value(unitE)
    E_in_val  = E_in.to_value(unitE)
    m_val     = m_keV.to_value(unitE)

    #sol = solve_theta_f(theta_i_rad, gamma, beta, E_out_val, E_in_val, m_val, theta0_rad, **kw)
    sol = solve_theta_f_bracketed(theta_i_rad, gamma, beta, E_out_val, E_in_val, m_val, theta0_rad, **kw)

    print ('E_out: ', E_out, ' E_in: ', E_in, ' gamma: ', gamma, ' theta_f_e: ', sol)
    
    return sol # * u.rad

def compute_theta_f_exact(theta_init, theta, Gamma_3d, beta,
                          E_fotof_3d, E_fotoi_3d, E_fotof_min, E_fotof_max,
                          fill_value=10000.0):
    """
    Compute theta_f_e preallocated and cleaner.
    """
    # Shapes: assume E_fotof_3d is (nEfof, nEfoi, nR)
    nJ, nK, nI = E_fotof_3d.shape

    out = np.full((nJ, nK, nI), fill_value, dtype=float)

    # Validity mask: note min/max are (nK, nI), broadcast to (nJ, nK, nI)
    valid = (E_fotof_3d > E_fotof_min[None, :, :]) & (E_fotof_3d < E_fotof_max[None, :, :])

    # Iterate only over valid points (still a loop, but much smaller)
    idxs = np.argwhere(valid)
    for j, k, i in idxs:
        x0 = theta_init[j, k, i]
        out[j, k, i] = solve_theta_f_quantity(
            theta[j, k, i],
            Gamma_3d[j, k, i],
            beta[j, k, i],
            E_fotof_3d[j, k, i],
            E_fotoi_3d[j, k, i],
            x0
        )
            #solver(x0,
            #       theta[j, k, i],
            #       Gamma_3d[j, k, i],
            #       beta[j, k, i],
            #       E_fotof_3d[j, k, i],
            #       E_fotoi_3d[j, k, i])
    return out * u.rad

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
