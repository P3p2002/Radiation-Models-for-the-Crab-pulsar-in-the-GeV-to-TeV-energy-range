# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:52:42 2024

@author: Pep Rubi
"""

import sys
import numpy as np
from astropy.constants import c
from astropy.constants import m_e
import astropy.units as u
from constants import m_keV
from scipy.optimize import fsolve, bisect, root_scalar
import matplotlib.pyplot as plt
from numba import njit

@njit
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
                            domain=(0.0, 2*np.pi),   # forward scattering only 
                            step0=1e-31,
                            max_expand=10000,
                            method="brentq",
                            xtol=1e-14, rtol=1e-14, maxiter=3000):
    """
    Solve eq_319(theta_f)=0 in theta_f using a bracketing method.

    - Searches for a sign-change bracket around theta0 by expanding symmetrically.
    - Then uses brentq (or another bracketed method) to find the root.

    Returns
    -------
    theta_f : float (rad)
    """

    a_dom, b_dom = domain
    L = b_dom - a_dom        

    def wrap(theta):
        return (theta - a_dom) % L + a_dom
    
    # Quick check if a solution exists: 
    xs = np.linspace(a_dom, b_dom, 2000, endpoint=False)
    ys = np.array([eq_319(x, theta_i, gamma, beta, E_out, E_in, m) for x in xs])
    #plt.plot(xs, ys)

    if np.all(ys > 0) or np.all(ys < 0):
        print(ys)
        raise ValueError(
            f"No root exists, min(Eq.3.19)={ys.min()}, max(Eq.3.19)={ys.max()} "
            f"for theta_i={theta_i}, gamma={gamma}, beta={beta}, "
            f"E_out={E_out}, E_in={E_in}, m={m}"
        )    
    
    # Clamp initial guess to domain
    #theta0 = float(np.clip(theta0, a_dom, b_dom))

    # wrap initial guess into periodic domain
    theta0 = wrap(theta0)    

    # ensures that values are mapped into [a_dom, b_dom),
    # distances remain meaningful and the periodic continuity is preserved
    theta0 = (theta0 - a_dom) % L + a_dom
    
    # recenter the problem around theta0
    def f_local(x):
        return eq_319(wrap(theta0 + x), theta_i, gamma, beta, E_out, E_in, m)
    
    #f = lambda x: eq_319((x - a_dom) % L + a_dom, theta_i, gamma, beta, E_out, E_in, m)    

    # Evaluate at initial guess
    #f0 = f(theta0)
    f0 = f_local(0.0)
    if f0 == 0.0:
        return theta0

    # Expand symmetric bracket around theta0 until sign change
    step = step0
    #a = theta0
    #b = theta0
    a = -step
    b = step
    #fa = f0
    #fb = f0
    fa = f_local(a)
    fb = f_local(b)

    for _ in range(max_expand):

        #a = max(a_dom, theta0 - step)
        #b = min(b_dom, theta0 + step)
        # use periodic wrapping , now the bracket can cross the boundary
        #a = (theta0 - step - a_dom) % L + a_dom
        #b = (theta0 + step - a_dom) % L + a_dom

        #print (f"a={a}, a_dom={a_dom}, b={b}, b_dom={b_dom}, fa={fa}, fb={fb}")

        if np.isfinite(fa) and np.isfinite(fb) and (fa == 0.0 or fb == 0.0 or fa * fb < 0.0):
            break

        # increase step exponentially
        step *= 1.1

        a = -step
        b =  step
        
        fa = f_local(a)
        fb = f_local(b)

        if step > 0.5*L:

            if ys[0] < 0:
                return theta0
            
            #np.set_printoptions(threshold=np.inf)
            #print(ys)
            #print(xs)                    
            raise ValueError(
                f"Could not bracket root within one full periodic domain: "
                f"theta0={theta0}, step={step}, fa={fa}, fb={fb}"
            )
        # If we've covered the whole domain and still no sign change, give up
        #if a <= a_dom + 1e-15 and b >= b_dom - 1e-15:            
        #    raise ValueError(f"Could not bracket a root in the specified domain: a={a}, a_dom={a_dom}, b={b}, b_dom={b_dom}")

    # If exact root at endpoint
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if not (fa * fb < 0.0):
        raise ValueError("Bracket search ended without a sign change.")

    sol = root_scalar(f_local, bracket=(a, b), method=method, xtol=xtol, rtol=rtol, maxiter=maxiter)
    if not sol.converged:
        raise RuntimeError(f"root_scalar did not converge: {sol.flag}")

    theta_root = wrap(theta0 + sol.root)
    return theta_root

@njit
def wrap_angle(theta, a_dom, L):
    return (theta - a_dom) % L + a_dom

@njit
def f_local_numba(x, theta0, a_dom, L,
                  theta_i, gamma, beta, E_out, E_in, m):
    theta = wrap_angle(theta0 + x, a_dom, L)
    return eq_319(theta, theta_i, gamma, beta, E_out, E_in, m)

@njit
def solve_theta_f_bracketed_fast(theta_i, gamma, beta, E_out, E_in, m,
                                 theta0,
                                 domain=(0.0, 2*np.pi),
                                 step0=1e-8,
                                 expand_factor=2.0,
                                 max_expand=100,
                                 method="brentq",
                                 xtol=1e-12,
                                 rtol=1e-12,
                                 maxiter=200,
                                 check_exists=True,
                                 ncheck=512):
    """
    Faster bracketed theta_f solver.

    Assumes all inputs are plain floats, not Astropy Quantities.
    """

    a_dom, b_dom = domain
    L = b_dom - a_dom

    theta0 = wrap_angle(theta0, a_dom, L)

    f0 = f_local_numba(0.0,theta0, a_dom, L,
                       theta_i, gamma, beta, E_out, E_in, m)

    if not np.isfinite(f0):
        return 50000.0, True  #raise ValueError(f"Non-finite f0 at theta0={theta0}: {f0}")

    if f0 == 0.0:
        return theta0, False

    #if check_exists:
    #    xs = np.linspace(a_dom, b_dom, ncheck, endpoint=False)
    #    ys = np.array([
    #        eq_319(x, theta_i, gamma, beta, E_out, E_in, m)
    #        for x in xs
    #    ])
    #
    #    if np.all(ys > 0) or np.all(ys < 0):
    #        raise ValueError(
    #            f"No root exists, min={ys.min()}, max={ys.max()}, "
    #            f"theta_i={theta_i}, gamma={gamma}, beta={beta}, "
    #            f"E_out={E_out}, E_in={E_in}, m={m}"
    #        )

    if check_exists:
        y0 = eq_319(a_dom, theta_i, gamma, beta, E_out, E_in, m)

        all_pos = y0 > 0.0
        all_neg = y0 < 0.0

        dx = (b_dom - a_dom) / ncheck

        ymin = y0
        ymax = y0

        for q in range(1, ncheck):
            x = a_dom + q * dx
            y = eq_319(x, theta_i, gamma, beta, E_out, E_in, m)
            
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y

            all_pos = all_pos and (y > 0.0)
            all_neg = all_neg and (y < 0.0)

        if all_pos or all_neg:
            return 40000.0, True

    step = step0

    for _ in range(max_expand):
        a = -step
        b = step

        fa = f_local_numba(a,theta0, a_dom, L,
                     theta_i, gamma, beta, E_out, E_in, m)
        fb = f_local_numba(b,theta0, a_dom, L,
                     theta_i, gamma, beta, E_out, E_in, m)
        
        if np.isfinite(fa) and np.isfinite(fb):
            if fa == 0.0:
                return wrap_angle(theta0 + a, a_dom, L), False
            if fb == 0.0:
                return wrap_angle(theta0 + b, a_dom, L), False
            if fa * fb < 0.0:
                break

        step *= expand_factor

        if step > 0.5 * L:
            return 30000.0, True
        #raise ValueError(
        #f"Could not bracket root within half periodic domain: "
        #        f"theta0={theta0}, step={step}, fa={fa}, fb={fb}"
        #    )
    else:   # The else clause executes after the loop completes normally. This means that the loop did not encounter a break statement.
        return 20000.0, True  # raise ValueError("Bracket search ended without sign change.")

    #sol = root_scalar(
    #    f_local,
    #    bracket=(a, b),
    #    method=method,
    #    xtol=xtol,
    #    rtol=rtol,
    #    maxiter=maxiter,
    #)
    #
    #if not sol.converged:
    #    raise RuntimeError(f"root_scalar did not converge: {sol.flag}")
    #return wrap_angle(theta0 + sol.root, a_dom, L)
    
    # NEED TO REPLACE ROOT_SCALAR BY OWN FUNCTION WHICH COMPILES ON NJIT

    left = a
    right = b

    fleft = f_local_numba(
        left, theta0, a_dom, L,
        theta_i, gamma, beta, E_out, E_in, m
    )
    
    fright = f_local_numba(
        right, theta0, a_dom, L,
        theta_i, gamma, beta, E_out, E_in, m
    )
    
    if fleft == 0.0:
        return wrap_angle(theta0 + left, a_dom, L), False
    
    if fright == 0.0:
        return wrap_angle(theta0 + right, a_dom, L), False
    
    if fleft * fright > 0.0:
        return 10000.0, True

    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        
        fmid = f_local_numba(
            mid, theta0, a_dom, L,
            theta_i, gamma, beta, E_out, E_in, m
        )
        
        if fmid == 0.0 or abs(right - left) < xtol:
            return wrap_angle(theta0 + mid, a_dom, L), False
        
    if fleft * fmid < 0.0:
        right = mid
        fright = fmid
    else:
        left = mid
        fleft = fmid
        
    return wrap_angle(theta0 + 0.5 * (left + right), a_dom, L), False


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

    #print ('E_out: ', E_out, ' E_in: ', E_in, ' gamma: ', gamma, ' theta_f_e: ', sol)
    
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

    print (E_fotof_max)


    valid = (E_fotof_3d >= E_fotof_min) & (E_fotof_3d <= E_fotof_max)

    # Iterate only over valid points (still a loop, but much smaller)
    idxs = np.argwhere(valid)

    #for j, k, i in sorted(idxs, key=lambda x: x[0], reverse=True):    
    for j, k, i in idxs:
        print(j, k , i)
        #print(j, k , i, '\r', end='')
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

@njit
def _solve_one_flat(n, theta_i, gamma, beta, E_out, E_in, theta0, m_val):
    val, isok = solve_theta_f_bracketed_fast(
        theta_i,
        gamma,
        beta,
        E_out,
        E_in,
        m_val,
        theta0,
        step0=1e-8,
        expand_factor=2.0,
        check_exists=True,
    )

    if np.isnan(val):
        return n, 60000.0, True

    return n, val, isok

def compute_theta_f_exact_parallel(theta_init, theta, Gamma_3d, beta,
                                   E_fotof_3d, E_fotoi_3d,E_fotof_min, E_fotof_max,
                                   fill_value=10000.0,
                                   n_jobs=-1):

    from joblib import Parallel, delayed

    nJ, nK, nI = E_fotof_3d.shape

    out = np.full((nJ, nK, nI), fill_value, dtype=float)

    valid = (
        (E_fotof_3d >= E_fotof_min)
        & (E_fotof_3d <= E_fotof_max)
    )

    idxs = np.argwhere(valid)

    # Convert units once for better performance
    theta_val = theta.to_value(u.rad)
    theta_init_val = theta_init.to_value(u.rad)

    unitE = E_fotof_3d.unit
    E_fotof_val = E_fotof_3d.to_value(unitE)
    E_fotoi_val = E_fotoi_3d.to_value(unitE)
    m_val = m_keV.to_value(unitE)

    # This avoids repeated 3D indexing in the workers and makes the loop faster
    jj, kk, ii = np.where(valid)

    theta_i_arr = theta_val[jj, kk, ii]
    gamma_arr   = Gamma_3d[jj, kk, ii]
    beta_arr    = beta[jj, kk, ii]
    Eout_arr    = E_fotof_val[jj, kk, ii]
    Ein_arr     = E_fotoi_val[jj, kk, ii]
    theta0_arr  = theta_init_val[jj, kk, ii]
    
    n_jobs=8
    batch_size=500
    
    values = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=batch_size,
    )(
        delayed(_solve_one_flat)(
            n,
            theta_i_arr[n],
            gamma_arr[n],
            beta_arr[n],
            Eout_arr[n],
            Ein_arr[n],
            theta0_arr[n],
            m_val,
        )
        for n in range(len(jj))
    )

    failed = []
    
    for n, val, bad in results:
        values[n] = val
        if bad:
            failed.append(n)

    print("Number of failed points:", len(failed))

    if failed:
        n = failed[0]
        print("First failed point:")
        print("j,k,i =", jj[n], kk[n], ii[n])
        print("theta_i =", theta_i_arr[n])
        print("gamma   =", gamma_arr[n])
        print("beta    =", beta_arr[n])
        print("E_out   =", Eout_arr[n])
        print("E_in    =", Ein_arr[n])
        print("theta0  =", theta0_arr[n])    
    
    out[jj, kk, ii] = values    

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
