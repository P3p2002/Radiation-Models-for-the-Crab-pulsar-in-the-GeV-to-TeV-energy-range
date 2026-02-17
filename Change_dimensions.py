# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:56:56 2024

@author: Pep Rubi
"""


import numpy as np
    
#function to convert 2D arrays into 3d arrays
#La idea d'això és que es crein arrays en 3D de tal forma que la primera capa representi l'energia inicial dels fotons
#la segona capa representa quina gamma se li dona i la última, i per tant cada valor representa un valor de la distància
#La idea es que vagi de la seguent forma:
#A = [[[A(R_0, gamma_0, E_0),A(R_1, gamma_0, E_0), ... , A(R_n, gamma_0, E_0)], [A(R_0, gamma_1, E_0),A(R_1, gamma_1, E_0), ... , A(R_n, gamma_1, E_0)], ... [A(R_0, gamma_n', E_0),A(R_1, gamma_n', E_0), ... , A(R_n, gamma_n', E_0)]], [[A(R_0, gamma_0, E_1),A(R_1, gamma_0, E_1), ... , A(R_n, gamma_0, E_1)], [A(R_0, gamma_1, E_1),A(R_1, gamma_1, E_1), ... , A(R_n, gamma_1, E_1)], ... [A(R_0, gamma_n', E_1),A(R_1, gamma_n', E_1), ... , A(R_n, gamma_n', E_1)]], ... [[A(R_0, gamma_0, E_n''),A(R_1, gamma_0, E_n''), ... , A(R_n, gamma_0, E_n'')], [A(R_0, gamma_1, E_n''),A(R_1, gamma_1, E_n''), ... , A(R_n, gamma_1, E_n'')], ... [A(R_0, gamma_n', E_n''),A(R_1, gamma_n', E_n''), ... , A(R_n, gamma_n', E_n'')]]]
#On la gamma representa el valor final de la gamma, E és l'energia inicial del fotó, i R la distància, i n, n' i n'' poden ser diferents a priori ja que les llistes no tenen perque tenir les mateixes longituds
#I de fet això és encara més genèric ja que afegeix una dimensió a l'array que ja teniem abans

def add_dimension_R(i_array,new_array):
    """
    Repeat i_array along a new first axis of length len(new_array).

    Resulting shape:
        (len(new_array),) + i_array.shape
    """
    i_array = np.asarray(i_array)
    n = len(new_array)

    return np.broadcast_to(i_array, (n,) + i_array.shape)    

    ##El nom de "R" ve donat perque els arrays que facin això només tindran
    ##components que varien en la dimensió R
    #f_array = []
    ##El que estic fent es crear un array 3D tal que cada element de l'antic array
    ##sigui un element d'una nova llista
    #for i in range(len(new_array)):
    #    f_array.append(i_array)
    #return np.array(f_array)

#La idea d'aquesta funció és: partint d'un array qualsevol que varia amb 
#l'energia inicial del fotó i, per tant, té la mateixa longitud, et torna un 
#array tridimensionals el qual té les dimensions d'energia final, energia 
#inicial, i de R respectivament

def add_dim_e_fi(E_fotof, E_fotoi, R):
    """
    Build a 3D array with shape (len(E_fotof), len(E_fotoi), len(R)),
    whose values are E_fotoi broadcast over the other axes.

    Parameters
    ----------
    E_fotof : array-like
        Final photon energy grid (used only for its length).
    E_fotoi : astropy.units.Quantity or array-like
        Initial photon energy grid; values fill the output.
    R : array-like
        Radius grid (used only for its length).

    Returns
    -------
    out : same type as E_fotoi (if Quantity) or ndarray
        3D array broadcast from E_fotoi.
    """
    n_f = len(E_fotof)
    n_i = len(E_fotoi)
    n_r = len(R)

    # Reshape E_fotoi to (1, n_i, 1) and broadcast to (n_f, n_i, n_r)
    base = E_fotoi.reshape(1, n_i, 1)
    return np.broadcast_to(base, (n_f, n_i, n_r))

    #ones = np.ones(len(R))
    #new_array = []#És l'array que tornarè
    #h_array = []#És un array que només fa que ajudar en el còdi
    #
    #for a in E_fotoi:
    #    h_array.append(np.array(a*ones))#Creo un nou array en 2D tal que 
    #    #tingui les dimensions de l'E_fi i de R, i 
    #    #que només variï en E_fi
    #
    #
    #for i in E_fotof:
    #    new_array.append(np.array(h_array))
    #    #Aquest array ja té les dimenions que toquen de E_ff, E_fi i R
    #
    #return np.array(new_array)*(E_fotoi.unit)

def add_dim_e_ff(E_fotof, E_fotoi, R):
    """
    Build a 3D array with shape (len(E_fotof), len(E_fotoi), len(R)),
    whose values are E_fotof broadcast over the other axes.

    Parameters
    ----------
    E_fotof : astropy.units.Quantity or array-like
        Final photon energy grid; values fill the output.
    E_fotoi : astropy.units.Quantity or array-like
        Initial photon energy grid (used only for its length and unit).
    R : array-like
        Radius grid (used only for its length).

    Returns
    -------
    out : same type as E_fotof (if Quantity) or ndarray
        3D array broadcast from E_fotof.
    """
    n_f = len(E_fotof)
    n_i = len(E_fotoi)
    n_r = len(R)

    # Reshape E_fotof to (n_f, 1, 1) and broadcast to (n_f, n_i, n_r)
    base = E_fotof.reshape(n_f, 1, 1)
    return np.broadcast_to(base, (n_f, n_i, n_r))    

    ##El nom d'això també ve donat perque els arrays que tinguin això només 
    ##tindran components que variïn en E_ff(energia fotó final)
    #ones = np.ones(len(R))
    #new_array = []#És l'array que tornarè
    #h_array = []#És un array que només fa que ajudar en el còdi
    #
    #for a in E_fotoi:
    #    h_array.append(np.array(ones))#Creo un nou array en 2D tal que 
    #    #tingui les dimensions de l'E_fi i de R
    #    
    #for i in E_fotof:
    #    new_array.append(i*np.array(h_array))
    #    #Aquest array ja té les dimenions que toquen de E_ff, E_fi i R
    #    
    #return np.array(new_array)*(E_fotoi.unit)
