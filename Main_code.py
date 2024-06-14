# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:14:49 2023

@author: Pep Rubi
"""

import numpy as np
import math as m
from Model_delta_t_E import *
from calculation_angles import *
from Change_dimensions import *
from xsection_jacobians_stuff import *
from f_function import *
import astropy.units as u
from astropy.constants import c
from astropy.constants import m_e
from astropy.constants import hbar
from astropy.constants import e
from astropy.constants import eps0
from astropy.constants import h
import matplotlib.pyplot as plt
import pandas as pds
import scienceplots

#The first number indicates the interval and the second one which peak

phase_1_2 = [0.19615384615384626, 0.2153846153846154, 0.2384615384615385, 0.2576923076923078, 0.27564098651592556, 0.2948717557466948, 0.3128204345703125, 0.33076923076923076, 0.3423076923076924, 0.35512812687800477, 0.3705127422626201, 0.37948714036207926, 0.3897435114933895, 0.40769230769230774, 0.4128204345703125, 0.4166665884164663, 0.423076923076923, 0.43076923076923074, 0.4397435114933894, 0.45384615384615384, 0.46923076923076923, 0.4884615384615382, 0.5051281268780048, 0.5217948326697714, 0.5423076923076922, 0.5666665884164662, 0.5897435114933893, 0.6153846153846152, 0.6397435114933894, 0.6679486788236175]
f_1_2 = [0.28806817550316777, 0.35391077980497765, 0.4197533841067873, 0.5843644159831451, 0.650207020284955, 0.8477378472716066, 1.1769568969431, 1.539095741724887, 1.9341573956981903, 2.395061653973303, 2.8559689263296386, 3.382715788906561, 3.942388474756222, 3.843623061262896, 3.3497959937962674, 2.8559689263296386, 2.4609072723563354, 2, 1.4732531374230773, 1.0781914834497741, 0.7160496245867647, 0.5514416067916292, 0.35391077980497765, 0.28806817550316777, 0.22222255712013572, 0.09053734851651618, 0.09053734851651618, 0.05761453932500027, 0.05761453932500027, 0.05761453932500027]

phase_2_2 = [0.1935896653395434, 0.2166665884164665, 0.23589735764723563, 0.25897428072415873, 0.27820504995492795, 0.29871790959284855, 0.3153846153846154, 0.3294871403620793, 0.3448717557466947, 0.3576923076923076, 0.36923076923076925, 0.37948714036207926, 0.3910256019005409, 0.40769230769230774, 0.4128204345703125, 0.41794867882361775, 0.424358896108774, 0.43461538461538457, 0.4474358191856971, 0.46025637113131007, 0.4769230769230769, 0.49358966533954307, 0.51025637113131, 0.5282050499549278, 0.5525640634390022, 0.5769230769230768, 0.6038461538461537, 0.6269230769230768, 0.6499999999999999, 0.6782050499549276]
f_2_2 = [0.3209879706134615, 0.4197533841067873, 0.45267619329830344, 0.650207020284955, 0.8477378472716066, 1.1440340877515838, 1.3744877239297515, 1.7695493779030547, 2.1316882226848417, 2.625515290151471, 3.1851849619199095, 3.5802466158932127, 4.106996492551358, 3.843623061262896, 3.2839503754132355, 2.7242807036447965, 2.1316882226848417, 1.637861155218213, 1.1440340877515838, 0.7818952429697967, 0.5185187976001131, 0.4197533841067873, 0.28806817550316777, 0.12345714362680993, 0.12345714362680993, 0.09053734851651618, 0.09053734851651618, 0.09053734851651618, -0.008228064976809613, -0.008228064976809613]
    
phase_3_2 = [0.18974351149338944, 0.20641021728515627, 0.22692307692307706, 0.2461538461538462, 0.2666665884164664, 0.28846153846153855, 0.3089742807241588, 0.3282050499549279, 0.3423076923076924, 0.35512812687800477, 0.36410252497746387, 0.3705127422626201, 0.37948714036207926, 0.387179448054387, 0.4038461538461538, 0.40897428072415865, 0.4128204345703125, 0.42051274226262014, 0.42820504995492786, 0.4358973576472356, 0.4525640634390024, 0.4717948326697714, 0.4910256019005409, 0.5115384615384613, 0.5320512038010816, 0.5589742807241584, 0.583333294208233, 0.6064102172851562, 0.633333294208233, 0.6589742807241583, 0.6833332942082329]
f_3_2 = [0.4197533841067873, 0.45267619329830344, 0.5843644159831451, 0.650207020284955, 0.8148150380800905, 1.0452686742582582, 1.3744877239297515, 1.8024691730133484, 2.2304536361781677, 2.526749876658145, 2.954734339822964, 3.3497959937962674, 3.8107002520713804, 4.2386847152362, 4.139919301742873, 3.613169425084729, 3.0534997533162898, 2.493827067466629, 1.9341573956981903, 1.4074075190400452, 0.9465032607649324, 0.5843644159831451, 0.28806817550316777, 0.12345714362680993, 0.09053734851651618, 0.05761453932500027, 0.05761453932500027, 0.05761453932500027, 0.05761453932500027, 0.024691730133484135, -0.041150874168325746]

phase_4_2 = [0.19230769230769232, 0.21025637113131024, 0.2320512038010819, 0.2538461538461539, 0.2743588961087741, 0.2961538461538462, 0.3153846153846154, 0.3320512038010819, 0.347435819185697, 0.3576923076923076, 0.3679486788236178, 0.3769230769230769, 0.3884615384615384, 0.40512812687800476, 0.4102563711313101, 0.4141025249774639, 0.4217948326697716, 0.42820504995492786, 0.437179448054387, 0.44999999999999996, 0.4653846153846152, 0.4884615384615382, 0.5076923076923077, 0.5320512038010816, 0.5551281268780046, 0.5794871403620792, 0.6025640634390023, 0.6243588961087739, 0.6461538461538459, 0.666666588416466]
f_4_2 = [0.5514416067916292, 0.5843644159831451, 0.650207020284955, 0.7818952429697967, 1.0452686742582582, 1.2757223104364257, 1.6049383460266968, 2.032922809191516, 2.4279844631648193, 2.921811530631448, 3.2181077711114257, 3.7119348385780544, 4.2386847152362, 4.041153888249548, 3.5144040115914033, 2.8888887214399324, 2.3621418588630094, 1.8353919822048645, 1.3415649147382354, 0.9465032607649324, 0.5843644159831451, 0.35391077980497765, 0.22222255712013572, 0.15637995281832606, 0.09053734851651618, 0.09053734851651618, 0.09053734851651618, 0.024691730133484135, 0.024691730133484135, -0.008228064976809613]

phase_5_2 = [0.1935896653395434, 0.21410252497746396, 0.23717944805438707, 0.26025637113131017, 0.28205120380108184, 0.3025640634390025, 0.3192307692307693, 0.33717944805438704, 0.351281973031851, 0.3615384615384616, 0.3756409865159255, 0.38589735764723554, 0.39743581918569704, 0.4064102172851562, 0.4115384615384615, 0.41538461538461535, 0.424358896108774, 0.4320512038010817, 0.44102560190054085, 0.4576923076923077, 0.4769230769230769, 0.49871790959284856, 0.5141025249774638, 0.5384615384615383, 0.5628204345703123, 0.583333294208233, 0.6076923076923075, 0.633333294208233, 0.6589742807241583, 0.6897435114933892]
f_5_2 = [0.650207020284955, 0.7489724337782808, 0.8477378472716066, 1.0452686742582582, 1.2427995012449098, 1.572018550916403, 1.8353919822048645, 2.2304536361781677, 2.6913578944532803, 3.1193423576181, 3.646092234276245, 4.172839096853167, 4.469135337333145, 3.8107002520713804, 3.1851849619199095, 2.559672685849661, 1.9670802048897063, 1.506172932533371, 1.0452686742582582, 0.650207020284955, 0.4197533841067873, 0.22222255712013572, 0.15637995281832606, 0.15637995281832606, 0.12345714362680993, 0.05761453932500027, 0.024691730133484135, 0.024691730133484135, -0.008228064976809613, 0.024691730133484135]

phase_6_2 = [0.2, 0.22051274226262024, 0.24230769230769245, 0.26282043457031257, 0.2871794480543871, 0.30641021728515627, 0.3217948326697716, 0.33717944805438704, 0.35384615384615387, 0.3653846153846153, 0.373076923076923, 0.38589735764723554, 0.39743581918569704, 0.40897428072415865, 0.4102563711313101, 0.4166665884164663, 0.423076923076923, 0.43333329420823313, 0.44358966533954325, 0.45897428072415863, 0.48076923076923056, 0.5025640634390023, 0.5243588961087738, 0.5499999999999998, 0.5743588961087739, 0.5987179095928484, 0.6243588961087739, 0.6551281268780047, 0.6807692307692307]
f_6_2 = [0.7818952429697967, 0.9135804515734163, 1.0452686742582582, 1.2098766920533937, 1.4732531374230773, 1.7366265687115385, 2, 2.4279844631648193, 2.7572035128363126, 3.2181077711114257, 3.646092234276245, 4.074073683359842, 4.502058146524661, 3.7119348385780544, 3.251030580302942, 2.6584380993429866, 2.0658456183830323, 1.4732531374230773, 0.9465032607649324, 0.5514416067916292, 0.35391077980497765, 0.22222255712013572, 0.15637995281832606, 0.15637995281832606, 0.12345714362680993, 0.09053734851651618, 0.024691730133484135, 0.05761453932500027, 0.05761453932500027]

phase_7_2 = [0.19487175574669482, 0.21410252497746396, 0.23589735764723563, 0.25897428072415873, 0.28205120380108184, 0.30128197303185106, 0.3153846153846154, 0.3333332942082333, 0.34615384615384615, 0.36025637113131015, 0.37179483266977154, 0.37948714036207926, 0.387179448054387, 0.39743581918569704, 0.40512812687800476, 0.4102563711313101, 0.41538461538461535, 0.4217948326697716, 0.42820504995492786, 0.43846153846153846, 0.45128197303185097, 0.46923076923076923, 0.4884615384615382, 0.5076923076923077, 0.5294871403620791, 0.5525640634390022, 0.5782050499549278, 0.6038461538461537, 0.6282050499549277, 0.6499999999999999]
f_7_2 = [0.8477378472716066, 1.012345865066742, 1.1111112785600679, 1.2427995012449098, 1.4732531374230773, 1.8353919822048645, 2.032922809191516, 2.3621418588630094, 2.6584380993429866, 3.0864195484265835, 3.5144040115914033, 3.975308269866516, 4.304527319538009, 4.534980955716177, 3.942388474756222, 3.3497959937962674, 2.7242807036447965, 2.1316882226848417, 1.539095741724887, 1.1111112785600679, 0.6831298294764709, 0.4197533841067873, 0.18930276200984197, 0.15637995281832606, 0.09053734851651618, 0.024691730133484135, -0.008228064976809613, 0.024691730133484135, -0.008228064976809613, 0.024691730133484135]

phase_8_2 = [0.19102560190054088, 0.21410252497746396, 0.23589735764723563, 0.2576923076923078, 0.2833332942082333, 0.30512812687800484, 0.32307692307692304, 0.3384615384615385, 0.35512812687800477, 0.3666665884164664, 0.3756409865159255, 0.38589735764723554, 0.3987179095928486, 0.4064102172851562, 0.40897428072415865, 0.4115384615384615, 0.4141025249774639, 0.4192307692307692, 0.424358896108774, 0.4358973576472356, 0.4474358191856971, 0.46923076923076923, 0.49358966533954307, 0.5141025249774638, 0.5384615384615383, 0.5653846153846152, 0.5910256019005407, 0.6153846153846152, 0.643589665339543, 0.6717948326697716]
f_8_2 = [0.9465032607649324, 1.1111112785600679, 1.1769568969431, 1.4074075190400452, 1.6707839644097289, 1.9341573956981903, 2.296296240479977, 2.5925924809599548, 3.020576944124774, 3.4814812023998867, 3.942388474756222, 4.370369923839819, 4.666666164319796, 4.172839096853167, 3.646092234276245, 3.2181077711114257, 2.5925924809599548, 2, 1.4403303282315614, 1.012345865066742, 0.6172842110934389, 0.28806817550316777, 0.09053734851651618, 0.05761453932500027, 0.024691730133484135, 0.09053734851651618, 0.05761453932500027, 0.024691730133484135, 0.024691730133484135, 0.024691730133484135]

phases = [phase_1_2, phase_2_2, phase_3_2, phase_4_2, phase_5_2, phase_6_2, phase_7_2, phase_8_2]
fs =  [f_1_2, f_2_2, f_3_2, f_4_2, f_5_2, f_6_2, f_7_2, f_8_2]


m = m_e*c**2
m = m.to('keV')
m_unitless = m/(u.keV)
plt.style.use(["science","no-latex"])
plt.rcParams["figure.figsize"] = (7,7)
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5

r_0 = (e.value)**2/(4*np.pi*eps0.value*m_e.value*c.value**2)*u.m


T     = 33*10**(-3)*u.s # període de Crab (s)
Omega = 2*np.pi/T       # velocitat angular
Rl    = c*T/(2*np.pi)   # radi de llum

delta_R = 0.05  # magnitud de cada pas per a l'integral (en unitats de Rl)

R  = np.arange(1, 10, delta_R)*Rl  # array de distancies respecte l'estrella de neutrons
a  = np.arcsin(Rl/R)             # array d'angles que s'obté a partir de la simplificació per a R>>Rl
w  = 1.-np.cos(a)                # pesos per a l'eficiencia de la dispersió IC

#Si nomes vull graficar una alpha necessito posarla com una llista o array
#alpha = np.array([1,3,10])
alpha = 1
#gamma_w = np.arange(55*10**4, 1*10**6, 0.5*10**4)      # factors gamma final que utilitza el model
gamma_w = 5.5*10**5

#Vigilar, perquè si només vull graficar un conjunt de Ri i Rf necessito posarles com a llistes
#Ri    = np.array([1, 20, 25])*Rl        # radi on es comencen a accelerar els electronsque anirà a les funcions de gamma i del moment 
#Rf    = np.array([30, 50, 70])*Rl         # radi final que diu el mateix model

Ri = 1*Rl#Radi inicial on els positrons comencen a accelerar
Rf = 2*Rl#Radi final on els positrons deixen d'accelerar

gamma_0 = 300#Factor gamma inicials dels positrons

E0 = 1*u.keV

E_fotoi2 = [0.5, 1.3, 3.0, 7.0, 12.0, 27.0, 65.0, 170.0]*(u.keV)#Energies en les quals s'ha dividit els pulse profiles
#Delta_E_fotoi = [0.2, 1.4, 2.0, 6.0, 4.0, 26.0, 50.0, 160.0]*(u.keV)


#E_fotoi = np.logspace(-1,2, steps)*(u.keV)
E_fotoi1 = np.logspace(-5,2,40)*(u.keV)#Energies inicials dels fotons

logE = np.log(E_fotoi1/E0)

E_fotoi = (E_fotoi1[1:] + E_fotoi1[:-1])/2#Bins d'energies dels fotons

Delta_E = E_fotoi1[1:] - E_fotoi1[:-1]#Espaiat d'energies dels fotons

Delta_log = np.log((E_fotoi1[1:])/E0) -np.log((E_fotoi1[:-1])/E0)#Espaiat logaritmic

steps = 40 #Numero de energia final dels fotons que tindrè
#El logspace em genera un conjunt de 10**n sent n valors entre 1 i 6 espaiats
#pels steps que li doni

E_fotof1 = np.logspace(6,9, steps)*(u.keV)#Energies finals dels fotons

E_fotof = (E_fotof1[1:]+E_fotof1[:-1])/2#Bins de les energies finals dels fotons

Delta_Ef = E_fotof1[1:]-E_fotof1[:-1]#Espaiat d'aquesta energia
    
E_fotoi_3d = add_dim_e_fi(E_fotof, E_fotoi, R)#Energia inicials dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f

E_fotof_3d = add_dim_e_ff(E_fotof, E_fotoi, R)#Energia final dels fotons en 3 Dimensions: la de R, la de E_i, la de E_f

Gamma = gammaw(R.value, Ri.value, Rf.value,
               gamma_0, gamma_w, alpha)       # array de factors gamma per a cada distància, segons el model de vent
M_i   = M(R.value, Ri.value, Rf.value,
               gamma_w, alpha, Omega)                # array de moments angulars que s'emporten els electrons

Gamma_2d = add_dimension_R(Gamma, E_fotoi)#Factor gamma dels positrons en 2 dimensions

Gamma_3d = add_dimension_R(Gamma_2d, E_fotof)#Factor gamma dels positrons en 3 dimensions

beta = beta_f(Gamma_3d)#Valor de la beta dels positrons en 3 dimensions

beta1d = beta_f(Gamma)#Valor de la beta dels positrons en 1 dimensio (la de R, que es de l'unic parametre que depen)

theta_1d = np.arcsin(M_i*c/(Gamma*m*R))        #Array d'angles de la colisio entre electrons i fotons

theta_2d = add_dimension_R(theta_1d, E_fotoi) #Array d'angles de la colisio de positrons i fotons en 2 dimensions

thetau = add_dimension_R(theta_2d, E_fotof) #Array d'angles de la colisio de positrons i fotons en 3 dimensions

theta = thetau*u.rad #Arary en 3 dimensions i amb unitats

thetaf2 = np.arccos(1/beta - E_fotoi_3d*(1/beta-np.cos(theta))/E_fotof_3d)#Primera approximacio del que val el valor final de l'angle de dispersió del foto

E_fotof_max = E_fotoi_3d[0]*m*Gamma_3d[0]*(1-beta[0]*np.cos(theta[0]))/(m*Gamma_3d[0]*(1-beta[0]) + E_fotoi_3d[0]*(1-np.cos(theta[0])))#Energia maxima que els fotons poden assolir, primera aproximacio
E_fotof_min = E_fotoi_3d[0]*m*Gamma_3d[0]*(1+beta[0]*np.cos(theta[0]))/(m*Gamma_3d[0]*(1+beta[0]) + E_fotoi_3d[0]*(1+np.cos(theta[0])))#Energia minima que els fotons poden assolir, primera aproximacio


#A partir d'aqui comença un calcul numeric del valor de l'angle de dispersió dels fotons

theta_f_e = []

for j in range(len(E_fotof)):
    print(j)
    auxiliar = []
    for k in range(len(E_fotoi)):
        auxiliar2 = []
        for i in range(len(R)):
            if (E_fotof_max[k][i] > E_fotof_3d[j][k][i] > E_fotof_min[k][i]):
                theta_fexacte1 = solver(thetaf2[j][k][i],theta[j][k][i], Gamma_3d[j][k][i], beta[j][k][i], E_fotof_3d[j][k][i], E_fotoi_3d[j][k][i])
                theta_fexacte2 = solver(theta_fexacte1,theta[j][k][i], Gamma_3d[j][k][i], beta[j][k][i], E_fotof_3d[j][k][i], E_fotoi_3d[j][k][i])                
                theta_fexacte3 = solver(theta_fexacte2,theta[j][k][i], Gamma_3d[j][k][i], beta[j][k][i], E_fotof_3d[j][k][i], E_fotoi_3d[j][k][i])                
                auxiliar2.append(float(theta_fexacte3))
            else:
                auxiliar2.append(10000)
        auxiliar.append(np.array(auxiliar2))
    theta_f_e.append(np.array(auxiliar))
theta_f_e = np.array(theta_f_e)*u.rad#Obtinc un valor 

comprovacions = equation_solve(theta_f_e, theta, Gamma_3d, beta, E_fotof_3d, E_fotoi_3d, m)

contador = 0
contador2 = 0

#Aquí comprovo com de precís és el resultat
for j in range(len(E_fotof)):
    for k in range(len(E_fotoi)):
        for i in range(len(R)):
            if (E_fotof_max[k][i] > E_fotof_3d[j][k][i] > E_fotof_min[k][i]):
                contador += 1 

                if abs(comprovacions[j][k][i]/E_fotoi[k]**2) < 0.1 :
                    contador2 += 1
            else:
                comprovacions[j][k][i] = 99999999*u.keV*u.keV
            
print(contador, contador2)
#Contador em dona el numero de bins que tinc entre el range de Energia maxima i Energia minima del foto
#Mentres que contador2 em dona el numero de bins que tinc amb un error menor al 10%
#AQUEST CALCUL S'HAURIA DE MILLORAR JA QUE L'ENERGIA FINAL DELS FOTONS CONVERGEIX POC

#Aquí ha acabt el calcul de l'angle de dispersió dels fotons

#Obtinc la secció eficaç segons el Peskin,
#i li trec les unitats per poder-ho visualitzar millor en el compilador
cross_section_exact = exact_xsection(theta, theta_f_e, beta, Gamma_3d, E_fotoi_3d, E_fotof_3d)
cross_sec_exact_unitless = cross_section_exact.value/((137)**2)

#Declaro les variables de l'espectre que necessito per una energia major a 0.2 keV
#Aquests valors s'obtenen del codi: "phase_a_spectra"
E0 = 1*u.keV
K = 0.488
a = 0.618
b = 0.133


#Declaro les variables de l'espectre que necessito per una energia menor a 0.2 keV
#Aquests valors també s'obtenen del codi: "phase_a_spectra"
K1 = 0.524
a1 = 0.452
b1 = 0.057
#Declaro un primer espectre que em donara una aproximacio
spectra1 = K*(E_fotoi_3d/E0)**(-a-1-b*np.log10(E_fotoi_3d/E0))/E0

#Aquest segon espectre es més precís
spectra = []
for j in range(len(E_fotof)):
    auxiliar1 = []
    for k in range(len(E_fotoi)):
        auxiliar2 = []
        for i in range(len(R)):
            if E_fotoi[k] > 0.2*u.keV:
                auxiliar = K*(E_fotoi[k]/E0)**(-a-1-b*np.log10(E_fotoi[k]/E0))/E0.value
            elif E_fotoi[k] < 0.2*u.keV:
                auxiliar = K1*(E_fotoi[k]/E0)**(-a1-1-b1*np.log10(E_fotoi[k]/E0))/E0.value
            auxiliar2.append(auxiliar)
        auxiliar1.append(np.array(auxiliar2))
    spectra.append(np.array(auxiliar1))
spectra = np.array(spectra)*(1/E0).unit

#Faig les multiplicacions adients
powers = (1-np.cos(theta))*cross_section_exact*spectra


#Calculo la primera integral
first_integral = []

for j in range(len(E_fotof)):
    auxiliar2 = []
    for i in range(len(R)):
        auxiliar = 0
        for k in range(len(E_fotoi)):
            if (E_fotof_max[k][i] > E_fotof_3d[j][k][i] > E_fotof_min[k][i]):
                auxiliar += powers[j][k][i]*Delta_log[k]*E_fotoi[k]
        auxiliar2.append(np.array(auxiliar))
    first_integral.append(np.array(auxiliar2))
    
#Li poso les unitats que toca i ho deixo en forma d'array
first_integral = np.array(first_integral)*powers.unit

#Trec les unitats per si ho he de mirar a la consola, fer-ho més fàcil
first_integralu = first_integral.value

#Inicialtzo la variable de la fase
delta_phase = 0.002
phase = np.arange(0.2,0.5,delta_phase)
phase_3d = add_dim_e_ff(phase, E_fotof, R).value#El poso en les dimensions que em conve
time = phase_3d*T


fi_time = add_dimension_R(first_integral, phase)*first_integral.unit#De nou, ho poso en unes altres dimensions

#Canvio les variables segons les dimensions que em conve
R_2d = add_dimension_R(R, E_fotof)*R.unit
R_3d = add_dimension_R(R_2d, phase)*R.unit

theta_2dt =add_dimension_R(theta_1d,E_fotof)
theta_t = add_dimension_R(theta_2dt, phase)*u.rad


#Busco els parametres dels pulse profiles
X0s, sigmes1, sigmes2, As, Cs, pcov = adjust_as_lor(phases, fs)


#funció que em treu un valor i el seu error directament per passar-ho a latex
def display_sigfig(x, xerr, sigfigs=2) -> str:
    '''
    Suppose we want to show 2 significant figures. Implicitly we want to show 3 bits of information:
    - The order of magnitude
    - Significant digit #1
    - Significant digit #2
    '''

    if sigfigs < 1:
        raise Exception('Cannot have fewer than 1 significant figures. ({} given)'.format(sigfigs))

    order_of_magnitude = np.floor(np.log10(np.abs(xerr)))

    # Because we get one sigfig for free, to the left of the decimal
    decimals = (sigfigs - 1)

    xerr /= np.power(10, order_of_magnitude)
    xerr = np.round(xerr, decimals)
    xerr *= np.power(10, order_of_magnitude)

    # Subtract from decimals the sigfigs we get from the order of magnitude
    decimals -= order_of_magnitude
    # But we can't have a negative number of decimals
    decimals = int(max(0, decimals))

    return '{:.{dec}f}'.format(x, dec=decimals)+'$\pm${:.{dec}f}'.format(xerr, dec=decimals)

#Bucle per passar coses a Latex
for i in range(len(X0s)):
    print("GRAFICA"+str(i+1))
    cov = pcov[i]
    print("$x_0$ = ", display_sigfig(X0s[i], np.sqrt(cov[0][0]))+",")
    print("$\sigma_1$ = ", display_sigfig(sigmes1[i], np.sqrt(cov[1][1]))+",")
    print("$\sigma_2$ = ", display_sigfig(sigmes2[i], np.sqrt(cov[2][2]))+",")
    print("$A = $", display_sigfig(As[i], np.sqrt(cov[3][3]))+",")
    print("$C = $", display_sigfig(Cs[i], np.sqrt(cov[4][4]))+".")


#És el temps que correspon al Pulse Profile
time_d = time - R_3d*(1-np.cos(theta_t))/c + theta_t.value/Omega

#el Pulse profile com a tal, amb una primer aproximació de que no depenen de l'energia
fun = funct_f(time_d, X0s[0], sigmes1[0], sigmes2[0], As[0], Cs[0])

#Seguent cosa que haure d'integrar
power2 = fun*fi_time*Rl**2/R_3d**2

#segona integral en una primera aproximacio
second_integral = np.sum(power2, axis = 2)*delta_R


#Ho trasposo i poso les unitats que toca
si_tr = np.transpose(second_integral)

Lsd = 4.6e31*u.J/u.s

dNdV = Lsd/(4*np.pi*c**3*gamma_w*m_e)

#carrega electro en unitats naturals alpha = e^2/(4pi) \approx 1/137
e2 = 4*np.pi/137

kevs_m = 8.07e8/(u.m*u.keV)

si_tr_units = si_tr*dNdV*e2/(kevs_m**2*Rl)

#Ho grafico
for i in range(len(E_fotof)):
    plt.plot(phase, si_tr_units[i], label = "approx"+str(i))
    ids = pds.Series(si_tr_units[i]).idxmax()
    print(phase[ids])
    if i%2 == 0:
        plt.legend()
        plt.show()
plt.xlabel("Phase(rad)")
plt.ylabel(r"$\frac{dN_{\gamma}}{dE_{\gamma}dSdt}$", fontsize = 15)
plt.legend()
plt.show()   
        
        
#Declaro una funcio pel temps dels Pulse Profiles
def Time_d(time, R, theta):
    return time - R*(1-np.cos(theta))/c + theta.value/Omega

#A partir d'aquí re faig les dues primeres integrals però considerant que els pulse profiles depenen de l'energia


#Els pulse profiles mes precisos en funcio de l'energia
Funct = []
for l in range(len(phase)):
    print(l)
    auxiliar3 = []
    for j in range(len(E_fotof)):
        auxiliar2 = []
        for i in range(len(E_fotoi)):
            auxiliar = []
            for k in range(len(R)):
                temps_pt = Time_d(phase[l]*T, R[k], theta_1d[k])
                for ñ in range(len(E_fotoi2)):
                    if E_fotoi[i]<E_fotoi2[ñ]:
                        suport = funct_f(temps_pt, X0s[ñ], sigmes1[ñ], sigmes2[ñ], As[ñ], Cs[ñ])
                        break
                auxiliar.append(suport)
            auxiliar2.append(np.array(auxiliar))
        auxiliar3.append(np.array(auxiliar2))
    Funct.append(np.array(auxiliar3))
Funct = np.array(Funct)


#L'angle d'interacció entre positrons i fotons amb les dimensions que vull
Theta2D = add_dimension_R(theta_1d, E_fotoi)*u.rad
Theta3D = add_dimension_R(Theta2D, E_fotof)*u.rad
Theta4D = add_dimension_R(Theta3D, phase)*u.rad

#L'espectre amb les dimensions que vull
xsec4D = add_dimension_R(cross_section_exact, phase)*cross_section_exact.unit
spectra4D = add_dimension_R(spectra, phase)*spectra.unit

#Primer producte
first = Funct*(1-np.cos(Theta4D))*xsec4D*spectra4D

#Calculo la primera integral
first_int = []
for l in range(len(phase)):
    print(l)
    auxiliar3 = []
    for j in range(len(E_fotof)):
        auxiliar2 = []
        for i in range(len(R)):
            auxiliar = 0
            for k in range(len(E_fotoi)):
                if (E_fotof_max[k][i] > E_fotof_3d[j][k][i] > E_fotof_min[k][i]):
                    auxiliar += first[l][j][k][i]*Delta_log[k]*E_fotoi[k]
            auxiliar2.append(np.array(auxiliar))
        auxiliar3.append(np.array(auxiliar2))
    first_int.append(np.array(auxiliar3))
        
#Amb unitats i sense, per segons em convingui
first_int = np.array(first_int)*first.unit*E_fotoi.unit
first_intu = first_int.value

#tono a integrar, tranposo i poso les unitats que toquen
second_int = np.sum(first_int*Rl**2/R_3d**2, axis = 2)*delta_R
secondi_tr = np.transpose(second_int)
secitr_units = secondi_tr*dNdV*e2/(kevs_m**2*Rl)

#Això em servira per compara els SEDs
"""
x = [100000000, 345511.48368313984, 492388.6470441612, 766683.7914064317, 1045274.7822721634, 1778278.058325248, 3025302.9216365786, 5146809.142031739, 11420677.265467953, 14896233.027498951, 20309180.95905017, 22189798.33029695, 26489698.039047025, 28942627.400982425, 30253059.877816208, 33054474.176812675, 34551078.33323037, 34551078.33323037, 34551078.33323037, 34551078.33323037, 34551078.33323037, 33054474.176812675, 253423.29584824733, 162756.47818676993, 119377.94337766837, 73347.36026464758, 51468.19574620963]
y = [-0.000006614946304329913, 6.409472670790406e-7, 0.0000014296313509278393, 0.000002218315434776638, 0.000002849262701855678, 0.0000036379467857044774, 0.000004268894052783516, 0.000004794685046597926, 0.000004794685046597926, 0.000004163737779518886, 0.0000032173168789003275, 0.0000022708959782817682, 0.0000012718945341580796, 3.780541770446496e-7, -5.157909938144078e-7, -0.000001462211894432968, -0.000002356052251546398, -0.0000030921557918900665, -0.000004038576692508627, -0.000004827260776357425, -0.000005668525403711354, -0.0000063520532142955235, -2.0031736027488833e-7, -9.890014441236888e-7, -0.0000017251049844673573, -0.0000026189453415807873, -0.000003670527329209606]
"""

x2 = [91.38064031403975, 143.41124711707917, 325.42897873361665, 835.0421424203932, 2973.7908453421987, 7630.668566358315, 12476.298173503372, 18039.65018323842, 24031.663174227924, 32013.850754225157, 44431.17228141686, 56813.21278921525, 66930.28098621606, 69729.53796180403, 85582.35356986578, 100822.89271839082, 118777.47306640366, 134312.01677685397]
y2 = [0.000007545003984212426, 0.00001573289719840635, 0.00003530793742217019, 0.00008528077408914708, 0.00012775659273060922, 0.00007362436798086763, 0.00003530793742217019, 0.00001573289719840635, 0.000006513732499309725, 0.0000022442107754792045, 8.02149475022789e-7, 2.3859397545156373e-7, 9.521881042259686e-8, 3.800021287685452e-8, 8.424246985134803e-9, 2.5995202050740402e-9, 8.633158520511537e-10, 2.5678750360758687e-10]

x = [108.57105939203441, 218.41711189283046, 439.3975286702991, 610.5401639918168, 1228.2501438265476, 2371.3741520050517, 3433.3209414598523, 9210.557634784978, 12798.061370079453, 22758.50332098324, 34333.31281068948, 71968.81499932126, 78137.24604821566, 117877.28186263177, 209618.50651936233, 329501.72436413093, 477058.91099048086, 636169.1203944023, 1085715.4984275685, 2096183.4870063816, 2912645.029041295]
y = [0.0002137189267770382, 0.00022159005881784106, 0.00022159005881784106, 0.00022159005881784106, 0.00018493367035466994, 0.00015434114063286706, 0.00011146035613731978, 0.00005215277739701573, 0.000024402507622277154, 0.000012274542945270864, 0.00001319529752225786, 0.000009190742948669466, 0.0000016196597796793656, 0.0000012127458765828707, 5.091038668057519e-7, 2.3821187691854446e-7, 2.061273863385585e-7, 1.5434114063286723e-7, 1.3847178134144655e-7, 1.335531054735925e-7, 1.8493336418610452e-7]

ns = np.array(y)
xns2 = np.array(x)

cr = np.array(x2)
cry = np.array(y2)

spec = np.sum(secitr_units, axis = 1)*delta_phase

SED = (spec*E_fotof**2).to('MeV')

plt.plot(E_fotof.to('MeV') , SED , label = "Theoretical SED")
plt.plot(E_fotof.to('MeV') , SED*0.015 , label = r"Theoretical SED, $\eta$")
plt.plot(xns2, ns,'.', label = " Data of the SED")
plt.plot(cr, cry, label ="Model of CR")
plt.ylim((1e-10, 1e-2))
plt.xlim((1e1, 1e7))
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$E_{\gamma}^2 \dfrac{dN^{(0)}}{dSdtd\epsilon}$ (MeVcm$^{-2}$s$^{-1}$)", fontsize = 20)
plt.xlabel(r"$E_{\gamma}$(MeV)", fontsize = 20)
plt.legend(fontsize = 20)
plt.savefig("SED_theoretical")
plt.show()
#Ara ja he comparat els SEDs

#M'ho imprimeixo per mi
print(Ri/Rl, Rf/Rl, R[0]/Rl, R[-1]/Rl)

#Això servira per la normalitació
phase_averaged = np.sum(secitr_units, axis = 1)*delta_phase

#Torno a plotejar, pero coses diferents, ara plotjo els Pulse profiles en funcio de l'energia
maxs = []
max_value = []
Es = []

adjusts = []
inc_adjusts = []

for i in range(9):
    
    x_initial = 0.3
    gamma_initial1 = 0.3
    gamma_initial2 = 0.1
    A_initial = 1
    C_initial = 1
    
    popt, pcov = curve_fit(asym_lorentz_C, phase, secitr_units[i], [x_initial, gamma_initial1, gamma_initial2, A_initial, C_initial], maxfev = 800000)
    
    adjusts.append(popt)
    inc_adjusts.append(pcov)
    
    plt.plot(phase, secitr_units[i], label = "excate"+str(i))
    
    max_value.append(np.max(asym_lorentz_C(phase,  *popt)))
    
    plt.plot(phase, asym_lorentz_C(phase, *popt))
    
    ids = pds.Series(secitr_units[i]).idxmax()
    maxs.append(phase[ids])
    Es.append(E_fotof[i]/(E_fotof[i].unit*1e6))#En GeV
    plt.axvline(phase[ids])
    print(phase[ids], E_fotof[i].to('GeV'),i)

    if (i+1)%3 == 0:
        plt.xlabel("Phase(rad)")
        plt.ylabel(r"$\frac{dN_{\gamma}}{dE_{\gamma}dSdt}$", fontsize = 15)
        plt.legend()
        plt.show()
plt.xlabel("Phase(rad)")
plt.ylabel(r"$\frac{dN_{\gamma}}{dE_{\gamma}dSdt}$", fontsize = 15)
plt.legend()
plt.show()    

#Ho torno a transposar per conveniencia
seci_units = np.transpose(secitr_units)

#Calculo el flux per compararho despres
flux = []
for j in range(len(phase)):
    aux = 0
    for i in range(len(E_fotof)):
        aux += seci_units[j][i]*Delta_Ef[i]
    flux.append(aux/aux.unit)
flux = np.array(flux)*aux.unit

phase_left = []
phase_right = []

for j in range(len(Es)):
    out = adjusts[j]
    maxim = max_value[j]
    left = out[0]-np.sqrt(out[-2]*out[1]/(np.pi*(maxim/2-out[-1]))-out[1]**2)
    right = out[0]+np.sqrt(2*out[2]**2*(out[-2]+np.pi*out[1]*out[-1])/(np.pi*out[1]*maxim)-out[2]**2)
    phase_left.append(left)
    phase_right.append(right)
    
    

#Grafico la posiscio dels peaks per comparar amb resultats
plt.plot(maxs, Es, 'x', color = "k")
plt.plot(phase_left, Es, '.', color = "g")
plt.plot(phase_right, Es, '.', color = "g")
plt.ylabel(r'$E_{\gamma}$ (GeV)', fontsize = 20)
plt.xlabel('Phase', fontsize = 20)
plt.yscale("log")
plt.savefig("Theoretical_Displacement_peak")
plt.show()

#A partir d'aqui ho faig per veure quina és l'energia inicial més baixa dels fotons que em permet
#tenir una energia finals dels fotons de 1 GeV
E_i_0 = 0.02*1e-3*u.keV

print(E_i_0*1000/E_i_0.unit, "eV")

def angle_max(tf, G, B, ti, Ei):
    eq = m*G*B*np.sin(tf)+Ei*np.sin(tf+ti)
    return eq

def second_der(tf, G, B, ti, Ei):
    eq = -m*G*B*np.cos(tf)-Ei*np.cos(tf+ti)
    return eq

#Aquí trobo quin es l'angle maxim
tmax = fsolve(lambda x: angle_max(x*u.rad, Gamma, beta1d, theta_1d, E_i_0), -theta_1d, xtol = 1e-8, maxfev = 3000)

tmax_com = second_der(tmax*u.rad, Gamma, beta1d, theta_1d, E_i_0)

#Grafico l'energia final dels fotons per aquest angle maxim
E_fotof_max2 = E_i_0*m*Gamma*(1-beta1d*np.cos(theta_1d))/(m*Gamma*(1-beta1d*np.cos(tmax*u.rad)) + E_i_0*(1-np.cos(theta_1d+tmax*u.rad)))

#plt.plot(R[R <= 2*Rl]/Rl, E_fotof_max2[R <= 2*Rl])
plt.plot(R/Rl, E_fotof_max2)
plt.xlabel(r"R/R$_{L}$", fontsize = 20)
plt.ylabel(r"$E_{\gamma}^{max}$(keV)", fontsize = 20)
plt.savefig("Energia_max_0_02eV")
plt.show()
   

#Resultats dels valors experimentals
A = 6.54617e+00
x0 = 4.05391e-01
sigma_2 = 5.86374e-03
sigma_1 = (2.11878e+00 + 1)*sigma_2
offset =   1.75916e+03 

#Aquí simplement garfico els resultats i els comparo amb els experimentals
result = as_lorentz(phase, x0, sigma_1, sigma_2, 1)

norm = np.sum(flux)*delta_phase

norm2 = np.sum(result)*delta_phase

max_f_1 = np.max(flux)
max_f_2 = np.max(result)

plt.plot(phase, flux/norm, label = "theoretical result")
plt.plot(phase, result/norm2, label = "experimental result")
plt.xlabel("phase", fontsize = 20)
plt.ylabel("Normalized flux", fontsize = 20)
plt.legend(fontsize = 20)
plt.savefig("Flux_obtained")
plt.show()

x1 = [0.14364259957498768, 0.1876288679254484, 0.2206185534575859, 0.24536081760668915, 0.2632302096289865, 0.27835056606165837, 0.2934707966486671, 0.30034360585264125, 0.31821306079777023, 0.32920958069326167, 0.33470790356383906, 0.3429553249468734, 0.36082477989200235, 0.36769758909597655, 0.38006872117052815, 0.3841924318620454, 0.3910652410660197, 0.3910652410660197, 0.3979381761156571, 0.40756008383208847, 0.4158075052151229, 0.44604809223480346, 0.4817868762793981, 0.42955324946873463]
y1 = [0.10421694023096655, 0.12409639949634975, 0.15843379773357388, 0.20542168575026074, 0.2487951717523466, 0.2993976272403529, 0.34457839697809856, 0.39879525448070585, 0.5036144847429602, 0.5596385432528681, 0.6084337150052147, 0.6698795419937424, 0.6481927989926995, 0.5975903435046932, 0.4981927989926995, 0.43674705473253095, 0.38975900125912566, 0.33734942749217806, 0.28493985372523045, 0.2325301145015644, 0.17108445296975508, 0.07530122774400316, 0.03915671122783776, 0.12048199748174875]

x2 = [0.16151205452011663, 0.21237113207455152, 0.252233689733495, 0.28109966457411545, 0.30859102723567566, 0.32783509435986474, 0.34707903563839065, 0.35945016771294214, 0.36494849058351947, 0.37869410899146805, 0.3814433333495883, 0.3883161425535626, 0.40618559749869154, 0.4089346960111486, 0.4116837945236057, 0.4144330188817259, 0.417182117394183, 0.41993121590664007, 0.42542953877721745, 0.42955324946873463, 0.43780067085176905, 0.44467348005574336, 0.46116832282181214, 0.47766316558788097]
y2 = [0.08975916671584408, 0.09879525448070582, 0.13674705473253096, 0.1783132569989571, 0.23975908398748486, 0.30843371500521466, 0.3753012277440032, 0.4457831424973927, 0.5072289694859204, 0.5596385432528681, 0.6174698854984357, 0.6734939440083435, 0.6463855152570398, 0.57771088423931, 0.5144578562434817, 0.46746988549843566, 0.39337356873044516, 0.33373502547757705, 0.29578322522575196, 0.24337348600208586, 0.2090362532215802, 0.14939770996871216, 0.10783134224556755, 0.06807242371480116]

x3 = [0.12852236898797897, 0.1560137316495393, 0.1917525786169656, 0.2206185534575859, 0.24261165617140049, 0.25773201260407236, 0.27972505239505535, 0.30034360585264125, 0.314089350106253, 0.3429553249468734, 0.35670106920048517, 0.3718212997874938, 0.3828178196829852, 0.3979381761156571, 0.4006872746281142, 0.4048109853196314, 0.40756008383208847, 0.40756008383208847, 0.4130584067026658, 0.4158075052151229, 0.41993121590664007, 0.4213058280857002, 0.42680415095627755, 0.43505157233931196, 0.4419243815432863, 0.45704461213029485, 0.48453610063751834]
y3 = [0.12409639949634975, 0.1439760242184514, 0.17289157124869636, 0.2108433715005215, 0.25421685750260736, 0.2939759414900922, 0.34457839697809856, 0.41144574426016856, 0.4873494274921781, 0.5849397709968713, 0.6554216857502608, 0.7277108015109508, 0.8180722582580828, 0.7710842875130366, 0.7132530279958282, 0.6391566284994785, 0.5939759414900923, 0.5415662849947853, 0.48915662849947855, 0.43674705473253095, 0.38795188298018435, 0.33373502547757705, 0.27228919848904926, 0.20180728373565973, 0.1457831424973927, 0.08614459924452463, 0.03915671122783776]

A_1_x = np.array(x1)
A_1_flux = np.array(y1)
max_A1 = np.max(A_1_flux)

A_2_x = np.array(x2)
A_2_flux = np.array(y2)
max_A2 = np.max(A_2_flux)

A_3_x = np.array(x3)
A_3_flux = np.array(y3)
max_A3 = np.max(A_3_flux)

#Això és sense experimentals
plt.plot(phase, flux/max_f_1, color = "c", label = "our model")
plt.plot(A_1_x, A_1_flux/max_A1, "--", color = "k", label = r"$\gamma$-ray $R_w = R_L$")
plt.plot(A_2_x, A_2_flux/max_A2, color = "b", label = r"$\gamma$-ray $R_w = 30R_L$")
plt.plot(A_3_x, A_3_flux/max_A3, color = "r", label = r"$\gamma$-ray anisot. wind")
plt.ylim((0,1.2))
plt.xlim((0.1,0.5))
plt.xlabel("phase", fontsize = 20)
plt.ylabel("Normalized flux", fontsize = 20)
plt.legend(fontsize = 17)
plt.savefig("Flux_theory")
plt.show()

"""
the_f = np.arange(0,2*np.pi,0.01)

g = 300
algo = m*g*(1-np.sqrt(1-1/g**2)*np.cos(the_f*u.rad))+E_i_0*(1-np.cos(the_f*u.rad+theta_1d[0]))

plt.plot(the_f, algo)
"""