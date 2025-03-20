import numpy as np
import math
from guezzi import *
import matplotlib.pyplot as plt

"""configurazione 1 (voltometro in parallelo col diodo)"""
def expon(x, a, b):
    return a*(np.exp(38.6/b * x)-1)


I = {        0.03:0.01, 0.05:0.01, 0.20:0.02, 3.41:0.01, 76.88:0.1, 910:2, 10500:400, 109000:1000, 490000:1000, 253:3, 2764:2, 4653:2, 32900:100, 64000:500, 253000:1000, 214000:1000, 1145:10, 4.59:0.01, 1661:1, 18100:100, 55100:1000} #micro ampere
I_ = np.array(list(I.keys()))
V = np.array([0.100   , 0.212    , 0.309    , 0.396    , 0.501    , 0.600, 0.700    , 0.796      , 0.880      , 0.559, 0.655 , 0.675 , 0.746    , 0.773    , 0.836      , 0.823      , 0.624  , 0.421    , 0.638 , 0.724    , 0.762])
#R_eq = np.sort(V/I_)
I_1 = create_dataset(I, magnitude=-6)
V_1 = create_dataset(V, 0.001)
#(a,b), (da, db), chi, dof = perform_fit(V_1, I_1, expon, p0=[10**(-10), 1],  chi_square=True)
#print(a, b, da, db)

"""configurazione 2 (voltometro in parallelo col generatore)"""
V = [0.0156    , 0.018      , 0.105     , 0.207      , 0.258     , 0.305     , 0.360     , 0.401     , 0.418     , 0.447     , 0.473    , 0.509    , 0.529    , 0.550     , 0.570     , 0.616   , 0.626 , 0.652 , 0.713 , 0.749  , 0.805]
I = {0.02: 0.01, 0.021: 0.01, 0.03: 0.01, 0.031: 0.01, 0.05: 0.01, 0.12: 0.01, 0.65: 0.01, 2.37: 0.02, 3.88: 0.02, 9.64: 0.02, 19.5: 0.1, 49.3: 0.1, 78.0: 0.1, 120.2: 0.2, 176.7: 0.2, 358.6: 1, 410: 1, 550: 2, 951: 2, 1214: 2, 1658: 2}
I_ = np.array(list(I.keys()))
R_eq = np.sort(V/I_)
I_2 = create_dataset(I, magnitude=-6)
V_2 = create_dataset(V, 0.001)
#(a,b), (da, db), chi, dof = perform_fit(V, I, expon, p0=[10**(-10), 1],  chi_square=True)
#print(a, b, da, db)

create_best_fit_line(V_1, I_1, V_2, I_2, func=expon, p0=[[10**(-10), 1],[10**(-10), 1]], # Guess iniziali NON obbligatorio
                    ylabel="Correnti (A)",
                    xlabel="Tensioni (V)", # Titoli condivisi
                    title="Diodo", 
                    label_fit=["configurazione 1", "configurazione 2"], # Nome delle linee
                    together=False) # Plotta separatamente i grafici

V2 = np.array([0.029, 0.047, 0.07, 0.154, 0.189, 0.223, 0.274, 0.316, 0.380, 0.420, 0.474, 0.501, 0.534, 0.59, 0.630, 0.680, 0.746, 0.787, 0.863, 0.992])
I2 = np.array([0.44e-6, 0.44e-6, 0.44e-6, 0.46e-6, 0.46e-6, 0.47e-6, 0.49e-6, 0.56e-6, 1.35e-6, 3.75e-6, 17.47e-6, 35.02e-6, 48.25e-6, 79.03e-6, 239.75e-6,
               435.20e-6, 0.7680e-3, 1.2025e-3, 2.154e-3, 3.2823e-3])
I = create_dataset(I2, 0.00001)
V = create_dataset(V2, 0.001)
create_best_fit_line(V_2, I_2, V, I, func=expon, p0=[[10**(-10), 1],[10**(-10), 1]], # Guess iniziali NON obbligatorio
                    ylabel="Correnti (A)",
                    xlabel="Tensioni (V)", # Titoli condivisi
                    title="Diodo", 
                    label_fit=["configurazione nostrana", "configurazione estranea"], # Nome delle linee
                    together=True) # Plotta separatamente i grafici
