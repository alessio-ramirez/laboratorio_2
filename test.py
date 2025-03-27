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
R_eq = np.array(V/I_)
I_err = np.array(list(I.values()))
I_1 = create_dataset(I, magnitude=-6)
V_1 = create_dataset(V, 0.001)
#latex_table("I", I_1, "V", V_1, "R_eq", create_dataset(R_eq), orientation='v')

#latex_table("I", I_1, "V", V_1, "R_eq", create_dataset(R_eq), orientation='v')
#(a,b), (da, db), chi, dof = perform_fit(V_1, I_1, expon, p0=[10**(-10), 1],  chi_square=True)
#print(a, b, da, db)

"""configurazione 2 (voltometro in parallelo col generatore)"""
V = np.array([0.0156    , 0.018      , 0.105     , 0.207      , 0.258     , 0.305     , 0.360     , 0.401     , 0.418     , 0.447     , 0.473    , 0.509    , 0.529    , 0.550     , 0.570     , 0.616   , 0.626 , 0.652 , 0.713 , 0.749  , 0.805])
I = {0.02: 0.01, 0.021: 0.01, 0.03: 0.01, 0.031: 0.01, 0.05: 0.01, 0.12: 0.01, 0.65: 0.01, 2.37: 0.02, 3.88: 0.02, 9.64: 0.02, 19.5: 0.1, 49.3: 0.1, 78.0: 0.1, 120.2: 0.2, 176.7: 0.2, 358.6: 1, 410: 1, 550: 2, 951: 2, 1214: 2, 1658: 2}
I_ = np.array(list(I.keys()))
R_eq = np.array(V/I_)
I_err = np.array(list(I.values()))
I_2 = create_dataset(I, magnitude=-6)
V_2 = create_dataset(V, 0.001)
#(a,b), (da, db), chi, dof = perform_fit(V, I, expon, p0=[10**(-10), 1],  chi_square=True)
#print(a, b, da, db)
#latex_table("I", I_2, "V", V_2, "R_eq", create_dataset(R_eq), orientation='v')
"""
create_best_fit_line(V_1, I_1, func=expon, p0=[10e-10, 1],
                     xlabel="Tensioni (V)", ylabel="Correnti (A)", title="diodo",
                     label_fit=["configurazione 1"], residuals=True,
                     show_chi_squared=True, show_fit_params=True)
"""
maschera = (V_1['value'] > 0.0) & (V_1['value'] < 0.9)
non_maschera = [True for _ in range(len(V_2['value']))]
create_best_fit_line(V_1, I_1, V_2, I_2, func=expon, p0=[[10**(-10), 1],[10**(-10), 1]], # Guess iniziali NON obbligatorio
                    ylabel="Correnti (A)",
                    xlabel="Tensioni (V)", # Titoli condivisi
                    title="Diodo", 
                    label_fit=["configurazione 1", "configurazione 2"], # Nome delle linee
                    show_chi_squared=True, show_fit_params=True, show_masked_points=False,
                    residuals=True, masks=[maschera, non_maschera], confidence_interval=0.9,
                    ) # Plotta separatamente i grafici


