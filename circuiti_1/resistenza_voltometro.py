import numpy as np
from guezzi import *
from guezzi import create_dataset as cd

I = np.array([2.07, 1.66, 1.26, 1.42, 5.30]) * 10 ** (-6)#microampere
V = np.array([4.835, 4.835, 4.835, 4.835, 4.835]) #volt
R_nota = np.array([3, 4, 6, 5, 1])  * 10 ** 6 #megaohm
R_nota_err = 1/100 * R_nota
R_voltometro = (V * R_nota)/(I * R_nota - V)

R_v = np.mean(R_voltometro)
R_v_err = np.std(R_voltometro)
err_rel = R_v_err/R_v

print(R_v, R_v_err, err_rel)