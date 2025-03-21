import numpy as np
from guezzi import *
from guezzi import create_dataset as cd

I = np.array([47.052, 23.785, 15.966, 11.925, 9.586])*10e-3 #milliampere 
V = np.array([4.804 for i in range(len(I))]) #volt
R_nota = np.array([100, 200, 300, 400, 500]) #ohm
R_amperometro = V/I - R_nota
R_nota_err = 1/100 * R_nota #1%
latex_table("I", cd(I, 0.001, magnitude=-3), "V", cd(V, 0.001), "R", cd(R_nota, R_nota_err))

R_a = np.mean(R_amperometro)
R_a_err = np.std(R_amperometro)
err_rel = R_a_err/R_a

print(R_a, R_a_err, err_rel)
