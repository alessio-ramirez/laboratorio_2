from resistenza_amperometro import R_a, R_a_err
from resistenza_voltometro import R_v, R_v_err
import numpy as np
from liblab import *
import math

R_nota_1 = 68.0 * 10**3 #ohm
R_nota_2 = 21.87 * 10**3 #ohm
R_atteso, R_atteso_err = eprop("r = 1 / (1 / a + 1 / b)", [R_nota_1, 0.1*10**3, R_nota_2, 0.01*10**3])

#resistenze in parallelo
I = np.array([1.11, 30.60, 61.03, 91.12, 122.77, 151.64, 183.30, 213.10, 243.24, 277.00, 303.11]) * 10 ** (-6)
V = np.array([0.018, 0.505, 1.008, 1.506, 2.029, 2.506, 3.029, 3.521, 4.019, 4.577, 5.009])
I = {val: 0.03 * 10 ** (-6) for val in I}
V = {val: 0.001 for val in V}

a, R, da, R_err = lst_squares(I, V)[0]
#print(test_comp(R_atteso, R_atteso_err, R, R_err))
print(R_atteso, R_atteso_err, R, R_err)