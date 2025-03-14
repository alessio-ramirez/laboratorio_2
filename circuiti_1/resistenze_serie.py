from resistenza_amperometro import R_a, R_a_err
from resistenza_voltometro import R_v, R_v_err
import numpy as np
from liblab import *
import math

#resistenze in serie
R_nota_1 = 68.0 * 10**3 #ohm
R_nota_2 = 21.87 * 10**3 #ohm
R_eq = R_nota_1 + R_nota_2
R_atteso = R_eq
R_atteso_err = math.sqrt((0.1*10**3)**2 + (0.1*10**3)**2)

I = np.array([0.16, 11.55, 22.76, 34.09, 45.26, 56.24, 67.40, 6.06, 16.81, 28.03]) * 10 **(-6)
V = np.array([0.013, 1.029, 2.028, 3.039, 4.034, 5.013, 6.008, 0.539, 1.498, 2.498])
I = {val: 0.03 * 10 ** (-6) for val in I}
V = {val: 0.001 for val in V}
a, R, da, R_err = lst_squares(I, V)[0]
print(R_atteso, R_atteso_err, R, R_err)


#print(test_comp(R_atteso, R_atteso_err, R, R_err))

