import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

I = [2.07, 1.66, 1.26, 1.42, 5.30] #microampere
V = [4.835, 4.835, 4.835, 4.835, 4.835] #volt
I_meas = Measurement(I, 0.001, magnitude=-6, name='$I$', unit='A')
V_meas = Measurement(V, 0.001, name='$V$', unit='V')
R_nota = np.array([3, 4, 6, 5, 1]) * 10 ** 6 #megaohm
R_nota = Measurement(R_nota, R_nota/100, name="$R_{\\text{nota}}$", unit='$\\Omega$')
R_voltometro = (V_meas * R_nota)/(I_meas * R_nota - V_meas)
R_voltometro.name, R_voltometro.unit = ('$R_{\\text{volt}}$', '$\\Omega$')

latex_table_data(I_meas, V_meas, R_nota, R_voltometro, orientation='h', caption='Caratterizzazione voltometro')
R_v = weighted_mean(R_voltometro)
err_rel = R_v.error/R_v.value
print(R_v, err_rel)