import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

I = [47.052, 23.785, 15.966, 11.925, 9.586] #milliampere 
V = [4.804 for i in range(len(I))] #volt
I_meas = Measurement(I, 0.001, magnitude=-3, name='$I$', unit='A')
V_meas = Measurement(V, 0.001, name='$V$', unit='V')
R_nota = np.array([100, 200, 300, 400, 500]) #ohm
R_nota = Measurement(R_nota, R_nota/100, name="$R_{\\text{nota}}$", unit='$\\Omega$')
R_amperometro = V_meas/I_meas - R_nota
R_amperometro.name, R_amperometro.unit = ('$R_{\\text{amp}}$', '$\\Omega$')

latex_table_data(I_meas, V_meas, R_nota, R_amperometro, orientation='h', caption='Caratterizzazione amperometro')
R_a = weighted_mean(R_amperometro)
err_rel = R_a.error/R_a.value
print(R_a, err_rel)
