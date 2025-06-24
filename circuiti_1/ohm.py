import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

#definisco la funzione di fit
def linear_func(I, a, R):
    return a + R * I

R_atteso = 1e6 #come scritto sulla strumentazione
R_atteso_err = 1/100 * R_atteso
R_atteso = Measurement(R_atteso, R_atteso_err, name='$R_{\\text{atteso}}$', unit='Ω')
I_err = 0.03
V_err = 0.01
#configurazione 2 (voltometro in parallelo con generatore)
#array di dati
I_2 = [0.03, 1.04, 2.02, 3.03, 4.03, 5.02, 6.03, 7.07, 8.03, 9.03, 10.07, 11.08, 12.02, 13.03, 14.04, 15.04, 16.05, 17.04, 18.04, 19.06, 20.03]
V_2 = [0.01, 1.02, 2.00, 3.01, 4.00, 5.00, 6.01, 7.05, 8.00, 9.00, 10.04, 11.04, 11.99, 13.00, 14.00, 15.01, 16.02, 17.01, 18.00, 19.03, 20.00]
I_2 = Measurement(I_2, I_err, magnitude=-6, name='$I_2$', unit='A')
print(f"I2 = {I_2}")
V_2 = Measurement(V_2, V_err, name='$V_2$', unit='V')
fit_2 = perform_fit(I_2, V_2, linear_func, p0=[1, 1e6], method='odr')
R_bias = (V_2 - I_2 * 1.8)/(I_2 - V_2/10596157 + 1.8/10596157 * I_2)
a, R_2 = (fit_2.parameters['a'], fit_2.parameters['R'])
print(f"z-score tra intercetta 2 e zero(1Mohm): {test_comp(a, Measurement(0,0))['z_score']}")
print(f"z-score tra R atteso e R2 (1Mohm): {test_comp(R_atteso, R_2)['z_score']}")

#configurazione 1 (voltomeytro in parallelo con resistenza)
#per tensioni alte la resistenza misurata risulta più alta della resistenza effettiva perchè la corrente si sdoppia tra la resistenza scelta e quella del voltmetro
I_1 = [0.06, 1.14, 2.22, 3.31, 4.43, 5.51, 6.75, 7.75, 8.86, 9.97, 11.04, 12.16, 13.22, 14.34, 15.48, 16.53, 17.62, 18.73, 19.84, 20.93, 22.04]
V_1 = [0.013, 1.010, 2.005, 3.002, 4.029, 5.009, 6.159, 7.02, 8.02, 9.02, 10.01, 11.03, 12.00, 13.01, 14.04, 15.00, 16.00, 17.00, 18.01, 19.00, 20.01]
I_1 = Measurement(I_1, I_err, magnitude=-6, name='$I_1$', unit='A')
V_1 = Measurement(V_1, V_err, name='$V_1$', unit='V')
fit_1 = perform_fit(I_1, V_1, linear_func, method='odr', p0=[1, 1e6])
R_bias = (V_1 - (I_1- V_1/10596157)*1.8) / (I_1 - V_1/10596157)
a, R_1 = (fit_1.parameters['a'], fit_1.parameters['R'])
print(f"z-score tra intercetta 1 e zero(1Mohm): {test_comp(a, Measurement(0,0))['z_score']}")
print(f"z-score tra R atteso e R1 (1Mohm): {test_comp(R_atteso, R_1)['z_score']}")

#latex_table_data(I_1, V_1, I_2, V_2, orientation='v', caption='dati con resistenza 1MΩ')
plot_fit(fit_1, fit_2, plot_residuals=True, show_plot=False, xlabel='Correnti (A)',
         ylabel='Tensioni (V)', save_path='./grafici/ohm_1M.pdf')
#latex_table_fit(fit_1, fit_2, orientation='v', caption='Tabella risultati dei fit con resistenza 1MΩ')
print(f"compatibilità tra le resistenze 1 e 2: {test_comp(R_1, R_2)['z_score']}")


############################################################
R_atteso = 10 #ohm
R_atteso_err = 1/100 * R_atteso
R_atteso = Measurement(R_atteso, R_atteso_err, name='$R_{\\text{atteso}}$', unit='Ω')
V_err = 0.001 #cambiata la sensibilità dello strumento

#configurazione 1
I = {45.591:0.00001, 95.51:0.01, 145.61:0.02, 190.02:0.01, 239.41:0.01, 284.45:0.01, 334.04:0.01, 378.75:0.1, 424.04:0.05, 473.20:0.1}
V = np.array([0.498, 1.006, 1.535, 2.003, 2.527, 3.005, 3.535, 4.014, 4.502, 5.037])
I_1 = Measurement(I, magnitude=-3, name='$I_1$', unit='A')
V_1 = Measurement(V, 0.001, name='$V_1$', unit='A')
fit_1 = perform_fit(I_1, V_1, linear_func, p0=[1,10], method='odr')
a, R_1 = (fit_1.parameters['a'], fit_1.parameters['R'])
print(f"z-score tra intercetta 1 e zero(10 ohm): {test_comp(a, Measurement(0,0))['z_score']}")
print(f"z-score tra R atteso e R1 (10 ohm): {test_comp(R_atteso, R_1)['z_score']}")
#configurazione 2
I = {0.12039:0.00002, 59.20:0.01, 112.29:0.01, 168.17:0.01, 223.94:0.01, 279.07:0.01, 335.41:0.03, 392.40:0.05, 444.00:0.04, 499.20:0.1}
V = np.array([0.001, 0.530, 1.005, 1.506, 2.006, 2.502, 3.011, 3.528, 4.000, 4.505])
I_2 = Measurement(I, magnitude=-3, name= '$I_2$', unit='A')
V_2 = Measurement(V, 0.001, name= '$V_2$', unit='V')
fit_2 = perform_fit(I_2, V_2, linear_func, p0=[1,10], method='odr')
a, R_2 = (fit_2.parameters['a'], fit_2.parameters['R'])
print(f"z-score tra intercetta 2 e zero(10 ohm): {test_comp(a, Measurement(0,0))['z_score']}")
print(f"z-score tra R atteso e R2 (10 ohm): {test_comp(R_atteso, R_2)['z_score']}")

latex_table_data(I_1, V_1, I_2, V_2, orientation='v', caption='dati con resistenza 10Ω')
plot_fit(fit_1, fit_2, plot_residuals=True, show_plot=False, xlabel='Correnti (A)',
         ylabel='Tensioni (V)', save_path='./grafici/ohm_10.pdf')
#latex_table_fit(fit_1, fit_2, orientation='v', caption='Tabella risultati dei fit con resistenza 10Ω')
print(f"compatibilità tra le resistenze 1 e 2: {test_comp(R_1, R_2)['z_score']}")
