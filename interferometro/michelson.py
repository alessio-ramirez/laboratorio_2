import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

#CALIBRAZIONE MICROMETRO
delta_n = [23, 25, 26, 23, 26, 24, 24]
delta_n = Measurement(delta_n, 1)
lambd = 632.8e-9
delta_d = lambd*delta_n/2
delta_d_m = weighted_mean(delta_d)
print(f"delta_m = {delta_d_m}")

#MISURA INDICE RIFRAZIONE ARIA
#indice di rifrazione dipende dalla pressione con una legge del tipo n=mP+1 -> trovare m
def aria_index(delta_P, m) :
    lambd = 632.8e-9
    s = 3e-2
    return 2*s*m/lambd * delta_P

s = Measurement(3, magnitude=-2) #spessore vacuum cell
delta_n = [3, 6, 9, 12, 15, 18, 21]
delta_P = [10, 20, 30, 40, 50, 60, 70] #kPa
delta_P = Measurement(delta_P, 2, magnitude=3, name="$\\Delta P$", unit="Pa")
delta_n = Measurement(delta_n, 1, name="$\\Delta N$", )
#latex_table_data(delta_P, delta_n, orientation='h')

fit = perform_fit(delta_P, delta_n, aria_index, method='odr')
plot_fit(fit, save_path='./grafici/indice_aria.pdf', plot_residuals=True, title='Scorrimento di frange al variare della pressione')
indice_aria = fit.parameters['m'] * 101.325e3 + 1
print(f"indice aria = {indice_aria.to_eng_string(sig_figs_error=3)}")

#MISURA INDICE RIFRAZIONE VETRO
lambd = 632.8e-9
t = Measurement(6.8, 0.1, magnitude=-3) #spessore vetro
theta_i = Measurement(0, 0.1) #angolo minimo cammino ottico (sensibilità goniometro 0.1 gradi)
theta_f = [2, 4, 6, 8, 10]
theta_f = Measurement(theta_f, 0.1)
theta = (theta_f-theta_i)*np.pi/180
delta_n = [3, 13, 26, 58, 87]
delta_n_err = [1, 1, 1, 2, 3]
delta_n = Measurement(delta_n, delta_n_err)
n_vetro = (2*t-delta_n*lambd)*(1-np.cos(theta))/(2*t*(1-np.cos(theta))-delta_n*lambd)
#print(n_vetro)
n_vetro_m = weighted_mean(n_vetro)
#print(n_vetro_m)

