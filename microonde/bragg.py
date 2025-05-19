import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

def parabola (x, a, b, c) :
    return a*x**2 + b*x + c 

n = 1
d = 3.8 / 100 #m #spessore strato
angolo =   [20,   23,   25,   27,  30,   32,   35,   38,  40,   43,   45 ,  48,   50,    55,   60] #angolo di rotazione del cubo rispetto all'emettitore
tensione = [0.03, 0.03, 0.03, 0.04, 0.09, 0.18, 0.21, 0.23, 0.28, 0.32, 0.30, 0.15, 0.14, 0.02, 0.05] #in volt è vero
tens_err = [0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01]
theta = Measurement (angolo, 2)*np.pi/180
theta.name, theta.unit = ('$\\theta$', 'rad')
segnale = Measurement (tensione, tens_err)
#latex_table_data (theta, segnale, caption='Dati sperimentali cubo di Bragg', orientation='v' )
#plot_measurements(theta, segnale, save_path='./grafici/bragg.png')
#ultime misure poco sensate perchè il rilevatore è inclinato al punto che il segnale dal trasmettitore ci finisce direttamente dentro senza fare riflessione

mask_ = (theta.value > (29*np.pi/180)) & (theta.value < (51*np.pi/180))
fit_max = perform_fit (theta, segnale, parabola, mask=mask_)
#plot_fit(fit_max, save_path='./grafici/bragg.pdf', xlabel='$\\theta$ (rad)', plot_residuals=True, ylabel= 'V (volt)', title='Segnale misurato al variare di $\\theta$')
#latex_table_fit(fit_max, fit_labels=['Fit massimo'], orientation='v')
a = -8.3
err_a = 2
b = 12.1
err_b = 2.8
c = -4
err_c = 1

theta_max = -b/2/a
err_theta_max = np.sqrt ((theta_max/b*err_b)**2+(theta_max/a*err_a)**2)
lambda_s = 2 * d * np.sin(theta_max)/n
err_lambda = np.sqrt(err_theta_max**2*(2*d*np.cos(theta_max)/n)**2 + 0.001**2*(lambda_s/d)**2)
print (lambda_s, err_lambda, err_lambda/lambda_s)