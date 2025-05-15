import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

spessore_fenditure = 1.5 #cm
distanza_fenditure = 7.5 #cm
#amplificazione 1x
#180 significa ricevitore e generatore muso a muso
angolo =   [180 , 175 , 177 , 173 , 170 , 167 , 165 , 163 , 160 , 157 , 155 , 152 , 150 , 148 , 145 , 143 , 140 , 138 , 135  ]
tensione = [0.66, 0.10, 0.32, 0.03, 0.04, 0.18, 0.32, 0.66, 0.66, 0.49, 0.28, 0.13, 0.01, 0.00, 0.07, 0.12, 0.15, 0.43, 0.65 ]
tens_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.03, 0.03 ]

angolo = Measurement(angolo, 1) * np.pi / 180
tensione = Measurement(tensione, tens_err) 

def cos(angolo, phi):
    return np.cos(angolo *  + phi)**2
fit = perform_fit(angolo, tensione, cos, [0.5, 0.5])
plot_fit(fit, plot_residuals=True, show_params=True, show_stats=True, save_path='./grafici/fenditura.png')

#distanza relativa ricevitore e sorgente 110cm, 1x amplificazione
distanza_fissa = 74 #cm
d_mobile = [56  , 55.5, 55  , 54.5, 54  , 53.5, 53.0, 52.5, 52.0, 51.5, 51.0, 50.5, 50.0, 49.5, 49.6]
tensione = [2.55, 1.26, 3.46, 2.23, 1.41, 3.96, 1.92, 1.78, 4.20, 1.74, 1.90, 4.28, 1.50, 2.50, 2.10]
tens_err = [0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01]

