import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

d_i = 68 #cm preso dalla base senza considerare la posizione della sorgente
#offset fisso, amplificazione 3x
#angolo di rotazione del ricevitore rispetto al suo asse
angolo =   [0   , 5   , 10  , 15  , 20  , 25  , 30  , 35  , 40  , 45  , 50  , 55  , 60  , 65  , 70  , 75  , 80  , 85  , 90  , 95  , 100 , 105 , 110 , 115 , 120 , 125 , 130 , 135 , 140 , 145 , 150 , 155 , 160 , 165 , 170 , 175 , 180 ]
angolo = Measurement(angolo, 4) * np.pi / 180 #errore corrispondente alla sensibilità
tensione = [2.97, 2.95, 2.90, 2.79, 2.67, 2.51, 2.33, 2.11, 1.92, 1.67, 1.42, 1.13, 0.89, 0.58, 0.37, 0.19, 0.07, 0.02, 0.00, 0.03, 0.12, 0.27, 0.47, 0.72, 0.98, 1.26, 1.52, 1.75, 2.00, 2.24, 2.45, 2.65, 2.81, 2.89, 2.93, 2.96, 2.97]
tens_err = [0.02, 0.02, 0.02, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.01, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02]
tensione = Measurement(tensione, tens_err)
angolo.name, angolo.unit = ('$\\alpha$', 'rad')
#latex_table_data (angolo, tensione, caption='Dati sperimentali Legge di Malus', orientation='v' )

def fit_(angolo, a, b):
    return a + b * (np.cos(angolo))**2

fit = perform_fit(angolo, tensione, fit_)
latex_table_fit(fit, fit_labels=['Fit Malus'], orientation='v', param_labels={'a': 'k', 'b': '$V_0$'})
#plot_fit(fit, save_path='./grafici/Malus.pdf', plot_residuals=True, xlabel= '$\\alpha$ (rad)', ylabel= 'V (volt)', title='Segnale misurato al variare di $\\alpha$')

#segnale proporzionale all'intensità (E^2)

#campionare il segnale al variare del'angolo tra ricevitore e emettitore
d_i = 65 #cm preso dalla base senza considerare la posizione della sorgente
angolo =       [0,   5,    10,   15,   20,   25,   30,   35,   40,   45,   50,   -5,   -10,   -15,   -20,  -25, -30,   -35, -40,  -45,  -50]
tensione =     [3.79, 3.50, 2.86, 1.84, 1.09, 0.75, 0.40, 0.18, 0.09, 0.03, 0.02, 3.41, 2.55, 1.60, 0.95, 0.50, 0.35, 0.14, 0.07, 0.03, 0.02]
tensione_err = [0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.04, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.01]
angolo = Measurement(angolo, 4) * np.pi / 180
tensione = Measurement(tensione, 0) #volt
def fit1 (x, a, b) :
    return b*np.exp (-a*x**2)
def fit2 (x, a, b) :
    return b/(1+a*x**2)

#fit = perform_fit(angolo, tensione, fit1)
#plot_measurements(angolo, tensione, save_path='./grafici/gg.png')
#plot_fit(fit, save_path='./grafici/test.png', show_params=True, show_stats=True, plot_residuals=True)

#campionare il segnale al variare della distanza su intervallo piccolo (per vedere oscillazioni onde stazionarie)
#considerare la distanza tra i due massimi per trovare lunghezza d'onda 
#105 - d
d =        [15,   15.2, 15.4, 15.6, 15.8, 16,   16.2, 16.4, 16.1, 15.9, 16.6, 16.8, 17,   16.9, 17.1, 17.3, 17.5, 17.4, 17.2, 17.7, 17.9] #cm
tensione = [0.63, 0.59, 0.62, 0.65, 0.67, 0.68, 0.67, 0.64, 0.71, 0.68, 0.63, 0.63, 0.66, 0.65, 0.69, 0.72, 0.70, 0.72, 0.71, 0.70, 0.66]
tens_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
distanze = 105 - Measurement (d, 0.1) #cm
tensione = Measurement(tensione, tens_err)
#plot_measurements(distanze, tensione, save_path='./grafici/distanza.png')

#campionare il segnale nei massimi al variare della distanza 
#VALUTARE ERRORE SU DISTANZE
massimi =  [16.1, 17.3, 18.9, 20.2, 21.6, 23.1, 24.5,  26,  27.5, 28.9, 30.3, 33.2, 36,   38.9, 41.8, 44.7]
tensione = [0.71, 0.72, 0.76, 0.84, 0.88, 0.93, 0.96, 0.97, 0.99, 1.01, 1.04, 1.15, 1.23, 1.33, 1.44, 1.55]
tens_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01] 
distanze = -(Measurement (massimi, 0.2)-105) #cm
print(Measurement(massimi))
print(distanze)
tensione = Measurement( tensione, tens_err)
#plot_measurements(distanze, tensione, save_path='./grafici/distanza2.png')
