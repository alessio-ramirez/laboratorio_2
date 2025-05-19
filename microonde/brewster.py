import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

def parabola (x, a, b, c) :
    return a*x**2 + b*x + c 
    
#STUDI0 ONDA TRASMESSA
#cono con lato lungo parallelo al tavolo
#offset fisso, amplificazione 3x
d = 70 #cm
#preso rispetto alla perpendicolare alla lastra 
#offset 3
#polarizzazione verticale, perpendicolare al piano di incidenza (piano in cui vive l'onda)
#lato corto perpendicolare al tavolo 
angoli =   [0   , 5   , 10  , 15  , 20  , 25  , 30  , 35  , 40  , 45  , 50  , 55  , 60  ] 
tensione = [2.50, 2.55, 2.84, 2.90, 2.76, 2.66, 2.65, 2.52, 2.56, 2.71, 2.86, 3.17, 2.64]
tens_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02]

theta = Measurement (angoli, 2)*np.pi/180
theta.name, theta.unit = ('$\\theta$', 'rad')
segnale = Measurement (tensione, tens_err, name='tensione', unit='V')
plot_measurements(theta, segnale, save_path='./grafici/brewster1.pdf', xlabel= '$\\theta$ (rad)', ylabel= 'V (volt)', title='Segnale trasmesso al variare di $\\theta$')
#latex_table_data (theta, segnale, caption='Dati trasmissione con polarizzazione perpendicolare al piano di incidenza', orientation='v' )

#lato corto parallelo al tavolo 
#polarizzazione parallela al tavolo 
angoli =   [0   , 5   , 10  , 15  , 20  , 25  , 30  , 35  , 40  , 45  , 50  , 55  , 60  , 65  , 70  ]
tensione = [2.47, 2.46, 2.54, 2.59, 2.54, 2.55, 2.52, 2.68, 3.02, 3.62, 4.34, 4.66, 4.45, 3.00, 1.58]
tens_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

theta = Measurement (angoli, 2)*np.pi/180
theta.name, theta.unit = ('$\\theta$', 'rad')
segnale = Measurement (tensione, tens_err, name='tensione', unit='V')
#latex_table_data (theta, segnale, caption='Dati trasmissione con polarizzazione parallela al piano di incidenza', orientation='v' )
mask_ = (theta.value > (40*np.pi/180)) 
fit_ = perform_fit (theta, segnale, parabola, mask=mask_)
#plot_fit(fit_, save_path='./grafici/brewster.pdf', plot_residuals=True, xlabel= '$\\theta$ (rad)', ylabel= 'V (volt)', title='Segnale trasmesso al variare di $\\theta$')
latex_table_fit(fit_, fit_labels=['Fit Brewster'], orientation='v')
a = -42.9
a_err = 4.3
b = 81.9
b_err = 8.4
c = -34.4
c_err = 4.1
theta_b = -b/2/a
err_theta_b = np.sqrt ((theta_b/b*b_err)**2+(theta_b/a*a_err)**2)
print (theta_b*180/np.pi, err_theta_b*180/np.pi)

#STUDIO CON ONDA RIFLESSA
#fisso lastra giro rilevatore
#lato corto parallelo al tavolo (polarizzazione parallela al tavolo, onda p)
#amplificazione 1x
d = 80 #cm
angolo_lastra = 40 
angolo_i = 40 
angolo_err = 2.5
angoli =   [60,   57,   55,   53,   50,   45,   40,   35,   30,   25,   20,   15,    10,    5,  0]
tensione = [0.12, 0.01, 0.04, 0.24, 0.28, 0.26, 0.61, 0.51, 0.80, 0.20, 0.13, 0.07, 0.03, 0.02, 0.01] #volt
tens_err = [0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.04, 0.04, 0.02, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01] #volt

#presunta polarizzazione verticale 
#lato corto perpendicolare al tavolo (onda s)
d = 80 #cm
angolo_lastra = 40 
angolo_i = 40 
angolo_err = 2.5
angoli =   [60,   55,   50,   45,   40,   35,   30,   25,   20,   15,   10,   5,    0]
tensione = [0.60, 1.60, 2.63, 3.26, 3.03, 2.54, 2.09, 1.02, 0.46, 0.20, 0.06, 0.02, 0.04 ]
tens_err = [0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.04, 0.03, 0.01, 0.01, 0.02]
