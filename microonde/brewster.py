import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

#STUDI0 ONDA TRASMESSA
#cono con lato lungo parallelo al tavolo
#offset fisso, amplificazione 3x
d = 70 #cm
#preso rispetto alla perpendicolare alla lastra 
#offset 3
#presunta polarizzazione verticale lato corto perpendicolare al tavolo
angoli =   [0   , 5   , 10  , 15  , 20  , 25  , 30  , 35  , 40  , 45  , 50  , 55  , 60  ] 
tensione = [2.50, 2.55, 2.84, 2.90, 2.76, 2.66, 2.65, 2.52, 2.56, 2.71, 2.86, 3.17, 2.64]
tens_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02]

#lato corto parallelo al tavolo
angoli =   [0   , 5   , 10  , 15  , 20  , 25  , 30  , 35  , 40  , 45  , 50  , 55  , 60  , 65  , 70  ]
tensione = [2.47, 2.46, 2.54, 2.59, 2.54, 2.55, 2.52, 2.68, 3.02, 3.62, 4.34, 4.66, 4.45, 3.00, 1.58]
tens_err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
angoli = Measurement(angoli, 2)
tensione = Measurement(tensione, tens_err)
def parab(angoli, a, b, c):
    return a*(angoli-b)**2 + c
fit = perform_fit(angoli[-6:], tensione[-6:], parab, p0=[-1, 55, 4.5])
plot_fit(fit, plot_residuals=True, save_path='./grafici/test.png', show_params=True, show_stats=True)