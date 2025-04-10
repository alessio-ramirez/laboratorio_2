import sys
sys.path.append('../')
import numpy as np
from burger_lib.guezzi import *

R_nota = Measurement(10, 0.1, magnitude=3, unit='Ohm', name='R nota')
R_L = Measurement(38.9, 0.1, unit='Ohm')

# Ampiezza del generatore (fissa in tutte le misure, l'abbiamo misurata ogni volta e rimaneva la stessa) misurata dall'oscilloscopio
amp_Vg = Measurement (4.16, 0.4, magnitude=1, unit= 'V', name='ampiezza Vg')

# Frequenza dell'onda sinusoidale presa dal generatore di funzioni
frequenza = [5, 7, 10, 15, 20, 25, 50, 75, 100, 125, 150, 180, 200, 225, 250] #kilo hertz
frequenza = Measurement(frequenza, magnitude=3, unit='Hz', name='Frequenza')
pulsazione = frequenza * 2 * np.pi 
# Ampiezza picco-picco ai capi dell'induttore, misurato dall'oscilloscopio con MATH (sottrazione delle due tensioni)
amp_Vl_pp_values = [0.640, 0.98, 1.36, 1.84, 2.32, 2.68, 3.80, 4.24, 4.32, 4.40, 4.40, 4.40, 4.32, 4.24, 4.16],
amp_Vl_pp_errors = [0.01, 0.02, 0.01, 0.01, 0.01, 0.04, 0.04, 0.04, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04]
amp_Vc_pp = Measurement(amp_Vl_pp_values, amp_Vl_pp_errors, unit='V', name='V_pp condensatore')
# Ampiezza picco-picco ai capi della resistenza, misurato con l'oscilloscopio misurando direttamente la tensione con la sonda
amp_Vr_pp_values = [4.07, 4, 4.04, 3.90, 3.74, 3.50, 2.50, 1.80, 1.34, 0.98, 0.74, 0.52, 0.40, 0.184, 0.092]
amp_Vr_pp_errors = [0.01, 0.02, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.002, 0.004]
amp_Vr = Measurement(amp_Vr_pp_values, amp_Vr_pp_errors, magnitude=0, unit='V', name='V_pp resistenza')
# Fase Vc misurata dall'oscilloscopio (non a mano coi cursori ma automaticamente)
fase_Vl_values = [-85, -95, -101, -105, -115, -120, -142, -155, -162, -167, -171, -174, -176, -179, 180]
fase_Vl_errors = [2, 2, 2, 6, 6, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
fase_Vc = Measurement(fase_Vl_values, fase_Vl_errors)
# Fase Vr misurata dall'oscilloscopio (non a mano coi cursori ma automaticamente)
fase_Vr_values = [-8.5, -13, -16, -25, -32, -37.5, -62, -77, -87, -95, -101, -107, -116, -121, 122]
fase_Vr_errors = [0.5, 1, 1, 1, 1, 0.5, 1, 2, 2, 4, 5, 5, 5, 5, 5]
fase_Vr = Measurement(fase_Vr_values, fase_Vr_errors)