import sys
sys.path.append('../')
from burger_lib.guezzi import *
import numpy as np

C_nota = Measurement(96, 1, magnitude=-9, unit='F', name='C nota') # Capacità indicata sul condensatore
R_nota = Measurement(10, 0.1, magnitude=3, unit= 'ohm', name='R nota') # Resistenza indicata sulla cassetta di resistenze
f_taglio = 1/(2*np.pi * R_nota * C_nota) # Frequenza di taglio ricavata

# Ampiezza del generatore (fissa in tutte le misure, l'abbiamo misurata ogni volta e rimaneva la stessa) misurata dall'oscilloscopio
amp_Vg = Measurement (4, 0.4, magnitude=1, unit= 'V', name='ampiezza Vg')

# Frequenza dell'onda sinusoidale presa dal generatore di funzioni
frequenza = [10, 15, 25, 50, 100, 120, 150, 200, 300, 450, 700, 1000, 1500, 2200, 3200, 4500, 6000, 7500, 10000, 15000, 20000, 30000] # hertz
frequenza = Measurement(frequenza, unit='Hz', name='Frequenza')
pulsazione = frequenza * 2 * np.pi 
# Ampiezza picco-picco ai capi del condensatore, misurato dall'oscilloscopio con MATH (sottrazione delle due tensioni)
amp_Vc_pp_values = [4, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.00, 4.00, 4.08, 4.04, 3.96, 3.76, 3.48, 3, 2.52, 2.12, 1.80, 1.40, 1.00, 0.760, 0.560] #volt
amp_Vc_pp_errors = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.2, 0.08, 0.4, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04] #volt error
amp_Vc_pp = Measurement(amp_Vc_pp_values, amp_Vc_pp_errors, unit='V', name='V_pp condensatore')
# Ampiezza picco-picco ai capi della resistenza, misurato con l'oscilloscopio misurando direttamente la tensione con la sonda
amp_Vr_pp_values = [12.8, 18.0, 28.8, 57.6, 112, 134, 168, 228, 336, 504, 784, 1090, 1580, 2140, 2700, 3180, 3480, 3640, 3800, 3920, 3980, 4040] #millivolt
amp_Vr_pp_errors = [0.4, 0.4, 0.4, 0.8, 2, 2, 2, 4, 2, 2, 2, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40] #millivolt error
amp_Vr = Measurement(amp_Vr_pp_values, amp_Vr_pp_errors, magnitude=-3, unit='V', name='V_pp resistenza')
# Fase Vc misurata dall'oscilloscopio (automaticamente)
fase_Vc_values = [178, 176, 179, 177, 177, 180, 175, 177, 174, 172, 170, 164, 157, 149, 139, 128, 120, 115, 108, 103, 102, 94] #gradi
fase_Vc_errors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #gradi error
fase_Vc = Measurement(fase_Vc_values, fase_Vc_errors)
# Fase Vr misurata dall'oscilloscopio (automaticamente)
fase_Vr_values = [92.1, 89.6, 88,9, 90.4, 88.2, 88.6, 88.0, 87.0, 84.9, 84.2, 79.6, 74.5, 67.6, 59.3, 47.9, 38.8, 31.1, 25.9, 19.4, 13.5, 10.1, 6.49] #gradi
fase_Vr_errors = [2, 1, 0.4, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05] #gradi error
fase_Vr = Measurement(fase_Vr_values, fase_Vr_errors)

