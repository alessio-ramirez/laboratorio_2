# --- START OF FILE rl.py ---

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import numpy as np
from burger_lib.guezzi import *
import matplotlib.pyplot as plt # Added for potential plt usage if needed later

R_nota = Measurement(10, 0.1, magnitude=3, unit='\\Omega', name='$R_{\\text{nota}}$') # Using Ohm symbol
R_L = Measurement(38.9, 0.1, unit='\\Omega', name='$R_L$') # Using Ohm symbol
L_guess = 50e-3

# Ampiezza del generatore (fissa in tutte le misure, l'abbiamo misurata ogni volta e rimaneva la stessa) misurata dall'oscilloscopio
amp_Vg = Measurement (4.16, 0.4, unit= 'V', name='$V_g$')

# Frequenza dell'onda sinusoidale presa dal generatore di funzioni
frequenza = [5, 7, 10, 15, 20, 25, 50, 75, 100, 125, 150, 180, 200, 225, 250] #kilo hertz
frequenza = Measurement(frequenza, magnitude=3, unit='Hz', name='Frequenza')
pulsazione = frequenza * 2 * np.pi
pulsazione.name, pulsazione.unit = ('$\\omega$', '') #pulsazione senza unità (rad/s implied)
# Ampiezza picco-picco ai capi dell'induttore, misurato dall'oscilloscopio con MATH (sottrazione delle due tensioni)
amp_Vl_pp_values = [0.640, 0.98, 1.36, 1.84, 2.32, 2.68, 3.80, 4.24, 4.32, 4.40, 4.40, 4.40, 4.32, 4.24, 4.16] # Corrected: removed extra comma and bracket
amp_Vl_pp_errors = [0.01, 0.02, 0.01, 0.01, 0.01, 0.04, 0.04, 0.04, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04]
amp_Vl_pp = Measurement(amp_Vl_pp_values, amp_Vl_pp_errors, unit='V', name='$V_L$')
# Ampiezza picco-picco ai capi della resistenza, misurato con l'oscilloscopio misurando direttamente la tensione con la sonda
amp_Vr_pp_values = [4.07, 4, 4.04, 3.90, 3.74, 3.50, 2.50, 1.80, 1.34, 0.98, 0.74, 0.52, 0.40, 0.184, 0.092]
amp_Vr_pp_errors = [0.01, 0.02, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.002, 0.004]
amp_Vr = Measurement(amp_Vr_pp_values, amp_Vr_pp_errors, unit='V', name='$V_R$')
# Fase Vl misurata dall'oscilloscopio (non a mano coi cursori ma automaticamente) # Note: Original code called this Vc phase
fase_Vl_values = [-90, -95, -101, -105, -115, -120, -142, -155, -162, -167, -171, -174, -176, -179, -180]
fase_Vl_errors = [2, 2, 2, 6, 6, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
fase_Vl = Measurement(fase_Vl_values, fase_Vl_errors) / 180 * np.pi
fase_Vl.name, fase_Vl.unit = ('$\\arg(H_L)$', 'rad')
# Fase Vr misurata dall'oscilloscopio (non a mano coi cursori ma automaticamente)
fase_Vr_values = [-8.5, -13, -16, -25, -32, -37.5, -62, -77, -87, -95, -101, -107, -116, -121, -122]
fase_Vr_errors = [0.5, 1, 1, 1, 1, 0.5, 1, 2, 2, 4, 5, 5, 5, 5, 5]
fase_Vr = Measurement(fase_Vr_values, fase_Vr_errors) / 180 * np.pi
fase_Vr.name, fase_Vr.unit = ('$\\arg(H_R)$', 'rad')

#funzioni

def fit_func_RL_lowpass(omega, L):
    R = R_nota.value
    return R / np.sqrt(R**2 + (omega * L)**2)

def fit_func_RL_highpass(omega, L):
    R = R_nota.value
    term_wL = omega * L
    return term_wL / np.sqrt(R**2 + term_wL**2)

def fase_Hr (omega, L, k) :
    R = R_nota.value
    return k - np.arctan(omega*L/R) #k atteso 0

def fase_Hl (omega, L, k) :
    R = R_nota.value
    return k - np.arctan(omega*L/R) # k atteso pi/2

#fit
ampiezza_Hl = amp_Vl_pp / amp_Vg
ampiezza_Hl.name = '$|H_L|$'
ampiezza_Hr = amp_Vr / amp_Vg
ampiezza_Hr.name = '$|H_R|$'
data_label = 'Dati Sperimentali'
mask = ampiezza_Hr.error > 0.015 

fit_Hr = perform_fit(pulsazione, ampiezza_Hr, fit_func_RL_lowpass, p0=[L_guess], parameter_names=['L'], method='minuit', minuit_limits={'L': (1e-5, 1)}, mask=mask)
plot_fit(fit_Hr, plot_residuals=True, title='Circuito RL - $|H_R(\\omega)|$', data_label=data_label, save_path='./grafici/rl_hr.pdf', show_params=True, show_stats=True)

fit_Hl = perform_fit(pulsazione, ampiezza_Hl, fit_func_RL_highpass, p0=[L_guess], parameter_names=['L'], method='minuit', minuit_limits={'L': (1e-5, None)}, mask=mask)
plot_fit(fit_Hl, plot_residuals=True, title='Circuito RL - $|H_L(\\omega)|$', data_label=data_label, save_path='./grafici/rl_hl.pdf', show_params=True, show_stats=True)

fit_fase_Hl = perform_fit(pulsazione, fase_Vl, func=fase_Hl, p0=[L_guess, np.pi/2], parameter_names=['L', 'k'], method='minuit', minuit_limits={'L': (1e-5, 1), 'k':(-10, 10)}) # Expect k around pi/2
plot_fit(fit_fase_Hl, plot_residuals=True, show_params=True, show_stats=True, title='Circuito RL - $\\arg(H_L(\\omega))$', data_label=data_label, save_path='./grafici/rl_fase_hl.pdf')

fit_fase_Hr = perform_fit(pulsazione, fase_Vr, fase_Hr, p0=[L_guess, 0.0], parameter_names=['L', 'k'], method='minuit', minuit_limits={'L': (1e-5, 1), 'k':(-10, 10)}) # Expect k around 0
plot_fit(fit_fase_Hr, plot_residuals=True, show_params=True, show_stats=True, title='Circuito RL - $\\arg(H_R(\\omega))$', data_label=data_label, save_path='./grafici/rl_fase_hr.pdf')

# --- Output Tables ---
latex_data = latex_table_data(frequenza, amp_Vg, amp_Vl_pp, amp_Vr, fase_Vl, fase_Vr,
                              orientation='v', sig_figs_error=1, caption='Dati Sperimentali per il Circuito RL')
print("\n--- LaTeX Data Table (RL) ---")
print(latex_data)

fit_names = ['$|H_R|$', '$|H_L|$', '$\\arg(H_R)$', '$\\arg(H_L)$']
latex_fit = latex_table_fit(fit_Hr, fit_Hl, fit_fase_Hr, fit_fase_Hl,
                            fit_labels=fit_names,
                            param_labels={'L': '$L$', 'k': '$k$'},
                            caption='Risultati dei Fit per il Circuito RL',
                            sig_figs_error=2) # Using 2 sig figs for fit params error
print("\n--- LaTeX Fit Summary Table (RL) ---")
print(latex_fit)

# --- END OF FILE rl.py ---