import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

def frequenza_taglio (L, C ) :
    return 1 / (2*np.pi*np.sqrt(L * C_nota.value)) # Use .value for C_nota here

def Hr (w, L, C) :
    R = R_nota.value
    return R / np.sqrt (R**2 + (w*L - 1/(w*C))**2)
def fase_Hr (w, k, L, C) :
    R = R_nota.value
    return k -np.arctan ((w*L - 1/(w*C))/R) #k atteso 0

def Hl (w, L, C) :
    R = R_nota.value
    return w*L / np.sqrt (R**2 + (w*L - 1/(w*C))**2)
def fase_Hl (w, k, L, C) : # This function is defined but not used for fitting in the script
    R = R_nota.value
    return k - np.arctan ((w*L - 1/(w*C))/R) #k atteso pi/2

def Hc (w, L, C) :
    R = R_nota.value
    return 1 / (w*C) / np.sqrt (R**2 + (w*L - 1/(w*C))**2)
def fase_Hc (w, k, L, C) : # This function is defined but not used for fitting in the script
    R = R_nota.value
    return k - np.arctan ((w*L - 1/(w*C))/R)

#regime sovrasmorzato
C_nota = Measurement(10, 1, magnitude=-9, unit='F', name='$C_{\\text{nota}}$') # Nano Farad
R_nota = Measurement(10, 0.1, magnitude=3, unit='Ohm', name='$R_{\\text{nota}}$') # Kilo Ohm
L_guess = 50e-3

# --- Data and Fit for H_R (Resistor Voltage Transfer Function) ---
frequenza_hr_vals = [100, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 18000, 25000, 30000, 40000, 50000, 65000, 80000, 100000, 120000, 150000, 200000, 240000, 270000] #hertz
frequenza_hr = Measurement(frequenza_hr_vals, name='Frequenza', unit='Hz')
pulsazione_hr = (frequenza_hr * 2 * np.pi)
pulsazione_hr.name = '$\\omega$'
pulsazione_hr.unit = 'rad/s'

amp_Vg_hr_vals = [4.16, 4.16, 4.16, 4.16, 4.16, 4.16, 4.16, 4.20, 4.32, 4.32, 4.16, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.36, 4.36, 4.36, 4.36, 4.36, 4.16] #volt
amp_Vg_hr_err = 0.04 #volt
amp_Vg_hr = Measurement(amp_Vg_hr_vals, amp_Vg_hr_err, name='$V_g$', unit='V')

amp_Vr_vals = [0.256, 1.16, 2.10, 3.24, 3.76, 3.92, 4.08, 4.20, 4.36, 4.32, 4.08, 3.96, 3.80, 3.40, 3.04, 2.60, 2.12, 1.72, 1.22, 1, 0.44, 0.22, 0.0576] #volt
amp_Vr_err_vals = [0.004, 0.02, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.04, 0.04, 0.02, 0.0008] #volt
amp_Vr = Measurement(amp_Vr_vals, amp_Vr_err_vals, name='$V_R$', unit='V')
amp_Hr = amp_Vr / amp_Vg_hr
amp_Hr.name = '$|H_R|$'
amp_Hr.unit = ''

fase_Vr_vals = [85, 73, 61, 38, 28, 16, 6.5, 2, -4, -13, -19, -30, -35, -47, -56, -68, -75, -83, -95, -105, -116, -123, -125] #gradi
fase_Vr_err_vals = [2, 1, 1, 3, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 4, 5, 1, 3, 5, 4, 5, 1] #gradi
fase_Vr = (Measurement(fase_Vr_vals, fase_Vr_err_vals) * np.pi / 180)
fase_Vr.name = '$\\arg(H_R)$'
fase_Vr.unit = 'rad'

fit_Hr = perform_fit(pulsazione_hr, amp_Hr, Hr, [L_guess, C_nota.value], method='minuit', minuit_limits={'L':(-1,1), 'C':(-1,1)}, parameter_names=['L', 'C'])
plot_fit(fit_Hr, plot_residuals=True, 
         data_label='dati', xlabel='$\\omega$ (rad/s)', ylabel='$|H_R|$', title='Circuito RLC - $|H_R(\\omega)|$', save_path='./grafici/rlc_hr.pdf')
fit_fase_Hr = perform_fit(pulsazione_hr, fase_Vr, fase_Hr, [0.0, L_guess, C_nota.value], method='minuit', minuit_limits={'k':(-10,10), 'L':(-1,1), 'C':(-1,1)}, parameter_names=['k','L', 'C'])
plot_fit(fit_fase_Hr, plot_residuals=True,
         data_label='dati', xlabel='$\\omega$ (rad/s)', ylabel='$\\arg(H_R)$ (rad)', title='Circuito RLC - $\\arg(H_R(\\omega))$', save_path='./grafici/rlc_fase_hr.pdf')

print("\n--- Tabella Dati per H_R (RLC) ---")
latex_data_hr = latex_table_data(frequenza_hr, amp_Vg_hr, amp_Vr, fase_Vr,
                                 orientation='v', sig_figs_error=1,
                                 caption='Dati Sperimentali per $H_R$ nel Circuito RLC (Regime Sovrasmorzato)')
print(latex_data_hr)

# --- Data and Fit for H_L (Inductor Voltage Transfer Function) ---
# una sonda ai capi del generatore e una sonda tra L e C (ai capi di C + R)
# poi differenza tra 2 segnali delle 2 sonde per trovare segnale ai capi di L
frequenza_hl_vals = [0.5, 1, 2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 350] #kilo hertz
frequenza_hl = Measurement(frequenza_hl_vals, magnitude=3, name='Frequenza', unit='Hz')
pulsazione_hl = (frequenza_hl * 2 * np.pi)
pulsazione_hl.name = '$\\omega$'
pulsazione_hl.unit = 'rad/s'

amp_Vg_hl_vals = [4.12, 4.16, 4.12, 4.12, 4.16, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.36, 4.36, 4.36, 4.36, 4.36, 4.36, 4.36] #volt
amp_Vg_hl_err = 0.04 #volt
amp_Vg_hl = Measurement(amp_Vg_hl_vals, amp_Vg_hl_err, name='$V_g$', unit='V')

amp_Vl_vals = [0.28, 0.28, 0.36, 0.68, 0.96, 1.40, 1.88, 2.28, 2.68, 3.0, 3.56, 3.96, 4.20, 4.36, 4.52, 4.60, 4.72, 4.60, 4.44, 4.40, 4.28] #volt
amp_Vl_err_vals = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.20, 0.04, 0.04, 0.04, 0.02, 0.04, 0.04] #volt
amp_Vl = Measurement(amp_Vl_vals, amp_Vl_err_vals, name='$V_L$', unit='V')
amp_Hl = amp_Vl / amp_Vg_hl
amp_Hl.name = '$|H_L|$'
amp_Hl.unit = ''

fit_Hl = perform_fit(pulsazione_hl, amp_Hl, Hl, [L_guess, C_nota.value], method='minuit', minuit_limits={'L':(-1,1), 'C':(-1,1)}, parameter_names=['L', 'C'])
plot_fit(fit_Hl, plot_residuals=True,
         data_label='dati', xlabel='$\\omega$ (rad/s)', ylabel='$|H_L|$', title='Circuito RLC - $|H_L(\\omega)|$', save_path='./grafici/rlc_hl.pdf')
# NOI TUTTI CI ABBIAMO PROVATO, MICHELE PIù DEGLI ALTRI, MA LA FASE NON SI TROVA "POTA"

print("\n--- Tabella Dati per H_L (RLC) ---")
latex_data_hl = latex_table_data(frequenza_hl, amp_Vg_hl, amp_Vl,
                                 orientation='v', sig_figs_error=1,
                                 caption='Dati Sperimentali per $H_L$ nel Circuito RLC')
print(latex_data_hl)

# --- Data and Fit for H_C (Capacitor Voltage Transfer Function) ---
# una sonda ai capi del generatore e una sonda tra L e C scambiate di posizione rispettoa prima (ai capi di L + R)
# poi differenza tra 2 segnali delle 2 sonde per trovare segnale ai capi di C (corrected comment, was L)
frequenza_hc_vals = [40, 100, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 9000, 12000, 16000, 22000, 30000, 40000, 70000, 100000] #hertz
frequenza_hc = Measurement(frequenza_hc_vals, name='Frequenza', unit='Hz')
pulsazione_hc = (frequenza_hc * 2 * np.pi)
pulsazione_hc.name = '$\\omega$'
pulsazione_hc.unit = 'rad/s'

amp_Vg_hc_vals = [4.12, 4.12, 4.16, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.32, 4.28, 4.28, 4.28, 4.32, 4.32, 4.28, 4.28, 4.32] #volt
amp_Vg_hc_err = 0.04 #volt
amp_Vg_hc = Measurement(amp_Vg_hc_vals, amp_Vg_hc_err, name='$V_g$', unit='V')

amp_Vc_vals = [4.20, 4.20, 4, 3.80, 3.64, 3.24, 2.84, 2.52, 2.2, 2.0, 1.84, 1.52, 1.36, 1.44, 1.24, 1.0, 0.8, 0.76, 0.72, 0.6, 0.5, 0.5] #volt
amp_Vc_err_vals = [0.04, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08 , 0.04, 0.08, 0.04, 0.04, 0.04, 0.08, 0.04, 0.08, 0.08] #volt
amp_Vc = Measurement(amp_Vc_vals, amp_Vc_err_vals, name='$V_C$', unit='V')
amp_Hc = amp_Vc / amp_Vg_hc
amp_Hc.name = '$|H_C|$'
amp_Hc.unit = ''

fit_Hc = perform_fit(pulsazione_hc, amp_Hc, Hc, [L_guess, C_nota.value], method='minuit', minuit_limits={'L':(-1,1), 'C':(-1,1)}, parameter_names=['L', 'C'])
plot_fit(fit_Hc, plot_residuals=True,
         data_label='dati', xlabel='$\\omega$ (rad/s)', ylabel='$|H_C|$', title='Circuito RLC - $|H_C(\\omega)|$', save_path='./grafici/rlc_hc.pdf')
print(fit_Hc.parameters['L'].value, fit_Hc.parameters['L'].error)
print("\n--- Tabella Dati per H_C (RLC) ---")
latex_data_hc = latex_table_data(frequenza_hc, amp_Vg_hc, amp_Vc,
                                 orientation='v', sig_figs_error=1,
                                 caption='Dati Sperimentali per $H_C$ nel Circuito RLC')
print(latex_data_hc)

# --- Tabella Risultati Fit Complessiva per RLC ---
print("\n--- Tabella Risultati Fit per Circuito RLC ---")
fit_names_rlc = ['$|H_R|$', '$\\arg(H_R)$', '$|H_L|$', '$|H_C|$']
# parameter_names=['L', 'C'] for amp_Hr, amp_Hl, amp_Hc
# parameter_names=['k','L', 'C'] for fase_Hr
latex_fit_rlc = latex_table_fit(fit_Hr, fit_fase_Hr, fit_Hl, fit_Hc,
                                fit_labels=fit_names_rlc,
                                param_labels={'L': '$L \\text{ (H)}$', 'C': '$C \\text{ (F)}$', 'k': '$k \\text{ (rad)}$'},
                                caption='Risultati dei Fit per il Circuito RLC',
                                sig_figs_error=2,
                                include_stats=['chi_square', 'dof', 'reduced_chi_square'])
print(latex_fit_rlc)