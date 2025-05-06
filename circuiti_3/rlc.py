import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

def frequenza_taglio (L, C ) :
    return 1 / (2*np.pi*np.sqrt(L * C_nota))

def Hr (w, L, C) :
    R = R_nota.value
    return R / np.sqrt (R**2 + (w*L - 1/(w*C))**2)
def fase_Hr (w, k, L, C) :
    R = R_nota.value
    return k -np.arctan ((w*L - 1/(w*C))/R) #k atteso 0

def Hl (w, L, C) :
    R = R_nota.value
    return w*L / np.sqrt (R**2 + (w*L - 1/(w*C))**2)
def fase_Hl (w, k, L, C) :
    R = R_nota.value
    return k - np.arctan ((w*L - 1/(w*C))/R) #k atteso pi/2

def Hc (w, L, C) :
    R = R_nota.value
    return 1 / (w*C) / np.sqrt (R**2 + (w*L - 1/(w*C))**2)
def fase_Hc (w, k, L, C) :
    R = R_nota.value
    return k - np.arctan ((w*L - 1/(w*C))/R)

#regime sovrasmorzato
C_nota = Measurement(10, 1, magnitude=-9, unit='F', name='C nota') # Nano Farad
R_nota = Measurement(10, 0.1, magnitude=3, unit='Ohm', name='R nota') # Kilo Ohm
L_guess = 50e-3

frequenza = [100, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 18000, 25000, 30000, 40000, 50000, 65000, 80000, 100000, 120000, 150000, 200000, 240000, 270000] #hertz
pulsazione = Measurement(frequenza) * 2 * np.pi
amp_Vg = [4.16, 4.16, 4.16, 4.16, 4.16, 4.16, 4.16, 4.20, 4.32, 4.32, 4.16, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.36, 4.36, 4.36, 4.36, 4.36, 4.16] #volt
amp_Vg_err = 0.04 #volt
amp_Vg = Measurement(amp_Vg, amp_Vg_err)

amp_Vr = [0.256, 1.16, 2.10, 3.24, 3.76, 3.92, 4.08, 4.20, 4.36, 4.32, 4.08, 3.96, 3.80, 3.40, 3.04, 2.60, 2.12, 1.72, 1.22, 1, 0.44, 0.22, 0.0576] #volt
amp_Vr_err = [0.004, 0.02, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.04, 0.04, 0.02, 0.0008] #volt
amp_Vr = Measurement(amp_Vr, amp_Vr_err)
amp_Hr = amp_Vr / amp_Vg
fase_Vr = [85, 73, 61, 38, 28, 16, 6.5, 2, -4, -13, -19, -30, -35, -47, -56, -68, -75, -83, -95, -105, -116, -123, -125] #gradi
fase_Vr_err = [2, 1, 1, 3, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 4, 5, 1, 3, 5, 4, 5, 1] #gradi
fase_Vr = Measurement(fase_Vr, fase_Vr_err) * np.pi / 180

fit_Hr = perform_fit(pulsazione, amp_Hr, Hr, [L_guess, C_nota.value], method='minuit', minuit_limits={'L':(-1,1), 'C':(-1,1)})
plot_fit(fit_Hr, plot_residuals=True, show_params=True, show_stats=True,
         data_label='dati', xlabel='$\\omega$', ylabel='$H_R$', title='Circuito RLC - $H_R(\\omega)$', save_path='./grafici/rlc_hr.pdf')
fit_fase_Hr = perform_fit(pulsazione, fase_Vr, fase_Hr, [0.0, L_guess, C_nota.value], method='minuit', minuit_limits={'k':(-10,10), 'L':(-1,1), 'C':(-1,1)})
plot_fit(fit_fase_Hr, plot_residuals=True, show_params=True, show_stats=True,
         data_label='dati', xlabel='$\\omega$', ylabel='$\\arg(H_R)$', title='Circuito RLC - $\\arg(H_R(\\omega))$', save_path='./grafici/rlc_fase_hr.pdf')
#una sonda ai capi del generatore e una sonda tra L e C (ai capi di C + R)
#poi differenza tra 2 segnali delle 2 sonde per trovare segnale ai capi di L
frequenza = [0.5, 1, 2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 350] #kilo hertz
pulsazione= Measurement(frequenza, magnitude=3) * 2 * np.pi
amp_Vg = [4.12, 4.16, 4.12, 4.12, 4.16, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.36, 4.36, 4.36, 4.36, 4.36, 4.36, 4.36] #volt
amp_Vg_err = 0.04 #volt
amp_Vg = Measurement(amp_Vg, amp_Vg_err)

amp_Vl = [0.28, 0.28, 0.36, 0.68, 0.96, 1.40, 1.88, 2.28, 2.68, 3.0, 3.56, 3.96, 4.20, 4.36, 4.52, 4.60, 4.72, 4.60, 4.44, 4.40, 4.28] #volt
amp_Vl_err = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.20, 0.04, 0.04, 0.04, 0.02, 0.04, 0.04] #volt
amp_Vl = Measurement(amp_Vl, amp_Vl_err)
amp_Hl = amp_Vl / amp_Vg

fit_Hl = perform_fit(pulsazione, amp_Hl, Hl, [L_guess, C_nota.value], method='minuit', minuit_limits={'L':(-1,1), 'C':(-1,1)})
plot_fit(fit_Hl, plot_residuals=True, show_params=True, show_stats=True,
         data_label='dati', xlabel='$\\omega$', ylabel='$H_L$', title='Circuito RLC - $H_L(\\omega)$', save_path='./grafici/rlc_hl.pdf')
# NOI TUTTI CI ABBIAMO PROVATO, MICHELE PIù DEGLI ALTRI, MA LA FASE NON SI TROVA "POTA"

#una sonda ai capi del generatore e una sonda tra L e C scambiate di posizione rispettoa prima (ai capi di L + R)
#poi differenza tra 2 segnali delle 2 sonde per trovare segnale ai capi di L
frequenza = [40, 100, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 9000, 12000, 16000, 22000, 30000, 40000, 70000, 100000] #hertz
pulsazione = Measurement(frequenza) * 2 * np.pi
amp_Vg = [4.12, 4.12, 4.16, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.32, 4.28, 4.28, 4.28, 4.32, 4.32, 4.28, 4.28, 4.32] #volt
amp_Vg_err = 0.04 #volt
amp_Vg = Measurement(amp_Vg, amp_Vg_err)

amp_Vc = [4.20, 4.20, 4, 3.80, 3.64, 3.24, 2.84, 2.52, 2.2, 2.0, 1.84, 1.52, 1.36, 1.44, 1.24, 1.0, 0.8, 0.76, 0.72, 0.6, 0.5, 0.5] #volt
amp_Vc_err = [0.04, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08 , 0.04, 0.08, 0.04, 0.04, 0.04, 0.08, 0.04, 0.08, 0.08] #volt
amp_Vc = Measurement(amp_Vc, amp_Vc_err)
amp_Hc = amp_Vc / amp_Vg

fit_Hc = perform_fit(pulsazione, amp_Hc, Hc, [L_guess, C_nota.value], method='minuit', minuit_limits={'L':(-1,1), 'C':(-1,1)})
plot_fit(fit_Hc, plot_residuals=True, show_params=True, show_stats=True,
         data_label='dati', xlabel='$\\omega$', ylabel='$H_C$', title='Circuito RLC - $H_C(\\omega)$', save_path='./grafici/rlc_hc.pdf')