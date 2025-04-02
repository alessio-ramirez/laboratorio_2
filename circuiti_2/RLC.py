import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from burger_lib.guezzi import *
import pyperclip

#REGIME SOTTOSMORZATO

R_nota = 500 #ohm
R_nota = Measurement(500, R_nota/100, unit='Ohm', name='R sottosmorzato')
R_osc = 1 #Mohm
R_osc = Measurement(1, 0, magnitude = 6, unit='Ohm', name='R oscillatore')
R_eq = (R_nota * R_osc) / (R_nota + R_osc)
C_att = 10 #nano farad
C_att = Measurement(C_att, C_att/10, magnitude=-9, unit='F', name='Capacità attesa')#errore 10% sia da misura multimetro sia da etichetta condensatore
L_stima = 0.07
L_err = 0.00028721
L_att = Measurement(L_stima, L_err)
def V (x, k, gamma, beta):
    return k * np.exp(-(x * gamma)) * np.sin(beta * x ) 

V0 = 80.8 #ohm
tempo_sotto = [0, 16, 24, 44, 60, 72, 88, 96, 120, 136, 148, 160, 172, 184, 200, 224, 244, 260, 276, 288, 312, 332, 348, 364, 380, 404, 420, 436, 444, 456, 468, 488, 516, 528, 540, 552, 580, 608, 624, 640, 672, 692, 704, 720, 736, 760, 776, 788, 800, 816, 832, 852] #microsecondi
V_r = [1.2, 45.4, 63.2, 80.8, 67.6, 45.6, 8.80, -8.8, -48, -52.8, -46.4, -32, -12.4, 7.6, 28.8, 41, 30.4, 12.8, -5.6, -17.6, -27.2, -20.4, -8.8, 4.8, 15.2, 20.8, 17.2, 8.8, 4.0, -3.2, -8.8, -13.6, -8.0, -3.2, 2.0, 6.4, 11.6, 7.2, 2.0, -2.4, -6.4, -4, -1.2, 2.0, 4.4, 6.4, 5.6, 4.0, 2.0, -0.4, -2, -2.8] #millivolt
tempo_sotto = Measurement (tempo_sotto, 1, magnitude=-6, unit= 's', name= 'Tempo')
V_r = Measurement(V_r, 0.4, magnitude=-3, unit='V', name='Tensione')

fit = perform_fit(tempo_sotto, V_r, V, [100, 50, 3000], ['V_0', 'gamma', 'beta'])
plot_fit(fit, plot_residuals=True, n_fit_points=10000, title='Circuito RLC - Regime Sottosmorzato', save_path='./grafici/sottosmorzato.png')
dati_1 = latex_table_data(tempo_sotto, V_r, orientation='v', caption='dati regime sottosmorzato')
risultati_fit_1 = latex_table_fit(fit, orientation='v', caption='fit regime sottosmorzato')
gamma_att = R_eq/(2 * L_att)

################################################################################################################################################################################################################################################################################################################################à

#REGIME SOVRASMORZATO
def V (x, V_0, gamma, beta):
    return V_0*np.exp(-gamma*x)*(np.exp(beta*x)-np.exp(-beta*x))
R_nota = 15 #kilo ohm
R_nota = Measurement(R_nota, R_nota/100, magnitude=3, unit='Ohm', name='R sovrasmorzato')
V0 = 3720 # millivolt

err_tempo = 1 #microsecondo
err_V_r = 40 #millivolt
tempo_sovra = [0, 2, 4, 6, 8, 10, 18, 24, 30, 40, 46, 50, 58, 64, 80, 90, 102, 128, 148, 168, 184, 204, 222, 242, 262, 282, 302, 316, 332, 352, 362, 386, 406, 434, 462] #microsecondi
V_r = [0, 720, 1720, 2520, 3000, 3320, 3680, 3640, 3480, 3280, 3120, 3040, 2880, 2760, 2480, 2280, 2120, 1840, 1520, 1320, 1160, 1040, 920, 800, 680, 600, 520, 480, 440, 360, 360, 280, 240, 200, 160] #millivolt
tempo_sovra = Measurement(tempo_sovra, err_tempo, magnitude=-6, unit='s', name='Tempo')
V_r = Measurement(V_r, err_V_r, magnitude=-3, unit='V', name='Tensione')
fit = perform_fit(tempo_sovra, V_r, V, [100, 3000, 3000], ['V_0', 'gamma', 'beta'])
plot_fit(fit, n_fit_points=10000, plot_residuals=True, title='Circuito RLC - Regime Sovrasmorzato', save_path='./grafici/sovrasmorzato.png')
dati_2 = latex_table_data(tempo_sovra, V_r, orientation='v', caption='dati regime sovrasmorzato')
risultati_fit_2 = latex_table_fit(fit, orientation='v', caption='fit regime sovrasmorzato')
##########################################################################################################################################################################################################################

#SMORZAMENTO CRITICO
def V (x, V0, gamma):
    return V0*x*np.exp(-gamma*x)

R_nota = 5300 #ohm
R_nota = Measurement(R_nota, R_nota/100, unit='Ohm', name='R smorzamento critico')
R = 2 * L_att / np.sqrt (L_att * C_att)
V0 = 3900 #millivolt

err_tempo = 1 #microsecondo
err_V_r = 40 #millivolt
tempo_critico = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 31, 34, 40, 43, 48, 51, 56, 62, 70, 78, 85, 93, 103, 111, 119, 129, 136, 147, 158, 165, 172, 181, 200] #microsecondi
V_r = [20, 400, 820, 1200, 1500, 1800, 2040, 2220, 2380, 2520, 2620, 2700, 2760, 2820, 2820, 2780, 2660, 2580, 2420, 2300, 2120, 1880, 1580, 1300, 1100, 880, 660, 520, 400, 300, 240, 160, 100, 80, 60, 40, 20] #millivolt
tempo_critico = Measurement(tempo_critico, err_tempo, magnitude=-6, unit='s', name='Tempo')
V_r = Measurement(V_r, err_V_r, magnitude=-3, unit='V', name='Tensione')
fit = perform_fit(tempo_critico, V_r, V, [100, 3000], ['V_0', 'gamma'])
plot_fit(fit, n_fit_points=10000, plot_residuals=True, title='Circuito RLC - Smorzamento Critico', save_path='./grafici/critico.png')
dati_3 = latex_table_data(tempo_critico, V_r, orientation='v', caption='dati smorzamento critico')
risultati_fit_3 = latex_table_fit(fit, orientation='v', caption='fit smorzamento critico')
pyperclip.copy(dati_1 + risultati_fit_1 + dati_2 + risultati_fit_2 + dati_3 + risultati_fit_3)

#create_best_fit_line(tempo_critico, V_r, func=V, p0=[[100, 3000]], show_chi_squared=True, show_fit_params=True, residuals=True)
V0_stima = 2.743 * 10**5
V0_stima__err = 2636
gamma_stima = 33860
gamma_stima_err = 231.6
chi_quadro_ = 29.18
chi = 0.8337
gdl = 35