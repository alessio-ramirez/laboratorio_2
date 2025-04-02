import numpy as np
import sys
import pyperclip
sys.path.append("../")
from burger_lib.guezzi import *

R_nota = Measurement(500, 5, unit='Ohm', name='resistenza nota')
err_tensione = 2 #millivolt
R_osc = Measurement(1, magnitude=6, unit='Ohm', name='resistenza oscilloscopio')
R_eq = (R_nota * R_osc)/(R_nota + R_osc)

#############################################################################################################################################################################################################################à

#INDUTTORE
def V_l(x, k, tau, c):
    return k * np.exp(-x/tau) + c
params = ['$k$', '$\\tau$', '$c$']
scarica_tempi_induttore = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 92, 100, 108, 116, 132, 148, 160, 180, 192, 208, 224, 236, 252, 264, 280, 304, 324, 364, 412, 448, 488, 516, 556, 604] #microsecondi
scarica_tensione_induttore = [-248, -224, -196, -172, -148, -128, -108, -88, -68, -52, -12, 0, 16, 28, 52, 72, 88, 108, 120, 132, 144, 152, 164, 168, 176, 188, 196, 208, 220, 224, 228, 232, 236, 240] #millivolt
scarica_tempi = Measurement(scarica_tempi_induttore, 1, magnitude=-6, unit='s', name='Tempo')
scarica_tensione = Measurement(scarica_tensione_induttore, 2, magnitude=-3, unit='V', name='Tensione')
fit = perform_fit(scarica_tempi, scarica_tensione, func=V_l, p0=[0.500, 0.0010, -1], parameter_names=params)
plot_fit(fit, plot_residuals=True, n_fit_points=10000, title='Circuito RL - Scarica Induttore' , save_path='./grafici/rl_scarica_induttore.png')
L_1 = fit.parameters['$\\tau$']*R_eq
dati_1 = latex_table_data(scarica_tempi, scarica_tensione, orientation='v', caption='dati scarica induttore')
risultati_fit_1 = latex_table_fit(fit, orientation='v', caption='fit scarica induttore')

##############################################################################################################################################################################
carica_tempi_induttore = [3.05, 3.07, 3.09, 3.11, 3.13 , 3.16, 3.18, 3.21, 3.22, 3.24, 3.29, 3.32, 3.36, 3.40, 3.44, 3.49, 3.54, 3.59, 3.62, 3.77] #millisecondi
carica_tensione_induttore = [248, 212, 152, 96, 48, -8, -40, -80, -92, -112, -152, -172, -192, -208, -216, -228, -232, -240, -240, -248] #millivolt
carica_tempi = Measurement(carica_tempi_induttore, 0.01, magnitude=-3, unit='s', name='Tempo') - 0.00305
carica_tensione = Measurement(carica_tensione_induttore, 2, magnitude=-3, unit='V', name='Tensione') + 0.248
fit = perform_fit(carica_tempi, carica_tensione, V_l, [-0.500, 0.0010, 1.0], parameter_names=params)
plot_fit(fit, plot_residuals=True, n_fit_points=10000, title='Circuito RL - Carica Induttore', save_path='./grafici/rl_carica_induttore.png')
dati_2 = latex_table_data(carica_tempi, carica_tensione, orientation='v', caption='dati carica induttore')
L_2 = fit.parameters['$\\tau$'] * R_eq
risultati_fit_2 = latex_table_fit(fit, orientation='v', caption='fit carica induttore')
#############################################################################################################################################################################


#RESISTENZA
def V_r(x, V_g, tau, c):
    return V_g * (1 - 2 * np.exp(-x/tau)) + c

scarica_tempi_resistenza = [0, 40, 60, 80, 100, 140, 180, 200, 240, 280, 320, 360, 400, 440, 480, 500, 540, 580, 600, 700] #microsecondi
scarica_tensione_resistenza = [552, 424, 368, 324, 284, 220, 172, 152, 120, 100, 80, 68, 56, 48, 44, 40, 36, 36, 32, 28] #millivolt
scarica_tempi = Measurement(scarica_tempi_resistenza, 1, magnitude=-6, unit='s', name='Tempo')
scarica_tensione = Measurement(scarica_tensione_resistenza, 2, magnitude=-3, unit='V', name='Tensione')
fit = perform_fit(scarica_tempi, scarica_tensione, V_r, [-0.300, 0.0010, 0.003], parameter_names=params)
plot_fit(fit, plot_residuals=True, n_fit_points=10000, title='Circuito RL - Scarica Resistenza', save_path='./grafici/rl_scarica_resistenza.png')
dati_3 = latex_table_data(scarica_tempi, scarica_tensione, orientation='v', caption='dati scarica resistenza')
risultati_fit_3 = latex_table_fit(fit, orientation='v', caption='fit scarica resistenza')
####################################################################################################################################################################################################
carica_tempi_resistenza = [3.06, 3.08, 3.10, 3.12, 3.14, 3.16, 3.18, 3.20, 3.22, 3.24, 3.26, 3.30, 3.33, 3.36, 3.39, 3.44, 3.47, 3.50, 3.54, 3.59, 3.64, 3.66, 3.67, 3.80] #millisecondi
carica_tensione_resistenza = [-572, -500, -436, -380, -336, -296, -260, -228, -200, -176, -160, -124, -108, -92, -80, -64, -56, -52, -44, -40, -36, -36, -32, -28] #millivolt
carica_tempi = Measurement(carica_tempi_resistenza, 0.02, magnitude=-3, unit='s', name='Tempo') - 0.00306
carica_tensione = Measurement(carica_tensione_resistenza, 2, magnitude=-3, unit='V', name='Tensione')
fit = perform_fit(carica_tempi, carica_tensione, V_r, [0.300, 0.0010, -0.003], parameter_names=params)
plot_fit(fit, plot_residuals=True, n_fit_points=10000, title='Circuito RL - Carica Resistenza', save_path='./grafici/rl_carica_resistenza.png')
dati_4 = latex_table_data(carica_tempi, carica_tensione, orientation='v', caption='dati carica resistenza')
risultati_fit_4 = latex_table_fit(fit, orientation='v', caption='fit carica resistenza')

pyperclip.copy (risultati_fit_1 + risultati_fit_2 + risultati_fit_3 + risultati_fit_4 + dati_1 + dati_2 + dati_3 + dati_4)
