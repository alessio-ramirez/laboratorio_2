import numpy as np
import sys
import pyperclip
sys.path.append('../')
from burger_lib.guezzi import *

C_att = 10 #nano farad
C_att_err = C_att / 10 #errore 10% sia da misura multimetro sia da etichetta condensatore
C_att = Measurement(C_att, C_att_err, magnitude=-9, unit='F', name='capacità attesa')
R_nota = 100 #kohm
R_nota_err = 1/100 * R_nota
R_nota = Measurement(R_nota, R_nota_err, magnitude=3, unit='Ohm', name='R nota')
tau_att = R_nota * C_att

frequenza_generatore = 100 #hertz quindi circa T = 10 tau
parameter_names_ = ["$V_g$", "$\\tau$", "c"]

##############################################################################
#CONDENSATORE
def V_c(x, V_g, tau, c):
    return V_g * (1 - 2 * np.exp(-x/tau)) + c

scarica_tensione_condensatore = [292, 264, 228, 196, 164, 136, 104, 68, 32, 4, -20, -48, -68, -92, -116, -136, -156, -176, -196, -212, -236, -252, -260, -268, -280, -288] #milli volt
scarica_tempi_condensatore = [-800, -740, -680, -620, -560, -500, -420, -340, -240, -160, -80, 20, 100, 200, 320, 440, 560, 700, 860, 1020, 1300, 1540, 1720, 2000, 2380, 2820] #micro secondi, l''errore da 1000 aumenta
scarica_tempi_condensatore_err = [1 if num <= 999 else 10 for num in scarica_tempi_condensatore] #cifra significativa
scarica_tempi = Measurement(scarica_tempi_condensatore, scarica_tempi_condensatore_err, magnitude=-6, unit='s', name='Tempo')
scarica_tensione = Measurement(scarica_tensione_condensatore, 4, magnitude=-3, unit='V', name='Tensione')
fit = perform_fit(scarica_tempi, scarica_tensione, V_c, [-0.200, 0.001, 0.200], parameter_names=parameter_names_)
plot_fit(fit, n_fit_points=10000, plot_residuals=True, title='Circuito RC - Scarica Condensatore', save_path='./grafici/rc_scarica_condensatore.png')
dati_1 = latex_table_data(scarica_tempi, scarica_tensione, orientation='v', caption='dati scarica condensatore')
risultati_fit_1 = latex_table_fit(fit, orientation='v', caption='fit scarica condensatore')

carica_tensione_condensatore = [-296, -272, -228, -208, -184, -148, -128, -108, -88, -40, -20, 12, 40, 64, 80, 104, 132, 144, 164, 180, 196, 208, 228, 240, 252, 260, 272, 280, 284, 288] #milli volt
carica_tempi_condensatore = [4.22, 4.26, 4.34, 4.38, 4.42, 4.50, 4.54, 4.58, 4.64, 4.76, 4.82, 4.92, 5.02, 5.12, 5.20, 5.30, 5.46, 5.54, 5.66, 5.78, 5.94, 6.06, 6.32, 6.54, 6.80, 7.02, 7.38, 7.82, 8.28, 8.78] #milli secondi
carica_tensione = Measurement(carica_tensione_condensatore, 4, magnitude=-3, unit='V', name='Tensione') #errore della scala, cambiava di 4 in 4
carica_tempi = Measurement(carica_tempi_condensatore, 0.01, magnitude=-3, unit='s', name='Tempo') #cifra significativa
fit = perform_fit(carica_tempi, carica_tensione, V_c, [0.200, 0.001, 0.200], parameter_names=parameter_names_)
plot_fit(fit, n_fit_points=10000, plot_residuals=True, title='Circuito RC - Carica Condensatore', save_path='./grafici/rc_carica_condensatore.png')
dati_2 = latex_table_data(carica_tempi, carica_tensione, orientation='v', caption='dati carica condensatore')
risultati_fit_2 = latex_table_fit(fit, orientation='v', caption='fit carica condensatore')


##############################################################################
#RESISTENZA
def V_r(x, V_g, tau, c):
    return 2 * V_g * np.exp(-x/tau) + c

carica_tensione_resistenza = [576, 540, 492, 452, 412, 380, 344, 316, 288, 248, 224, 204, 184, 160, 124, 100, 84, 72, 68, 60, 52, 48, 36, 28, 24, 16, 12, 10] #milli volt
carica_tempi_resistenza = [20, 80, 180, 260, 340, 420, 520, 600, 700, 820, 920, 1020, 1120, 1240, 1460, 1680, 1800, 1960, 2080, 2200, 2320, 2440, 2600, 2800, 2960, 3160, 3400, 3600] #micro secondi
carica_tempi_resistenza_err = [1 if num <= 999 else 10 for num in carica_tempi_resistenza] #cifra significativa
carica_tensione = Measurement(carica_tensione_resistenza, 4, magnitude=-3, unit='V', name='Tensione')
carica_tempi = Measurement(carica_tempi_resistenza, carica_tempi_resistenza_err, magnitude=-6, unit='s', name='Tempo')
fit = perform_fit(carica_tempi, carica_tensione, V_c, [0.200, 0.001, 0.200], parameter_names=parameter_names_)
plot_fit(fit, n_fit_points=10000, plot_residuals=True, title='Circuito RC - Carica Resistenza', save_path='./grafici/rc_carica_resistenza.png')
dati_3 = latex_table_data(carica_tempi, carica_tensione, orientation='v', caption='dati carica resistenza')
risultati_fit_3 = latex_table_fit(fit, orientation='v', caption='fit carica resistenza')


scarica_tensione_resistenza = [-588, -552, -520, -480, -456, -428, -400, -388, -356, -348, -332, -312, -300, -276, -260, -240, -224, -220, -204, -184, -176, -164, -132, -116, -96, -88, -76, -64, -56, -48, -40, -36, -32, -24, -20, -16, -12, -8] #milli volt
scarica_tempi_resistenza = [5, 5.06, 5.12, 5.20, 5.24, 5.30, 5.36, 5.40, 5.48, 5.50, 5.54, 5.60, 5.64, 5.72, 5.78, 5.86, 5.90, 5.94, 6, 6.1, 6.16, 6.22, 6.40, 6.52, 6.72, 6.80, 6.92, 7.08, 7.22, 7.38, 7.52, 7.62, 7.78, 7.98, 8.28, 8.42, 8.66, 8.82] #milli secondi
scarica_tensione = Measurement(scarica_tensione_resistenza, 4, magnitude=-3, unit='V', name='Tensione')
scarica_tempi = Measurement(scarica_tempi_resistenza, 0.01, magnitude=-3, unit='s', name='Tempo')
fit = perform_fit(scarica_tempi, scarica_tensione, V_c, [0.200, 0.001, 0.200], parameter_names=parameter_names_)
plot_fit(fit, n_fit_points=10000, plot_residuals=True, title='Circuito RC - Scarica Resistenza', save_path='./grafici/rc_scarica_resistenza.png')
dati_4 = latex_table_data(scarica_tempi, scarica_tensione, orientation='v', caption='dati scarica resistenza')
risultati_fit_4 = latex_table_fit(fit, orientation='v', caption='fit scarica resistenza')

pyperclip.copy (risultati_fit_1 + risultati_fit_2 + risultati_fit_3 + risultati_fit_4 + dati_1 + dati_2 + dati_3 + dati_4)
