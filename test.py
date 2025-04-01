#from guezzi import *
import numpy as np
from burger_lib.guezzi import *
import matplotlib.pyplot as plt

R_nota = Measurement(500, 1, unit='Ohm', name='resistenza nota')
err_tensione = 2 #millivolt
R_osc = Measurement(1, magnitude=6, unit='Ohm', name='resistenza oscilloscopio')
R_eq = (R_nota * R_osc)/(R_nota + R_osc)

#############################################################################################################################################################################################################################à

#INDUTTORE
def V_l(x, k, tau, c):
    return k * np.exp(-x/tau) + c

scarica_tempi_induttore = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 92, 100, 108, 116, 132, 148, 160, 180, 192, 208, 224, 236, 252, 264, 280, 304, 324, 364, 412, 448, 488, 516, 556, 604] #microsecondi
scarica_tensione_induttore = [-248, -224, -196, -172, -148, -128, -108, -88, -68, -52, -12, 0, 16, 28, 52, 72, 88, 108, 120, 132, 144, 152, 164, 168, 176, 188, 196, 208, 220, 224, 228, 232, 236, 240] #millivolt
scarica_tempi = Measurement(scarica_tempi_induttore, 1, magnitude=-6)
scarica_tensione = Measurement(scarica_tensione_induttore, 2, magnitude=-3)
statistica = perform_fit(scarica_tempi, scarica_tensione, V_l, [0.500, 0.0010, -1], ['k', 'tau', 'c'])
plot_fit(statistica, plot_residuals=True, n_fit_points=10000, xlabel='tempi', show_stats=True, show_plot=False, show_params=True)
latex_table_fit(statistica, orientation='v')

"""
tau = create_dataset(parametri['value'][1], parametri['error'][1])
L = error_prop(lambda x, y : x*y, tau, R_eq)
tau = 0.00014075
tau_err = 5.01117404 * 10 ** -7
L = 0.07033969
L_err = 0.00028721
chi_quadro = 6.063
chi_quadro_ridotto = 0.1956
gdl = 31

##############################################################################################################################################################################
carica_tempi_induttore = [3.05, 3.07, 3.09, 3.11, 3.13 , 3.16, 3.18, 3.21, 3.22, 3.24, 3.29, 3.32, 3.36, 3.40, 3.44, 3.49, 3.54, 3.59, 3.62, 3.77] #millisecondi
carica_tensione_induttore = [248, 212, 152, 96, 48, -8, -40, -80, -92, -112, -152, -172, -192, -208, -216, -228, -232, -240, -240, -248] #millivolt
carica_tensione_induttore = carica_tensione_induttore + np.full_like(carica_tensione_induttore, 248)
carica_tempi_induttore = carica_tempi_induttore - np.full_like(carica_tempi_induttore, 3.05)
carica_tempi = create_dataset(carica_tempi_induttore, 0.01, magnitude=-3)
carica_tensione = create_dataset(carica_tensione_induttore, 2, magnitude=-3)
#create_best_fit_line(carica_tempi, carica_tensione, func=V_l, show_chi_squared=True, show_fit_params=True, p0=[[-0.500, 0.00010, 1]])
parametri = perform_fit(carica_tempi, carica_tensione, func=V_l, p0=[0.500, 0.00010, 1])['parameters']

#############################################################################################################################################################################


#RESISTENZA
def V_r(x, V_g, tau, c):
    return V_g * (1 - 2 * np.exp(-x/tau)) + c

scarica_tempi_resistenza = [0, 40, 60, 80, 100, 140, 180, 200, 240, 280, 320, 360, 400, 440, 480, 500, 540, 580, 600, 700] #microsecondi
scarica_tensione_resistenza = [552, 424, 368, 324, 284, 220, 172, 152, 120, 100, 80, 68, 56, 48, 44, 40, 36, 36, 32, 28] #millivolt
scarica_tempi = create_dataset(scarica_tempi_resistenza, 1, magnitude=-6)
scarica_tensione = create_dataset(scarica_tensione_resistenza, 2, magnitude=-3)
#create_best_fit_line(scarica_tempi, scarica_tensione, func=V_r, show_chi_squared=True, show_fit_params=True, p0=[[0.300, 0.00010, 0.300]])

####################################################################################################################################################################################################
carica_tempi_resistenza = [3.06, 3.08, 3.10, 3.12, 3.14, 3.16, 3.18, 3.20, 3.22, 3.24, 3.26, 3.30, 3.33, 3.36, 3.39, 3.44, 3.47, 3.50, 3.54, 3.59, 3.64, 3.66, 3.67, 3.80] #millisecondi
carica_tensione_resistenza = [-572, -500, -436, -380, -336, -296, -260, -228, -200, -176, -160, -124, -108, -92, -80, -64, -56, -52, -44, -40, -36, -36, -32, -28] #millivolt
carica_tensione_resistenza = carica_tensione_resistenza+ np.full_like(carica_tensione_resistenza, 572)
carica_tempi_resistenza = carica_tempi_resistenza - np.full_like(carica_tempi_resistenza, 3.06)
carica_tempi = create_dataset(carica_tempi_resistenza, 0.02, magnitude=-3)
carica_tensione = create_dataset(carica_tensione_resistenza, 2, magnitude=-3)
create_best_fit_line(carica_tempi, carica_tensione, func=V_r, show_chi_squared=True, show_fit_params=True, p0=[[0.300, 0.00010, 0.003]])
"""