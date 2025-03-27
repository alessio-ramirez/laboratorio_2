from guezzi import *
import numpy as np

R_nota = create_dataset(500, 1) #ohm
err_tensione = 2 #millivolt

#INDUTTORE
def V_l(x, V_g, tau, c):
    return 2 * V_g * np.exp(-x/tau) + c
scarica_tempi_induttore = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 92, 100, 108, 116, 132, 148, 160, 180, 192, 208, 224, 236, 252, 264, 280, 304, 324, 364, 412, 448, 488, 516, 556, 604] #microsecondi
scarica_tensione_induttore = [-248, -224, -196, -172, -148, -128, -108, -88, -68, -52, -12, 0, 16, 28, 52, 72, 88, 108, 120, 132, 144, 152, 164, 168, 176, 188, 196, 208, 220, 224, 228, 232, 236, 240] #millivolt
scarica_tempi = create_dataset(scarica_tempi_induttore, 1, magnitude=-6)
scarica_tensione = create_dataset(scarica_tensione_induttore, 1, magnitude=-3)
#create_best_fit_line(scarica_tempi, scarica_tensione, func=V_l, show_chi_squared=True, show_fit_params=True, p0=[[0.300, 0.0010, 0.300]])
parametri = perform_fit(scarica_tempi, scarica_tensione, func=V_l, p0=[0.300, 0.0010, 0.300])['parameters']
tau = create_dataset(parametri['value'][1], parametri['error'][1])
L = error_prop(lambda x, y : x*y, tau, R_nota)
print(L)
carica_tempi_induttore = [3.05, 3.07, 3.09, 3.11, 3.13 , 3.16, 3.18, 3.21, 3,22, 3.24, 3.29, 3.32, 3.36, 3.40, 3.44, 3.49, 3.54, 3.59, 3.62, 3.77] #millisecondi
carica_tensione_induttore = [248, 212, 152, 96, 48, -8, -40, -80, -92, -112, -152, -172, -192, -208, -216, -228, -232, -240, -240, -248] #millivolt

#RESISTORE
scarica_tempi_resistenza = [0, 40, 60, 80, 100, 140, 180, 200, 240, 280, 320, 360, 400, 440, 480, 500, 540, 580, 600, 700] #micro secondi
scarica_tensione_resistenza = [552, 424, 368, 324, 284, 220, 172, 152, 120, 100, 80, 68, 56, 48, 44, 40, 36, 36, 32, 28] #millivolt
carica_tempi_resistenza = [3.06, 3.08, 3.10, 3.12, 3.14, 3.16, 3.18, 3.20, 3.22, 3.24, 3.26, 3.30, 3.33, 3.36, 3.39, 3.44, 3.47, 3.50, 3.54, 3.59, 3.64, 3.66, 3.67, 3.80] #millisecondi
carica_tensione_resistenza = [-572, -500, -436, -380, -336, -296, -260, -228, -200, -176, -160, -124, -108, -92, -80, -64, -56, -52, -44, -40, -36, -36, -32, -28] #millivolt
