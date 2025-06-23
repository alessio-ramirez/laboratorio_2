import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

def righello (N, lambd) :
    d = 1e-3 #passo righello
    return N*lambd/d

l = 124 #distanza sorgente-muro
l_err = 0.1 #errore grande perchè l'area illuminata dal righello non è un punto ma un tratto
l = Measurement(l, l_err, magnitude=-2)
diam_0 = 4.7
diam_err = 0.2
diam_0 = Measurement(diam_0, diam_err, magnitude=-2)
theta_inc = np.arctan(diam_0/l)
diam = [7, 8.5, 9.6, 11, 12] #2*altezza massimi rispetto 
diam = Measurement (diam, diam_err, magnitude=-2, name="$r_N$", unit="m")
theta_n = np.arctan(diam/l)
y = np.cos(theta_inc) - np.cos(theta_n)
y.name, y.unit = ("$\\cos(\\theta_i) - \\cos(\\theta_N)", "")
N = Measurement([1, 2 , 3, 4, 5], name="$N$", unit="")
latex_table_data(diam, y, N)

fit = perform_fit(N, y, righello, method='minuit')
plot_fit(fit, plot_residuals=True, title="", save_path="./grafici/righello.pdf")
latex_table_fit(fit, caption="Risultati del Fit - Interferenza del Righello", orientation="v")