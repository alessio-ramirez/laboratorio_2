import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

#FIGURA INTERFERENZA
def int_fabry (cos, A, d) :
    lambd = 632.8 *10e-9 
    return A + (2*d/lambd)*cos

l = Measurement(128, 0.5, magnitude=-2)
diam = [2.0, 3.6, 4.8, 5.7, 6.5]
err_diam = 0.1
diam = Measurement(diam, err_diam, magnitude=-2)
cos= np.cos(np.arctan(diam/(2*l)))
k = Measurement([5,4,3,2,1])
fit = perform_fit(cos, k, int_fabry, method='odr')
plot_fit(fit, show_plot=True, show_params=True, show_stats=True)

#CALIBRAZIONE MICROMETRO 
#nonio mosso sempre tra le stesse posizioni
#delta d misurato con calibro = 50 +- 50 micrometri
delta_n = [24, 23, 23, 24, 25, 25, 23]
delta_n = Measurement(delta_n, 1)
lambd = 632.8 *10e-9
delta_d = lambd*delta_n/2
delta_d_m = weighted_mean(delta_d)
#print(delta_d_m)
