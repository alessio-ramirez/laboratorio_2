import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

#FIGURA INTERFERENZA
def int_fabry (cos, A, d) :
    lambd = 632.8e-9 
    return A + (2*d/lambd)*cos

def int_fabry_perot (N, A, d):
    lambd = 632.8e-9
    return N * lambd / (2 * d) + A

l = Measurement(128, 0.5, magnitude=-2)
diam = [2.0, 3.6, 4.8, 5.7, 6.5]
diam = Measurement(diam, 0.1, -2, name='Diametro', unit='m')
cos = np.cos(np.arctan(diam/2/l))
cos.name, cos.unit = ('$\\cos(\\theta)$', 'rad')
N = Measurement ([5,4,3,2,1], name='$N$')
fit = perform_fit(N, cos, int_fabry_perot, method='odr')
plot_fit(fit, plot_residuals=True, title='', save_path='./grafici/fabry_perot_interferenza.pdf')
latex_table_data(diam, cos, N, orientation='v', caption='Dati Figura di Interferenza (Fabry-Perot)')
latex_table_fit(fit, orientation='v', caption='Risultati Interpolazione (Fabry-Perot)')

#CALIBRAZIONE MICROMETRO 
#nonio mosso sempre tra le stesse posizioni
#delta d misurato con calibro = 50 +- 50 micrometri
delta_n = [24, 23, 23, 24, 25, 25, 23]
delta_n = Measurement(delta_n, 1)
lambd = 632.8e-9
delta_d = lambd*delta_n/2
delta_d_m = weighted_mean(delta_d)
print(f"delta_m = {delta_d_m}")


