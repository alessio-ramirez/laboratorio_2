import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

err_angoli= 0.25 #sensibilità valida per tutti
theta_i = Measurement(142.5, err_angoli) #angolo per cui si vede raggio riflesso  
beta = Measurement(19.0, err_angoli) #angolo di cui ruotare la base del prisma per vedere il raggio riflesso dalla seconda faccia lucida
alfa = ((theta_i - beta)/2) *np.pi/180

#LEGGE DI CAUCHY
def Cauchy (l, A, B) :
    return A + B/(l**2)

#in ordine: giallo (media tra le due linee), verde, petrolio, blu, viola
angolo   = Measurement([48.5, 47.5, 47  , 47.5, 47.5], err_angoli)
angolo_0 = Measurement([97  , 95.5, 96.5,  98,   99], err_angoli)
ang_min_ = (angolo_0 - angolo)
ang_min = (angolo_0 - angolo) * np.pi / 180
n = np.sin((ang_min + alfa)/2)/np.sin(alfa/2)
n.name, n.unit = ('indice rifrazioni', '')
l = [578.01305, 546.9598, 435.8328, 434.75, 404.6563]
lambda_ = Measurement(l, magnitude=-9, name='$\\lambda$', unit='m')
#latex_table_data(lambda_, angolo, angolo_0, ang_min_, n)


fit_cauchy = perform_fit(lambda_, n, Cauchy)
plot_fit(fit_cauchy, save_path='./grafici/cauchy.pdf', plot_residuals=True, ylabel= 'n', title='Relazione indice di rifrazione-lunghezza d onda')
latex_table_fit(fit_cauchy, orientation='v')
A = fit_cauchy.parameters['A']
B = fit_cauchy.parameters['B']

#IDENTIFICAZIONE GAS IGNOTO

#in ordine: rosso profondo, arancione chiaro, azzurro, blu scuro
angolo =   Measurement([62.5,  63,   63.5,  63], err_angoli)
angolo_0 = Measurement([110.5, 111.5, 112.5, 113], err_angoli)
ang_min = (angolo_0 - angolo) * np.pi / 180
n = np.sin((ang_min + alfa)/2)/np.sin(alfa/2)
lambda_ignota = np.sqrt(B/(n-A))
#print(f"n={n}, A={A}, B={B}, angolo minima deviazione={ang_min}")
print(lambda_ignota)
