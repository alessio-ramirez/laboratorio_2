import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
#determinare passo dei reticoli valutando posizione angolare seconda riga gialla del sodio

#600 RIGHE AL MM
#calibrazione fatta valutando la simmetria dei massimi di ordine 2 rispetto al massimo di ordine 0
#calcoliamo angolo totale tra i due max e poi dividiamo per ridurre errore
k = 2
err_ang = 0.5 #errore maggiore perchè aumentando N numero di fenditure illuminate aumenta la loro larghezza
lambda_nota = 588.995e-9 #considerata senza errore, lunghezza d'onda della riga più intensa (giallo)
theta_0 = Measurement(82, err_ang)
theta_dx = Measurement(127, err_ang)
theta_sx = Measurement(37.5, err_ang)
theta = ((theta_dx - theta_sx)/2)*np.pi/180 #in radianti

d = k*lambda_nota/np.sin(theta)
d_att = Measurement(0.001/600) 
print (d)

#1200 RIGHE AL MM
k = 1 #usanso il reticolo con N grande si riescono a vedere sono i massimi del primo ordine (molto lontani dal centro)
err_ang = 0.5 #gradi
lambda_nota = 588.995 * 10**(-9) #considerata senza errore, lunghezza d'onda della riga più intensa (giallo)
theta_0 = Measurement(82, err_ang)
theta_dx = Measurement(127, err_ang)
theta_sx = Measurement(37.5, err_ang)
theta = ((theta_dx - theta_sx)/2)*np.pi/180 #in radianti

d = k*lambda_nota/np.sin(theta)
d_att = Measurement(0.001/1200) 
