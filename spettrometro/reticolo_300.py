import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

#reticolo 300 righe al mm

#calibrazione fatta valutando la simmetria dei massimi di ordine 2 rispetto al massimo di ordine 0
#calcoliamo angolo totale tra i due max e poi dividiamo per ridurre errore
k = 2
err_ang = 0.25 # gradi
theta_0 = Measurement(84.5, err_ang)
theta_dx = Measurement(105.25, err_ang)
theta_sx = Measurement(64.25, err_ang)
theta = ((theta_dx - theta_sx)/2)*np.pi/180 #in radianti
latex_table_data (theta_dx, theta_sx, theta*180/np.pi, caption='Posizione angolare massimi di interferenza', orientation='h' )
lambda_nota = 588.995 * 10**(-9) #considerata senza errore, lunghezza d'onda della riga più intensa

d = k*lambda_nota/np.sin(theta)
d= Measurement(3.36, 0.05, magnitude=-6)
d_att = Measurement(0.001/300) #reticolo con 300 righe al mm

#valutare la posizione di altre righe spettrali e ricavarne la lunghezza d'onda
k1=1
k2=2
k3=3
#arancione1, verde1
angolo_dx_1 = [94.5, 94]
angolo_sx_1 = [74.5, 74.75]
#verde2
angolo_dx_2 = [104.25]
angolo_sx_2 = [65]
#arancione3
angolo_dx_3 = [113.75]
angolo_sx_3 = [51]

theta_1 = ((np.array(angolo_dx_1) - np.array(angolo_sx_1))/2)
theta_2 = ((np.array(angolo_dx_2) - np.array(angolo_sx_2))/2)
angolo_1 = Measurement(theta_1, err_ang)*np.pi/180
angolo_2 = Measurement(theta_2, err_ang)*np.pi/180
lambda_s_1 = d*np.sin(angolo_1)/k1
lambda_s_2 = d*np.sin(angolo_2)/k2
righe = {'lambda':[lambda_s_1.value, lambda_s_2.value], 'errore': [lambda_s_1.error, lambda_s_2.error], 'colori': ['yellow', 'lime', 'lime']}
#print (lambda_s_1, lambda_s_2, lambda_s_3)

def plot_spettro_colorato(righe, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Impostazioni grafiche
    ax.set_xlabel('Lunghezza d\'onda (nm)', fontsize=12)
    ax.set_title('Righe emissioni sodio', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Altezza fissa per tutte le righe
    intensita_fissa = 1.
    
    # Plot delle righe con colori e errori
    for lam, err, colore in zip(righe['lambda'], righe['errore'], righe['colori']):
        # Linea verticale colorata
        ax.vlines(lam, 0, intensita_fissa, color=colore, linestyle='-', linewidth=2, alpha=0.7, label=f'{lam} nm')
        # Barra di errore orizzontale (nera per contrasto)
        ax.errorbar(lam, intensita_fissa, xerr=err, fmt='none', ecolor='black', capsize=4, elinewidth=1)
    
    # Legenda e ottimizzazione layout
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    # Salva o mostra
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_spettro_colorato(righe, save_path='./grafici/spettro_colorato.png')






#IDENTIFICAZIONE GAS IGNOTO
#in ordine: giallo senape, rosso intenso, azzurro, blu scuro
k = 1
angolo_sx = [76, 74.75, 77.75, 78.25]
angolo_dx = [96, 97.25, 94.25, 94]
theta = ((np.array(angolo_dx) - np.array(angolo_sx))/2) #in radianti
angolo = Measurement(theta, err_ang)*np.pi / 180
lambda_ignoto = d*np.sin(angolo)/k
#print(lambda_ignoto)




