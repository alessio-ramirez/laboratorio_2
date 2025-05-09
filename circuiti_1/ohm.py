import numpy as np
import matplotlib.pyplot as plt
from guezzi import create_dataset, test_comp, perform_fit
import scipy.stats as stats

#definisco la funzione di fit
def linear_func(x, a, b):
    return a + b * x

R_atteso = 10**6 #come scritto sulla strumentazione
R_atteso_err = 1/100 * R_atteso
I_err = 0.03*10**(-6)
V_err = 0.01
#configurazione 2 (voltometro in parallelo con generatore)
#array di dati
I_2 = np.array([0.03, 1.04, 2.02, 3.03, 4.03, 5.02, 6.03, 7.07, 8.03, 9.03, 10.07, 11.08, 12.02, 13.03, 14.04, 15.04, 16.05, 17.04, 18.04, 19.06, 20.03]) * 10**(-6)
V_2 = np.array([0.01, 1.02, 2.00, 3.01, 4.00, 5.00, 6.01, 7.05, 8.00, 9.00, 10.04, 11.04, 11.99, 13.00, 14.00, 15.01, 16.02, 17.01, 18.00, 19.03, 20.00])
I_2_ = create_dataset(I_2, I_err)
V_2_ = create_dataset(V_2, V_err)
#latex_table("correnti", I_2_, "tensioni", V_2_, orientation="v")


#spacchetto i risultati
#(a, R), (da, R_err), chi, dof = perform_fit(I_2_, V_2_, linear_func, p0=[1,1000000], chi_square=True)
#R_bias = lambda V_2, I_2:(V_2 - I_2*1.8)/(I_2- V_2/10596157 + 1.8/10596157 * I_2)
#print(error_prop(R_bias, [V_2_, I_2_]))
#print(chi, dof)
#print (a, R, da, R_err)
#test di compatibilità, mettere il valore seguito dall'errore cosi
#test_comp(R, R_err, R_atteso, R_atteso_err)

#configurazione 1 (voltomeytro in parallelo con resistenza)
#per tensioni alte la resistenza misurata risulta più alta della resistenza effettiva perchè la corrente si sdoppia tra la resistenza scelta e quella del voltmetro
I = np.array ([0.06, 1.14, 2.22, 3.31, 4.43, 5.51, 6.75, 7.75, 8.86, 9.97, 11.04, 12.16, 13.22, 14.34, 15.48, 16.53, 17.62, 18.73, 19.84, 20.93, 22.04]) * 10**(-6)
V = np.array ([0.013, 1.010, 2.005, 3.002, 4.029, 5.009, 6.159, 7.02, 8.02, 9.02, 10.01, 11.03, 12.00, 13.01, 14.04, 15.00, 16.00, 17.00, 18.01, 19.00, 20.01])
I_1 = create_dataset(I, I_err)
V_1 = create_dataset(V, V_err)
#R_bias = (V - (I- V/10596157)*1.8) / (I - V/10596157)
#latex_table("correnti", I_1, "tensioni", V_1, orientation="v")
#latex_table("$I_1$", I_1, "$V_1$", V_1, "$I_2$", I_2_, "$V_2$", V_2_, orientation="v")

"""
create_best_fit_line(I_1, V_1, I_2_, V_2_, func=linear_func, p0=[[1,1000000], [1, 1000000]], # Guess iniziali NON obbligatorio
                    xlabel="Correnti (A)",
                    ylabel="Tensioni (V)", # Titoli condivisi
                    title="Legge di Ohm, $1MΩ$", 
                    label_fit=["configurazione 1", "configurazione 2"], # Nome delle linee
                    together=True) # Plotta separatamente i grafici
"""
#(a, R), (da, R_err), chi, dof = perform_fit(I_1, V_1, linear_func, p0=[1,1000000], chi_square=True) #hanno gli stessi nomi ma ormai i risultati di prima sono gia stati printati
#test_comp(R, R_err, R_atteso, R_atteso_err)
#print(chi, dof)
#print(a, R, da, R_err)


############################################################
R_atteso = 10 #ohm
R_atteso_err = 1/100 * R_atteso
V_err = 0.001 #cambiata la sensibilità dello strumento

#configurazione 1
I = {0.16830:0.00002, 45.591:0.00001, 95.51:0.01, 145.61:0.02, 190.02:0.01, 239.41:0.01, 284.45:0.01, 334.04:0.01, 378.75:0.1, 424.04:0.05, 473.20:0.1}
V = np.array([0.018, 0.498, 1.006, 1.535, 2.003, 2.527, 3.005, 3.535, 4.014, 4.502, 5.037])
I_2 = create_dataset(I, magnitude=-3)
V_2 = create_dataset(V, 0.001)
#(a, R), (da, R_err), chi, dof = perform_fit(I_2, V_2, linear_func, p0=[1,10], chi_square=True)
#print(chi, dof)
#test_comp(R_atteso, R_atteso_err, R, R_err)
#print(a, R, da, R_err)

#configurazione 2
I = {0.12039:0.00002, 59.20:0.01, 112.29:0.01, 168.17:0.01, 223.94:0.01, 279.07:0.01, 335.41:0.03, 392.40:0.05, 444.00:0.04, 499.20:0.1}
V = np.array([0.001, 0.530, 1.005, 1.506, 2.006, 2.502, 3.011, 3.528, 4.000, 4.505])
I_1 = create_dataset(I, magnitude=-3)
V_1 = create_dataset(V, 0.001)
#(a, R), (da, R_err), chi, dof = perform_fit(I_1, V_1, linear_func, p0=[1,10], chi_square=True)
#print(chi, dof)
#print(a, R, da, R_err)
#test_comp(R_atteso, R_atteso_err, R, R_err)
#latex_table("correnti", I_1, "tensioni", V_1, orientation="v")
#latex_table("$I_1$", I_1, "$V_1$", V_1, "$I_2$", I_2, "$V_2$", V_2, orientation="v")


"""
create_best_fit_line(I_1, V_1, I_2, V_2, func=linear_func, p0=[[1,1000000], [1, 1000000]], # Guess iniziali NON obbligatorio
                    xlabel="Correnti (A)",
                    ylabel="Tensioni (V)", # Titoli condivisi
                    title="Legge di Ohm, $10Ω$", 
                    label_fit=["configurazione 2", "configurazione 1"], # Nome delle linee
                    together=True) # Plotta separatamente i grafici"
"""