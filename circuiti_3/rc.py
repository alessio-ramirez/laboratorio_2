import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import numpy as np
from burger_lib.guezzi import *
import matplotlib.pyplot as plt

C_nota = Measurement(96, 1, magnitude=-9, unit='F', name='$C_{\\text{nota}}') # Capacità indicata sul condensatore
R_nota = Measurement(10, 0.1, magnitude=3, unit= 'ohm', name='$R_{\\text{nota}}') # Resistenza indicata sulla cassetta di resistenze
f_taglio = 1 / (2 * np.pi * R_nota * C_nota) # Frequenza di taglio ricavata

# Ampiezza del generatore (fissa in tutte le misure, l'abbiamo misurata ogni volta e rimaneva la stessa) misurata dall'oscilloscopio
amp_Vg = Measurement (4, 0.4, unit= 'V', name='$V_g$')

# Frequenza dell'onda sinusoidale presa dal generatore di funzioni
frequenza = [10, 15, 25, 50, 100, 120, 150, 200, 300, 450, 700, 1000, 1500, 2200, 3200, 4500, 6000, 7500, 10000, 15000, 20000, 30000] # hertz
frequenza = Measurement(frequenza, unit='Hz', name='Frequenza')
pulsazione = (frequenza * 2 * np.pi)
pulsazione.name, pulsazione.unit = ('$\\omega$', '') #pulsazione senza unità
# Ampiezza picco-picco ai capi del condensatore, misurato dall'oscilloscopio con MATH (sottrazione delle due tensioni)
amp_Vc_pp_values = [4, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.00, 4.00, 4.08, 4.04, 3.96, 3.76, 3.48, 3, 2.52, 2.12, 1.80, 1.40, 1.00, 0.760, 0.560] #volt
amp_Vc_pp_errors = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.2, 0.08, 0.4, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04] #volt error
amp_Vc_pp = Measurement(amp_Vc_pp_values, amp_Vc_pp_errors, unit='V', name='$V_C$')
# Ampiezza picco-picco ai capi della resistenza, misurato con l'oscilloscopio misurando direttamente la tensione con la sonda
amp_Vr_pp_values = [12.8, 18.0, 28.8, 57.6, 112, 134, 168, 228, 336, 504, 784, 1090, 1580, 2140, 2700, 3180, 3480, 3640, 3800, 3920, 3980, 4040] #millivolt
amp_Vr_pp_errors = [0.4, 0.4, 0.4, 0.8, 2, 2, 2, 4, 2, 2, 2, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40] #millivolt error
amp_Vr = Measurement(amp_Vr_pp_values, amp_Vr_pp_errors, magnitude=-3, unit='V', name='$V_R$')
# Fase Vc misurata dall'oscilloscopio (automaticamente)
fase_Vc_values = [178, 176, 179, 177, 177, 180, 175, 177, 174, 172, 170, 164, 157, 149, 139, 128, 120, 115, 108, 103, 102, 94] #gradi
fase_Vc_errors = [2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #gradi error
fase_Vc = Measurement(fase_Vc_values, fase_Vc_errors) / 180 * np.pi
fase_Vc.name, fase_Vc.unit = ('$\\arg(H_C)$', 'rad')
# Fase Vr misurata dall'oscilloscopio (automaticamente)
fase_Vr_values = [92.1, 89.6, 88.9, 90.4, 88.2, 88.6, 88.0, 87.0, 84.9, 84.2, 79.6, 74.5, 67.6, 59.3, 47.9, 38.8, 31.1, 25.9, 19.4, 13.5, 10.1, 6.49] #gradi
fase_Vr_errors = [2, 1, 0.4, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05] #gradi error
fase_Vr = Measurement(fase_Vr_values, np.array(fase_Vr_errors)*3) / 180 * np.pi
fase_Vr.name, fase_Vr.unit = ('$\\arg(H_R)$', 'rad')

#funzioni

def fit_func_RC_lowpass(omega, C):
    R = R_nota.value # Use nominal value of R
    return 1.0 / np.sqrt(1.0 + (omega * R * C)**2)

def fase_Hc(omega, C, k):
    R = R_nota.value 
    return  k - np.arctan(omega*R*C) #k atteso 0

def fit_func_RC_highpass(omega, C):
    R = R_nota.value
    term = omega * R * C
    return term / np.sqrt(1.0 + term**2)

def fase_Hr(omega, C, k):
    R = R_nota.value
    return k - np.arctan (omega*R*C) #k atteso pi/2

ampiezza_Hc = amp_Vc_pp / amp_Vg
ampiezza_Hc.name = '$|H_C|$'
ampiezza_Hr = amp_Vr / amp_Vg
ampiezza_Hr.name = '$|H_R|$'
data_label = 'Dati Sperimentali'

fit_Hc = perform_fit(pulsazione, ampiezza_Hc, fit_func_RC_lowpass, p0=[C_nota.value], method='minuit', minuit_limits={'C': (1e-15, None)})
main_ax, residual_ax = plot_fit(fit_Hc, plot_residuals=True, title='Circuito RC - $|H_C(\\omega)|$', data_label=data_label)
main_ax.set_xscale('log')
main_ax.set_yscale('log')
residual_ax.set_xscale('log')
#plt.savefig('./grafici/rc_hc.pdf')

fit_Hr = perform_fit(pulsazione, ampiezza_Hr, fit_func_RC_highpass, p0=[C_nota.value], method='minuit', minuit_limits={'C': (1e-15, None)},)
main_ax, residual_ax = plot_fit(fit_Hr, n_fit_points=10000, plot_residuals=True, title='Circuito RC - $|H_R(\\omega)|$', data_label=data_label)
main_ax.set_xscale('log')
main_ax.set_yscale('log')
residual_ax.set_xscale('log')
#plt.savefig('./grafici/rc_hr.pdf')

fit_fase_Hc = perform_fit(pulsazione, fase_Vc, fase_Hc, p0=[C_nota.value, 0.0], method='minuit', minuit_limits={'C': (1e-12, 1), 'k':(0, 10)})
main_ax, residual_ax = plot_fit(fit_fase_Hc, n_fit_points=10000, plot_residuals=True, title='Circuito RC - $\\arg(H_C(\\omega))$', data_label=data_label)
main_ax.set_xscale('log')
residual_ax.set_xscale('log')
#plt.savefig('./grafici/rc_fase_hc.pdf')

fit_fase_Hr = perform_fit(pulsazione, fase_Vr, fase_Hr, p0=[C_nota.value, np.pi/2], method='minuit', minuit_limits={'C': (1e-12, 1e-6), 'k':(0, 10)})
main_ax, residual_ax = plot_fit(fit_fase_Hr, n_fit_points=10000, plot_residuals=True, title='Circuito RC - $\\arg(H_R(\\omega))$', data_label=data_label)
main_ax.set_xscale('log')
residual_ax.set_xscale('log')
#plt.savefig('./grafici/rc_fase_hr.pdf')

#data

latex_data = latex_table_data(frequenza, amp_Vg, amp_Vc_pp, amp_Vr, fase_Vc, fase_Vr,
                              orientation='v', sig_figs_error=1, caption='Dati Sperimentali per il Circuito RC')
print(latex_data)
fit_names = ['$|H_C|$', '$|H_R|$', '$\\arg(H_C)$', '$\\arg(H_R)$']
latex_fit = latex_table_fit(fit_Hc, fit_Hr, fit_fase_Hc, fit_fase_Hr,
                            fit_labels=fit_names, param_labels={'C': '$C$', 'k': '$k$'}, caption='Risultati dei Fit per il Circuito RC')
print(latex_fit)

# --- Test di Compatibilità tra i valori di C ---
print("\n--- Test di Compatibilità tra i valori di C ---")

# Estrarre i valori di C (come oggetti Measurement) da ogni fit
C_values_from_fits = {
    fit_names[0]: fit_Hc.parameters['C'],
    fit_names[1]: fit_Hr.parameters['C'],
    fit_names[2]: fit_fase_Hc.parameters['C'],
    fit_names[3]: fit_fase_Hr.parameters['C'],
}
print(test_comp(fit_Hr.parameters['C'], fit_Hc.parameters['C']))
alpha_compat = 0.05 # Livello di significatività per il test Z
compatibility_results_storage = {} # Dizionario per memorizzare i risultati (z_score, compatibile_bool)

# Eseguire i test di compatibilità per ogni coppia unica di fit
for i in range(len(fit_names)):
    for j in range(i + 1, len(fit_names)): # Considera solo j > i per evitare duplicati e auto-confronti
        name1 = fit_names[i]
        name2 = fit_names[j]
        
        c1 = C_values_from_fits[name1]
        c2 = C_values_from_fits[name2]
        
        # Esegui il test di compatibilità (assumendo indipendenza tra i fit)
        test_result = test_comp(c1, c2, alpha=alpha_compat, assume_correlated=False)
        
        # Salva i risultati (Z-score e boolean di compatibilità)
        compatibility_results_storage[(name1, name2)] = (test_result['z_score'], test_result['compatible'])
        # Per simmetria nella tabella, salva anche il risultato inverso
        compatibility_results_storage[(name2, name1)] = (test_result['z_score'], test_result['compatible'])

# --- Generazione Tabella LaTeX per Compatibilità dei valori di C ---
latex_compat_lines = []
latex_compat_lines.append("\\begin{table}[htbp]")
latex_compat_lines.append("\\centering")
# Specifiche colonne: una colonna 'l' per le etichette di riga, e N colonne 'c' per i fit
col_spec = "|l|" + "c" * len(fit_names) + "|"
latex_compat_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
latex_compat_lines.append("\\hline")

# Riga di intestazione con i nomi dei fit
# I nomi dei fit sono già formattati per LaTeX (es. '$|H_C|$')
header_row_content = ["Fit Confrontato"] + [f"{fn}" for fn in fit_names]
latex_compat_lines.append(" & ".join(header_row_content) + " \\\\\\hline\\hline")

# Righe della tabella
for i in range(len(fit_names)):
    current_fit_name_row = fit_names[i]
    row_cells_content = [f"{current_fit_name_row}"] # La prima cella è l'etichetta della riga

    for j in range(len(fit_names)):
        current_fit_name_col = fit_names[j]
        if i == j:
            row_cells_content.append("---") # Diagonale (auto-confronto)
        else:
            # Recupera il risultato del test di compatibilità
            z_score, compatible = compatibility_results_storage[(current_fit_name_row, current_fit_name_col)]
            # Simbolo per compatibilità: checkmark se compatibile, x se non compatibile
            compat_symbol = "\\checkmark" if compatible else "\\times"
            # Usa \shortstack per celle multi-riga in LaTeX
            row_cells_content.append(f"\\shortstack{{Z={z_score:.2f} \\\\ {compat_symbol}}}")
            
    latex_compat_lines.append(" & ".join(row_cells_content) + " \\\\\\hline")

latex_compat_lines.append("\\end{tabular}")
# Caption della tabella
caption_text = (f"Risultati dei test di compatibilità (Z-score e valutazione a $\\alpha={alpha_compat:.2f}$) "
                f"tra i valori del parametro $C$ ottenuti dai diversi fit per il circuito RC. "
                f"Si assume indipendenza tra i risultati dei fit. "
                f"{{\\footnotesize (Z: Z-score; $\\checkmark$: Compatibile; $\\times$: Non Compatibile)}}")
latex_compat_lines.append(f"\\caption{{{caption_text}}}")
latex_compat_lines.append("\\label{tab:rc_compatibilita_C}") # Etichetta per riferimenti incrociati
latex_compat_lines.append("\\end{table}")

# Stampa la tabella LaTeX generata
latex_compatibility_table_C = "\n".join(latex_compat_lines)
print("\n" + latex_compatibility_table_C)
b = list(C_values_from_fits.values())
a = weighted_mean(b)
print(b)
print(a.to_eng_string())
