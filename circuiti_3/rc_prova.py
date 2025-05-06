# --- START OF FILE rc.py ---

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import numpy as np
from burger_lib.guezzi import *
import matplotlib.pyplot as plt

# --- Data Definition ---
C_nota = Measurement(96, 1, magnitude=-9, unit='F', name='C nota') # Capacità indicata sul condensatore
R_nota = Measurement(10, 0.1, magnitude=3, unit='ohm', name='R nota') # Resistenza indicata sulla cassetta di resistenze
f_taglio_teorica = 1/(2*np.pi * R_nota * C_nota) # Frequenza di taglio teorica

# Ampiezza del generatore (misurata ogni volta, usiamo la media o valore più rappresentativo)
# NOTE: The original text implies Vg might vary slightly. Using a single value assumes it was constant enough,
# or you should use an array if it varied significantly across frequency points. Let's use the single value provided.
amp_Vg = Measurement (4, 0.4, unit= 'V', name='Vg') # Using single value for amplitude

# Frequenza
frequenza_vals = [10, 15, 25, 50, 100, 120, 150, 200, 300, 450, 700, 1000, 1500, 2200, 3200, 4500, 6000, 7500, 10000, 15000, 20000, 30000] # hertz
frequenza = Measurement(frequenza_vals, unit='Hz', name='Frequenza')
pulsazione = frequenza * 2 * np.pi
pulsazione.name = "$\\omega$" # Use LaTeX for name
pulsazione.unit = "rad/s"

# Ampiezza Vc (picco-picco)
amp_Vc_pp_values = [4, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.00, 4.00, 4.08, 4.04, 3.96, 3.76, 3.48, 3, 2.52, 2.12, 1.80, 1.40, 1.00, 0.760, 0.560] #volt
amp_Vc_pp_errors = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.2, 0.08, 0.4, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04] #volt error
amp_Vc_pp = Measurement(amp_Vc_pp_values, amp_Vc_pp_errors, unit='V', name='$V_C$(pp)')

# Ampiezza Vr (picco-picco)
amp_Vr_pp_values = [12.8, 18.0, 28.8, 57.6, 112, 134, 168, 228, 336, 504, 784, 1090, 1580, 2140, 2700, 3180, 3480, 3640, 3800, 3920, 3980, 4040] #millivolt
amp_Vr_pp_errors = [0.4, 0.4, 0.4, 0.8, 2, 2, 2, 4, 2, 2, 2, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40] #millivolt error
amp_Vr_pp = Measurement(amp_Vr_pp_values, amp_Vr_pp_errors, magnitude=-3, unit='V', name='$V_R$(pp)')

# Fase Vc (rispetto a Vg)
fase_Vc_values = [178, 176, 179, 177, 177, 180, 175, 177, 174, 172, 170, 164, 157, 149, 139, 128, 120, 115, 108, 103, 102, 94] #gradi
# NOTE: Error increased by factor 3 as requested in original text, now applied here.
fase_Vc_errors = np.array([2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 3 #gradi error * 3
fase_Vc_deg = Measurement(fase_Vc_values, fase_Vc_errors, unit='deg', name='$\\Delta\\phi_C$')
# Convert to radians for fitting, but keep degrees potentially for tables/plotting if needed
fase_Vc = fase_Vc_deg * (np.pi / 180.0)
fase_Vc.unit = 'rad'

# Fase Vr (rispetto a Vg)
fase_Vr_values = [92.1, 89.6, 88.9, 90.4, 88.2, 88.6, 88.0, 87.0, 84.9, 84.2, 79.6, 74.5, 67.6, 59.3, 47.9, 38.8, 31.1, 25.9, 19.4, 13.5, 10.1, 6.49] #gradi
# NOTE: Error increased by factor 3 as requested in original text, now applied here.
fase_Vr_errors = np.array([2, 1, 0.4, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05]) * 3 #gradi error * 3
fase_Vr_deg = Measurement(fase_Vr_values, fase_Vr_errors, unit='deg', name='$\\Delta\\phi_R$')
# Convert to radians for fitting
fase_Vr = fase_Vr_deg * (np.pi / 180.0)
fase_Vr.unit = 'rad'

# --- Model Functions ---
# Note: R is fixed to R_nota.value in these models for fitting C

def fit_func_RC_lowpass(omega, C):
    """Magnitude of RC Low-Pass Filter Transfer Function (|Vc/Vg|)"""
    R = R_nota.value # Use nominal value of R
    # Vc/Vg = 1 / sqrt(1 + (wRC)^2)
    return 1.0 / np.sqrt(1.0 + (omega * R * C)**2)

def fit_func_RC_highpass(omega, C):
    """Magnitude of RC High-Pass Filter Transfer Function (|Vr/Vg|)"""
    R = R_nota.value # Use nominal value of R
    # Vr/Vg = wRC / sqrt(1 + (wRC)^2)
    term = omega * R * C
    return term / np.sqrt(1.0 + term**2)

def fase_Hc(omega, C, k):
    """Phase of RC Low-Pass Filter Transfer Function (Vc vs Vg)"""
    # arg(Hc) = -arctan(wRC) + k (offset k is fitted)
    R = R_nota.value
    return k - np.arctan(omega*R*C) # k atteso 0 (or pi if definition differs)

def fase_Hr(omega, C, k):
    """Phase of RC High-Pass Filter Transfer Function (Vr vs Vg)"""
    # arg(Hr) = pi/2 - arctan(wRC) + k (offset k is fitted)
    # Combined offset: k' = k + pi/2
    R = R_nota.value
    # Fit k directly, which should approximate pi/2 if theory holds
    return k - np.arctan(omega*R*C) # k atteso pi/2

# --- Calculate Transfer Functions ---
ampiezza_Hc = amp_Vc_pp / amp_Vg # Convert pp to amplitude before dividing
ampiezza_Hc.name = '|H_C|'
ampiezza_Hc.unit = '' # Dimensionless

ampiezza_Hr = amp_Vr_pp / amp_Vg # Convert pp to amplitude before dividing
ampiezza_Hr.name = '|H_R|'
ampiezza_Hr.unit = '' # Dimensionless

# --- Perform Fits ---
print("\n--- RC Circuit Fits ---")
fit_Hc = perform_fit(pulsazione, ampiezza_Hc, fit_func_RC_lowpass, p0=[C_nota.value],
                     parameter_names=['C'], method='minuit', minuit_limits={'C': (1e-12, 1e-6)}, # Adjusted limits slightly
                     calculate_stats=True)

fit_Hr = perform_fit(pulsazione, ampiezza_Hr, fit_func_RC_highpass, p0=[C_nota.value],
                     parameter_names=['C'], method='minuit', minuit_limits={'C': (1e-12, 1e-6)}, # Adjusted limits slightly
                     calculate_stats=True)

# Note: The phase fits are sensitive to the expected offset (0 vs pi, pi/2 vs -pi/2).
# Initial guesses for k should reflect the expected approximate phase at low/high frequencies.
# fase_Hc -> should go from 0 towards -pi/2 (or pi towards pi/2 if Vg-Vr used)
# fase_Hr -> should go from pi/2 towards 0
# The data seems shifted (fase_Vc starts near pi, fase_Vr starts near pi/2). Fitting k allows this.
# The 'k' in fase_Hr below represents the high-frequency limit (expected pi/2).
# The 'k' in fase_Hc below represents the low-frequency limit (expected pi if data starts there, or 0 if reference inverted).
# Let's use initial guesses based on data observations. fase_Vc starts near pi, fase_Hr starts near pi/2.
fit_fase_Hc = perform_fit(pulsazione, fase_Vc, fase_Hc, p0=[C_nota.value, np.pi], method='minuit', minuit_limits={'C': (1e-12, 1e-6), 'k':(-7, 7)})
fit_fase_Hr = perform_fit(pulsazione, fase_Vr, fase_Hr, p0=[C_nota.value, np.pi/2], method='minuit', minuit_limits={'C': (1e-12, 1e-6), 'k':(-7, 7)})

# --- Plotting ---
plt.close('all') # Close previous plots before creating new ones

# Plot Hc Magnitude
main_ax_hc, residual_ax_hc = plot_fit(fit_Hc, plot_residuals=True, show_params=True, show_stats=True,
                                       use_names_units=True, # Use names/units from Measurement objects
                                       title='Circuito RC - Modulo $H_C$', data_label='Dati Sperimentali', fit_label='Fit $|H_C|$')
main_ax_hc.set_xscale('log')
main_ax_hc.set_yscale('log')
residual_ax_hc.set_xscale('log') # Ensure residual x-axis is also log
main_ax_hc.figure.savefig('./grafici/rc_hc_mag.pdf', bbox_inches='tight')
# plt.close(main_ax_hc.figure) # Close figure after saving if generating many

# Plot Hr Magnitude
main_ax_hr, residual_ax_hr = plot_fit(fit_Hr, plot_residuals=True, show_params=True, show_stats=True,
                                       use_names_units=True,
                                       title='Circuito RC - Modulo $H_R$', data_label='Dati Sperimentali', fit_label='Fit $|H_R|$')
main_ax_hr.set_xscale('log')
main_ax_hr.set_yscale('log')
residual_ax_hr.set_xscale('log')
main_ax_hr.figure.savefig('./grafici/rc_hr_mag.pdf', bbox_inches='tight')
# plt.close(main_ax_hr.figure)

# Plot Hc Phase
main_ax_fase_hc, residual_ax_fase_hc = plot_fit(fit_fase_Hc, n_fit_points=500, plot_residuals=True, show_params=True, show_stats=True,
                                                use_names_units=True,
                                                title='Circuito RC - Fase $H_C$', data_label='Dati Sperimentali', fit_label='Fit $\\arg(H_C)$')
# Use semilogx for phase plots usually
main_ax_fase_hc.set_xscale('log')
residual_ax_fase_hc.set_xscale('log')
# Optional: Convert y-axis ticks back to degrees for readability if desired
# from matplotlib.ticker import FuncFormatter
# main_ax_fase_hc.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{np.degrees(y):.0f}°'))
# residual_ax_fase_hc.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{np.degrees(y):.0f}°'))
main_ax_fase_hc.figure.savefig('./grafici/rc_fase_hc.pdf', bbox_inches='tight')
# plt.close(main_ax_fase_hc.figure)

# Plot Hr Phase
main_ax_fase_hr, residual_ax_fase_hr = plot_fit(fit_fase_Hr, n_fit_points=500, plot_residuals=True, show_params=True, show_stats=True,
                                                use_names_units=True,
                                                title='Circuito RC - Fase $H_R$', data_label='Dati Sperimentali', fit_label='Fit $\\arg(H_R)$')
main_ax_fase_hr.set_xscale('log')
residual_ax_fase_hr.set_xscale('log')
# Optional y-axis formatting to degrees
# main_ax_fase_hr.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{np.degrees(y):.0f}°'))
# residual_ax_fase_hr.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{np.degrees(y):.0f}°'))
main_ax_fase_hr.figure.savefig('./grafici/rc_fase_hr.pdf', bbox_inches='tight')
# plt.close(main_ax_fase_hr.figure)

# --- Statistical Analysis ---
print("\n--- RC Statistical Analysis ---")

# Collect C values from fits
c_values = []
fit_results_list = [fit_Hc, fit_Hr, fit_fase_Hc, fit_fase_Hr]
fit_names = ['|Hc|', '|Hr|', 'arg(Hc)', 'arg(Hr)']

for fit, name in zip(fit_results_list, fit_names):
    if fit.success and 'C' in fit.parameters:
        c_meas = fit.parameters['C']
        c_meas.name = f"C from {name}" # Assign a descriptive name
        c_values.append(c_meas)
        print(f"Fit {name}: C = {c_meas.to_eng_string(2)}")
    else:
        print(f"Fit {name}: Failed or C not found.")

# Calculate weighted mean if we have valid C values
if len(c_values) > 1:
    try:
        C_mean = weighted_mean(c_values)
        C_mean.name = "C weighted mean"
        print(f"\nWeighted Mean C = {C_mean.to_eng_string(2)}")

        # Perform compatibility tests
        print("\nCompatibility Tests (alpha=0.05):")
        # Compare first two C values
        if len(c_values) >= 2:
             comp_c1_c2 = test_comp(c_values[0], c_values[1])
             print(f"- {c_values[0].name} vs {c_values[1].name}: {comp_c1_c2['interpretation']}")

        # Compare weighted mean C with C_nota
        comp_mean_nota = test_comp(C_mean, C_nota)
        print(f"- {C_mean.name} vs {C_nota.name}: {comp_mean_nota['interpretation']}")

        # Optionally, test all pairs or mean vs individual fits

    except ValueError as e:
        print(f"\nError calculating weighted mean or compatibility: {e}")
elif len(c_values) == 1:
     print("\nOnly one successful fit for C, cannot calculate weighted mean or compare fits.")
     C_mean = c_values[0] # Use the single value as the 'best' estimate
     comp_mean_nota = test_comp(C_mean, C_nota)
     print("\nCompatibility Test (alpha=0.05):")
     print(f"- {C_mean.name} vs {C_nota.name}: {comp_mean_nota['interpretation']}")
else:
    print("\nNo successful fits found for C.")
    C_mean = Measurement(np.nan, np.nan, name="C mean (failed)") # Placeholder

# --- Generate LaTeX Tables ---
print("\n--- Generating LaTeX Tables ---")

# Data Table (Vertical orientation often better for many points)
try:
    # Include amplitudes (pp) and phases (degrees) as measured
    latex_data_table = latex_table_data(
        frequenza, amp_Vg, amp_Vc_pp, amp_Vr_pp, fase_Vc_deg, fase_Vr_deg,
        orientation='v',
        sig_figs_error=2, # Use 2 sig figs for errors in data table
        caption="Dati sperimentali per il circuito RC.",
        # Use default labels from Measurement names
        copy_to_clipboard=False # Set True if desired
    )
    print("\nLaTeX Data Table (RC):")
    print(latex_data_table)
except Exception as e:
    print(f"\nError generating LaTeX data table: {e}")

# Fit Results Table
try:
    latex_fit_table = latex_table_fit(
        fit_Hc, fit_Hr, fit_fase_Hc, fit_fase_Hr,
        orientation='h', # Params as rows, Fits as columns
        sig_figs_error=2,
        fit_labels=fit_names, # Use the names defined earlier
        param_labels={'C': '$C$', 'k': '$k$'}, # Use LaTeX for symbols
        stat_labels={'chi_square': '$\\chi^2$', 'dof': 'DoF', 'reduced_chi_square': '$\\chi^2/\\nu$'},
        caption="Risultati dei fit per il circuito RC. Il parametro $R$ è stato fissato a $R_{nota}$.",
        copy_to_clipboard=False
    )
    print("\nLaTeX Fit Results Table (RC):")
    print(latex_fit_table)
except Exception as e:
    print(f"\nError generating LaTeX fit results table: {e}")

# NOTE: The compatibility test results are printed to the console.
# You should summarize these results manually in your LaTeX document text
# or create a small, dedicated table for them in the LaTeX source.

print("\n--- RC Analysis Complete ---")

# --- END OF FILE rc.py ---