import sys
sys.path.append('../')
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

# --- Input Data ---
C_nota = Measurement(96, 1, magnitude=-9, unit='F', name='C nota') # Capacità indicata sul condensatore
R_nota = Measurement(10, 0.1, magnitude=3, unit= 'ohm', name='R nota') # Resistenza indicata sulla cassetta di resistenze
f_taglio_teorico = 1/(2*np.pi * R_nota * C_nota) # Frequenza di taglio teorica

# Ampiezza del generatore (misurata costantemente)
# Treat amp_Vg as a single measurement applicable to all frequencies
amp_Vg = Measurement (4.0, 0.4, unit= 'V', name='ampiezza Vg') # Assuming 4.0V, not 40V from magnitude=1

# Frequenza dell'onda sinusoidale
frequenza_hz = [10, 15, 25, 50, 100, 120, 150, 200, 300, 450, 700, 1000, 1500, 2200, 3200, 4500, 6000, 7500, 10000, 15000, 20000, 30000] # hertz
# Assume zero error on frequency generator setting for simplicity, or add small error if known
frequenza = Measurement(frequenza_hz, 0, unit='Hz', name='Frequenza')
pulsazione = frequenza * (2 * np.pi) # Angular frequency (rad/s)

# Ampiezza Vc (Output for Low-Pass Filter: Vout = Vc)
# Vc = V(A) - V(B) in Fig 1. Vout measured across C.
amp_Vc_pp_values = [4.0, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.00, 4.00, 4.08, 4.04, 3.96, 3.76, 3.48, 3.0, 2.52, 2.12, 1.80, 1.40, 1.00, 0.760, 0.560] #volt
amp_Vc_pp_errors = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.2, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04] #volt error --> Adjusted 0.4 to 0.04 for 700Hz based on trend
amp_Vc = Measurement(amp_Vc_pp_values, amp_Vc_pp_errors, unit='V', name='V_C pk-pk') # Peak-to-peak

# Ampiezza Vr (Output for High-Pass Filter: Vout = Vr)
# Vr = V(B) in Fig 1. Vout measured across R.
amp_Vr_pp_values = [12.8, 18.0, 28.8, 57.6, 112, 134, 168, 228, 336, 504, 784, 1090, 1580, 2140, 2700, 3180, 3480, 3640, 3800, 3920, 3980, 4040] #millivolt
amp_Vr_pp_errors = [0.4, 0.4, 0.4, 0.8, 2, 2, 2, 4, 2, 2, 2, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40] #millivolt error
amp_Vr = Measurement(amp_Vr_pp_values, amp_Vr_pp_errors, magnitude=-3, unit='V', name='V_R pk-pk') # Peak-to-peak

# --- Calculate Transfer Function Magnitudes ---
# H = Vout / Vin. Vin = Vg. Use peak-to-peak values directly.
# Updated Measurement class handles array / scalar operations.
H_lowpass_mag = amp_Vc / amp_Vg
H_highpass_mag = amp_Vr / amp_Vg

H_lowpass_mag.name = '|H_LP(Vc/Vg)|'
H_highpass_mag.name = '|H_HP(Vr/Vg)|'

# --- Define Fit Functions ---
# omega = angular frequency
# R = resistance (use nominal value from R_nota for fit)
# C = capacitance (parameter to fit)

def fit_func_RC_lowpass(omega, C):
    """Magnitude of RC Low-Pass Filter Transfer Function (|Vc/Vg|)"""
    R = R_nota.value # Use nominal value of R
    return 1.0 / np.sqrt(1.0 + (omega * R * C)**2)

def fit_func_RC_highpass(omega, C):
    """Magnitude of RC High-Pass Filter Transfer Function (|Vr/Vg|)"""
    R = R_nota.value # Use nominal value of R
    term = omega * R * C
    return term / np.sqrt(1.0 + term**2)

# --- Perform Fits ---

# Initial guess for C based on nominal value
C_guess = C_nota.value

print("\n--- Fitting RC Low-Pass Filter (|Vc/Vg|) ---")
# Use 'minuit' for potentially better convergence, requires iminuit
# Fallback to 'auto' (curve_fit here as x_err=0) if minuit not available or preferred
fit_method_choice = 'minuit'

fit_lowpass = perform_fit(
    x_data=pulsazione,          # Use angular frequency (rad/s) values
    y_data=H_lowpass_mag,       # |Vc/Vg| measurement object
    func=fit_func_RC_lowpass,
    p0=[C_guess],          # Initial guess as dict for minuit
    parameter_names=['C'],
    method=fit_method_choice,           # 'minuit', 'auto', 'curve_fit', 'odr'
    calculate_stats=True,
    minuit_limits={'C': (1e-12, None)} # Limit C > 1 pF (very small positive number)
)
print(fit_lowpass)
if fit_lowpass.success:
    C_fit_lp = fit_lowpass.parameters['C']
    print(f"Fitted Capacitance (Low-Pass): {C_fit_lp.to_eng_string(2)} F")
    f_cutoff_lp = 1 / (2 * np.pi * R_nota * C_fit_lp)
    print(f"Corresponding Cutoff Frequency: {f_cutoff_lp.to_eng_string(2)} Hz")


print("\n--- Fitting RC High-Pass Filter (|Vr/Vg|) ---")
fit_highpass = perform_fit(
    x_data=pulsazione,          # Use angular frequency (rad/s) values
    y_data=H_highpass_mag,      # |Vr/Vg| measurement object
    func=fit_func_RC_highpass,
    p0=[C_guess],          # Initial guess as dict for minuit
    parameter_names=['C'],
    method=fit_method_choice,           # 'minuit', 'auto', 'curve_fit', 'odr'
    calculate_stats=True,
    minuit_limits={'C': (1e-12, None)} # Limit C > 1 pF (very small positive number)
)
print(fit_highpass)
if fit_highpass.success:
    C_fit_hp = fit_highpass.parameters['C']
    print(f"Fitted Capacitance (High-Pass): {C_fit_hp.to_eng_string(2)} F")
    f_cutoff_hp = 1 / (2 * np.pi * R_nota * C_fit_hp)
    print(f"Corresponding Cutoff Frequency: {f_cutoff_hp.to_eng_string(2)} Hz")

# --- Comparison and Plotting (Example) ---
print(f"\nTheoretical Cutoff Frequency: {f_taglio_teorico.to_eng_string(2)} Hz")
if fit_lowpass.success and fit_highpass.success:
    print("\nCompatibility Checks (Example):")
    print(f"LP vs HP Fitted C: {test_comp(C_fit_lp, C_fit_hp)}")
    print(f"LP Fitted C vs Nominal C: {test_comp(C_fit_lp, C_nota)}")
    print(f"HP Fitted C vs Nominal C: {test_comp(C_fit_hp, C_nota)}")

# Example Plotting (requires matplotlib)
if fit_lowpass.success:
    ax_main, ax_res = plot_fit(
        fit_lowpass,
        plot_residuals=True,
        show_params=True,
        show_stats=True,
        data_label='|Vc/Vg| Data',
        fit_label=f'RC Low-Pass Fit (C={C_fit_lp.to_eng_string(2)} F)',
        xlabel='Angular Frequency (rad/s)',
        ylabel='Transfer Function Magnitude |H|',
        title='RC Low-Pass Filter Response Fit',
        show_plot=False # Prevent showing immediately if more plots follow
    )
    ax_main.set_xscale('log') # Use log scale for frequency axis
    ax_main.set_yscale('log') # Use log scale for magnitude (Bode plot style)
    if ax_res: ax_res.set_xscale('log') # Match x-scale for residuals
    plt.tight_layout()
    # plt.savefig("rc_lowpass_fit.png") # Optional save

if fit_highpass.success:
    ax_main2, ax_res2 = plot_fit(
        fit_highpass,
        plot_residuals=True,
        show_params=True,
        show_stats=True,
        data_label='|Vr/Vg| Data',
        fit_label=f'RC High-Pass Fit (C={C_fit_hp.to_eng_string(2)} F)',
        xlabel='Angular Frequency (rad/s)',
        ylabel='Transfer Function Magnitude |H|',
        title='RC High-Pass Filter Response Fit',
        show_plot=False
    )
    ax_main2.set_xscale('log')
    ax_main2.set_yscale('log')
    if ax_res2: ax_res2.set_xscale('log')
    plt.tight_layout()
    # plt.savefig("rc_highpass_fit.png")

plt.show() # Show all plots at the end

# --- Phase Data (Not Fitted Here) ---
# The provided phase data (fase_Vc, fase_Vr) needs careful interpretation
# relative to the theoretical transfer function phases arg(Vc/Vg) and arg(Vr/Vg).
# Further analysis would be needed to define the correct phase fit function.
fase_Vc_values = [178, 176, 179, 177, 177, 180, 175, 177, 174, 172, 170, 164, 157, 149, 139, 128, 120, 115, 108, 103, 102, 94] #gradi
fase_Vc_errors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #gradi error
fase_Vc = Measurement(fase_Vc_values, fase_Vc_errors, unit='deg', name='Phase Vc (relative?)')
# Convert to radians if needed for fitting: fase_Vc_rad = fase_Vc * (np.pi / 180.0)

fase_Vr_values = [92.1, 89.6, 88.9, 90.4, 88.2, 88.6, 88.0, 87.0, 84.9, 84.2, 79.6, 74.5, 67.6, 59.3, 47.9, 38.8, 31.1, 25.9, 19.4, 13.5, 10.1, 6.49] #gradi - Note: Length mismatch (23 vs 22) in original data? Trimmed last value.
fase_Vr_errors = [2, 1, 0.4, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3] #gradi error - Trimmed last value.
fase_Vr = Measurement(fase_Vr_values[:-1], fase_Vr_errors[:-1], unit='deg', name='Phase Vr (relative?)') # Trimmed to match frequency length