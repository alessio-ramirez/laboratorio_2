# --- START OF UPDATED FILE rl.py ---
import sys
# Adjust the path if 'burger_lib' is not directly in the parent directory
sys.path.append('../') # Assumes 'rl.py' is in a subdir and 'burger_lib' is one level up
try:
    # Try importing from the assumed structure first
    from burger_lib.guezzi import *
except ImportError:
    print("Warning: Could not import guezzi library from '../burger_lib'.")
    print("Attempting to import directly (ensure 'guezzi' directory is in the path or installed)...")
    try:
        from guezzi import *
    except ImportError as e:
        print(f"Fatal Error: Failed to import guezzi library. {e}")
        print("Please ensure the library is correctly placed or installed.")
        sys.exit(1) # Exit if library cannot be found

import numpy as np
import matplotlib.pyplot as plt

# --- Input Data ---
R_nota = Measurement(10, 0.1, magnitude=3, unit='Ohm', name='R nota') # 10 kOhm
# R_L is the resistance of the inductor itself, measured separately?
# If R_L is significant, the fit functions need modification (using R_total = R_nota + R_L).
# For now, assume R_L is negligible or included in the effective L fit.
# R_L_val = Measurement(38.9, 0.1, unit='Ohm', name='R_L Inductor')

# Ampiezza del generatore (input voltage)
# Adjusted error to 0.04V from 0.4V which seemed large for a 4V signal.
amp_Vg = Measurement (4.16, 0.04, unit= 'V', name='ampiezza Vg')

# Frequenza
frequenza_khz = [5, 7, 10, 15, 20, 25, 50, 75, 100, 125, 150, 180, 200, 225, 250] #kilo hertz
frequenza = Measurement(frequenza_khz, 0, magnitude=3, unit='Hz', name='Frequenza') # k multiplier in magnitude
pulsazione = frequenza * (2 * np.pi) # Angular frequency (rad/s)
pulsazione.name = 'Pulsazione' # Give it a name for potential use in tables/plots

# Ampiezza Vl (Output for High-Pass Filter: Vout = Vl)
# Vl = V(A) - V(B) in Fig 1. Vout measured across L.
amp_Vl_pp_values = [0.640, 0.98, 1.36, 1.84, 2.32, 2.68, 3.80, 4.24, 4.32, 4.40, 4.40, 4.40, 4.32, 4.24, 4.16]
amp_Vl_pp_errors = [0.01, 0.02, 0.01, 0.01, 0.01, 0.04, 0.04, 0.04, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04] # Length matched to values
amp_Vl = Measurement(amp_Vl_pp_values, amp_Vl_pp_errors, unit='V', name='V_L pk-pk')

# Ampiezza Vr (Output for Low-Pass Filter: Vout = Vr)
# Vr = V(B) in Fig 1. Vout measured across R.
amp_Vr_pp_values = [4.07, 4.0, 4.04, 3.90, 3.74, 3.50, 2.50, 1.80, 1.34, 0.98, 0.74, 0.52, 0.40, 0.184, 0.092] # Corrected 4,0 -> 4.0
amp_Vr_pp_errors = [0.01, 0.02, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.002, 0.004]
amp_Vr = Measurement(amp_Vr_pp_values, amp_Vr_pp_errors, unit='V', name='V_R pk-pk')

# --- Calculate Transfer Function Magnitudes ---
# H = Vout / Vin. Vin = Vg. Uses updated Measurement class for division.
H_lowpass_mag = amp_Vr / amp_Vg
H_highpass_mag = amp_Vl / amp_Vg

H_lowpass_mag.name = '|H_LP(Vr/Vg)|'
H_highpass_mag.name = '|H_HP(Vl/Vg)|'
H_lowpass_mag.unit = '' # Transfer functions are dimensionless
H_highpass_mag.unit = ''

# --- Define Fit Functions ---
# omega = angular frequency (rad/s)
# R = resistance (use nominal value R_nota)
# L = inductance (parameter to fit)
# These functions assume an ideal inductor (zero internal resistance).

def fit_func_RL_lowpass(omega, L):
    """Magnitude of RL Low-Pass Filter Transfer Function (|Vr/Vg|)"""
    R = R_nota.value # Use the nominal value for the known resistance
    # Model: R / sqrt(R^2 + (wL)^2)
    return R / np.sqrt(R**2 + (omega * L)**2)

def fit_func_RL_highpass(omega, L):
    """Magnitude of RL High-Pass Filter Transfer Function (|Vl/Vg|)"""
    R = R_nota.value # Use the nominal value for the known resistance
    # Model: wL / sqrt(R^2 + (wL)^2)
    term_wL = omega * L
    return term_wL / np.sqrt(R**2 + term_wL**2)

# --- Perform Fits ---

# Initial guess for L
# From previous estimate based on cutoff frequency: L ~ 20 mH
L_guess = 20e-3 # 20 mH

print("\n--- Fitting RL Low-Pass Filter (|Vr/Vg|) ---")
# Use 'minuit' if available for robustness, otherwise fallback to 'auto' (curve_fit)
fit_method_choice = 'curve_fit'

fit_lowpass = perform_fit(
    x_data=pulsazione,          # Independent variable: angular frequency (values only)
    y_data=H_lowpass_mag,       # Dependent variable: |H| measurement object
    func=fit_func_RL_lowpass,   # Fit function defined above
    p0={'L': L_guess},          # Initial guess for 'L' (required for minuit)
    parameter_names=['L'],      # Name of the parameter to fit
    method=fit_method_choice,   # Use 'minuit' or 'auto'
    calculate_stats=True,       # Calculate chi-squared, etc.
    minuit_limits={'L': (1e-6, None)} # Add limit: L must be positive (> 1uH)
)
print(fit_lowpass)
L_fit_lp = None # Initialize in case fit fails
f_cutoff_lp = None
if fit_lowpass.success:
    L_fit_lp = fit_lowpass.parameters['L']
    print(f"Fitted Inductance (Low-Pass): {L_fit_lp.to_eng_string(2)} H")
    # Calculate cutoff frequency: f_c = R / (2*pi*L)
    f_cutoff_lp = R_nota / (2 * np.pi * L_fit_lp)
    print(f"Corresponding Cutoff Frequency: {f_cutoff_lp.to_eng_string(2)} Hz")
else:
    print("Low-pass fit failed to converge or produce valid results.")

print("\n--- Fitting RL High-Pass Filter (|Vl/Vg|) ---")
fit_highpass = perform_fit(
    x_data=pulsazione,
    y_data=H_highpass_mag,
    func=fit_func_RL_highpass,
    p0={'L': L_guess},
    parameter_names=['L'],
    method=fit_method_choice,
    calculate_stats=True,
    minuit_limits={'L': (1e-6, None)} # Add limit: L must be positive (> 1uH)
)
print(fit_highpass)
L_fit_hp = None # Initialize in case fit fails
f_cutoff_hp = None
if fit_highpass.success:
    L_fit_hp = fit_highpass.parameters['L']
    print(f"Fitted Inductance (High-Pass): {L_fit_hp.to_eng_string(2)} H")
    f_cutoff_hp = R_nota / (2 * np.pi * L_fit_hp)
    print(f"Corresponding Cutoff Frequency: {f_cutoff_hp.to_eng_string(2)} Hz")
else:
    print("High-pass fit failed to converge or produce valid results.")


# --- Comparison and Plotting ---
# Compare the results from the two fits if both succeeded
if L_fit_lp and L_fit_hp:
    print("\nCompatibility Check:")
    comp_L = test_comp(L_fit_lp, L_fit_hp)
    print(f"LP vs HP Fitted L: {comp_L['interpretation']} (Z={comp_L['z_score']:.2f}, p={comp_L['p_value']:.3f})")

    # Calculate weighted mean if compatible
    if comp_L['compatible']:
        L_combined = weighted_mean([L_fit_lp, L_fit_hp])
        print(f"Weighted Mean L: {L_combined.to_eng_string(2)} H")


# --- Plotting ---
# Create plots only if the corresponding fit was successful

plot_kwargs_base = dict(
    plot_residuals=True,
    show_params=True,
    show_stats=True,
    xlabel='Angular Frequency $\omega$ (rad/s)', # Use LaTeX for omega
    ylabel='Transfer Function Magnitude |H|',
    marker='o', markersize=5, capsize=3, # Common data point style
    show_plot=False # Don't show plots immediately, show all at the end
)

if fit_lowpass.success:
    ax_main_lp, ax_res_lp = plot_fit(
        fit_lowpass,
        data_label='|Vr/Vg| Data',
        fit_label=f'RL Low-Pass Fit (L={L_fit_lp.to_eng_string(2)} H)',
        title='RL Low-Pass Filter Response Fit (|V_R / V_g|)',
        **plot_kwargs_base
    )
    ax_main_lp.set_xscale('log') # Use log scale for frequency axis
    ax_main_lp.set_yscale('log') # Use log scale for magnitude (Bode plot style)
    if ax_res_lp: ax_res_lp.set_xscale('log') # Match x-scale for residuals
    plt.tight_layout()
    # fig_lp.savefig("rl_lowpass_fit.png") # Optional save

if fit_highpass.success:
    ax_main_hp, ax_res_hp = plot_fit(
        fit_highpass,
        data_label='|Vl/Vg| Data',
        fit_label=f'RL High-Pass Fit (L={L_fit_hp.to_eng_string(2)} H)',
        title='RL High-Pass Filter Response Fit (|V_L / V_g|)',
        **plot_kwargs_base
    )
    ax_main_hp.set_xscale('log')
    ax_main_hp.set_yscale('log')
    if ax_res_hp: ax_res_hp.set_xscale('log')
    plt.tight_layout()
    # fig_hp.savefig("rl_highpass_fit.png") # Optional save

# Show all created plots
plt.show()


# --- Phase Data (Available but not fitted here) ---
# As noted before, fitting phase requires defining the theoretical phase function arg(H(omega))
# and ensuring the measured phase is relative to the correct reference (Vg).
fase_Vl_values = [-85, -95, -101, -105, -115, -120, -142, -155, -162, -167, -171, -174, -176, -179, 180]
fase_Vl_errors = [2, 2, 2, 6, 6, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
fase_Vl = Measurement(fase_Vl_values, fase_Vl_errors, unit='deg', name='Phase Vl (relative?)')
# Phase RL HP: arctan(R / (wL)) - ranges from 90deg (low freq) to 0deg (high freq)

fase_Vr_values = [-8.5, -13, -16, -25, -32, -37.5, -62, -77, -87, -95, -101, -107, -116, -121, -122] # Check sign of last value
fase_Vr_errors = [0.5, 1, 1, 1, 1, 0.5, 1, 2, 2, 4, 5, 5, 5, 5, 5]
fase_Vr = Measurement(fase_Vr_values, fase_Vr_errors, unit='deg', name='Phase Vr (relative?)')
# Phase RL LP: -arctan(wL / R) - ranges from 0deg (low freq) to -90deg (high freq)

# --- END OF UPDATED FILE rl.py ---