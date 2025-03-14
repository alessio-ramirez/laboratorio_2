import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Configuration 1 data (Voltmeter in parallel with diode)
I = np.array([0.03, 0.05, 0.20, 2.46, 53, 720, 14600, 125000, 482000]) * 1e-6  # A
V = np.array([0.100, 0.212, 0.309, 0.396, 0.501, 0.599, 0.712, 0.795, 0.838])  # V
I_err = np.array([0.01, 0.01, 0.02, 0.01, 1, 2, 50, 2000, 2000]) * 1e-6  # A
V_err = 0.01  # V

# Threshold voltage fit (linear region: I > 10 µA)
mask = I > 10e-6
p_linear = np.polyfit(V[mask], I[mask], 1)
V_soglia = -p_linear[1] / p_linear[0]  # x-intercept

# Shockley fit (I = I0 * exp(qV/(gkT)))
def shockley_fit(V, I0, g):
    q = 1.6e-19
    k = 1.38e-23
    T = 300
    return I0 * (np.exp(q * V / (g * k * T)) - 1)

popt, pcov = curve_fit(shockley_fit, V, I, p0=[1e-12, 1])
I0, g = popt

# Plot results
plt.errorbar(V, I, xerr=V_err, yerr=I_err, fmt='o', label='Data')
plt.plot(V, np.polyval(p_linear, V), '--', label='Linear Fit (Threshold)')
plt.plot(V, shockley_fit(V, *popt), 'r-', label='Shockley Fit')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.yscale('log')
plt.legend()
plt.show()

print(f"Threshold Voltage (V_soglia) = {V_soglia:.3f} V")
print(f"Shockley Parameters: I0 = {I0:.2e} A, g = {g:.2f}")