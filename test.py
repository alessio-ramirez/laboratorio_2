import numpy as np
import uncertainties as unp
from uncertainties import ufloat
from guezzi import error_prop
from liblab import *
import math

# Given constants
a = 1.8
b = 10596157

# Given data
I_2_values = np.array([0.03, 1.04, 2.02, 3.03, 4.03, 5.02, 6.03, 7.07, 8.03, 9.03, 
                       10.07, 11.08, 12.02, 13.03, 14.04, 15.04, 16.05, 17.04, 
                       18.04, 19.06, 20.03]) * 10**(-6)
V_2_values = np.array([0.01, 1.02, 2.00, 3.01, 4.00, 5.00, 6.01, 7.05, 8.00, 9.00, 
                       10.04, 11.04, 11.99, 13.00, 14.00, 15.01, 16.02, 17.01, 
                       18.00, 19.03, 20.00])

#print(np.cov(I_2_values, V_2_values, bias=False))

i2 = {val: 0.03 * 10 ** (-6) for val in I_2_values}
v2 = {val: 0.01 for val in V_2_values}

# Associated uncertainties
I_2_unc = 0.03 * 10 ** (-6)  # Uncertainty in I_2
V_2_unc = 0.01               # Uncertainty in V_2

# Create arrays of ufloat values
I_2 = np.array([ufloat(val, I_2_unc) for val in I_2_values])
V_2 = np.array([ufloat(val, V_2_unc) for val in V_2_values])


# Compute R_bias with uncertainty propagation
R_bias = lambda I_2, V_2: (V_2 - I_2 * a) / (I_2 - V_2 / b + (a / b) * I_2)
print(error_prop(R_bias, [i2, v2], use_covariance=False, copy_latex=True))
R_bias = (V_2 - I_2 * a) / (I_2 - V_2 / b + (a / b) * I_2)
# Print results
#for i, R in enumerate(R_bias):
    #print(f"R_bias[{i}] = {R}")

omega2 = [2.59, 2.58, 2.60, 2.62, 2.59, 2.60, 2.63, 2.59, 2.58, 2.60]
d_omega2 = std_err(omega2)
omega2 = mean(omega2)

gamma2 = [0.120, 0.118, 0.116, 0.116, 0.119, 0.120, 0.122, 0.116, 0.119, 0.119]
#latex_table(gamma2)
d_gamma2 = std_err(gamma2)
gamma2 = mean(gamma2)
omega_att, d_omega_att = eprop("o = sp.sqrt(g ** 2 + m**2)", gamma2, d_gamma2, omega2, d_omega2, copy = False )
f = lambda gamma, omega: sp.sqrt(gamma**2 + omega**2)
print(omega_att, d_omega_att, f(gamma2, omega2), error_prop(f, [{gamma2:d_gamma2}, {omega2:d_omega2}]))

