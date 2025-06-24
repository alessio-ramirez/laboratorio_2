import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np
import matplotlib.pyplot as plt

def linear_func(x, a, b):
    return a + b * x

I = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65]) #ampere
theta = np.array([273, 270, 265, 260, 258, 253, 249, 246, 243, 237, 235, 233, 230, 226, 225, 223, 220, 218, 215, 213, 212, 210, 209, 207, 206, 204, 203, 202, 197, 194, 192, 190, 188, 186]) #gradi

i= np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24]) #ampere
angoli = np.array([273, 270, 265, 260, 258, 253, 249, 246, 243, 237, 235, 233, 230, 226, 225, 223, 220, 218, 215, 213, 212, 210, 209, 207, 206]) #gradi
theta_rad = np.radians (273-angoli) #angolo in radianti
theta_err = np.radians (1) #sensibilità dello strumento
i_err = 0.01 #sensibilità strumento
tangenti = np.tan (theta_rad)
tan_err = 1/(1+tangenti**2) * theta_err
N = 31
L = 0.0425 #metri
d = 0.24 #metri
B = (4*np.pi*10**-7) * N * i / d
B_err = (4*np.pi*10**-7) * N / d * np.full_like(i, i_err) #ampere
#ax.scatter(B, tangenti)

#la relazione è: tan(theta)=B_solenoide/B_terrestre

B_terr_att = 50 * 10**(-6) #campo terrestre medio
B_ = Measurement(B, B_err, name='B', unit='T') 
tangenti_ = Measurement(tangenti, tan_err, name='$tan(\\theta)')
fit= perform_fit(B_, tangenti_, linear_func, p0=[[1,1]])
a, m = (fit.parameters['a'], fit.parameters['m'])
B_s = 1/m
print(f"z-score tra B terrestre e B sperimentale: {test_comp(B_terr_att, B_s)['z_score']}")