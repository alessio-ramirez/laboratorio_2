from burger_lib.guezzi import *
import numpy as np
import matplotlib as plt

def frequenza_taglio (L, C ) :
    return 1 / (2*np.pi*np.sqrt(L * C_nota))

def Hr (w, R, L, C) :
    return R / np.sqrt (R**2 + (w*L - 1/(w*C)**2))
def fase_Hr (w, R, L, C) :
    -np.arctan ((w*L - 1/(w*C))/R)

def Hl (w, R, L, C) :
    return w*L / np.sqrt (R**2 + (w*L - 1/(w*C)**2))
def fase_Hl (w, R, L, C) :
    np.pi/2 - np.arctan ((w*L - 1/(w*C))/R)

def Hc (w, R, L, C) :
    1 / (w*C)**2 / np.sqrt (R**2 + (w*L - 1/(w*C)**2))
def fase_Hc (w, R, L, C) :
    np.pi/2 - np.arctan ((w*L - 1/(w*C))/R)


#regime sovrasmorzato
C_nota = Measurement(10, 1, magnitude=-9, unit='F', name='C nota')
R_nota = Measurement(10, 0.1, magnitude=3, unit='Ohm', name='R nota')

#
frequenza = [100, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 18000, 25000, 30000, 40000, 50000, 65000, 80000, 100000, 120000, 150000, 200000, 240000, 270000] #hertz
amp_Vg = [4.16, 4.16, 4.16, 4.16, 4.16, 4.16, 4.16, 4.20, 4,32, 4.32, 4.16, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.36, 4.36, 4.36, 4.36, 4.36, 4.16] #volt
amp_Vg_err = 0.04 #volt
amp_Vr = [0.256, 1.16, 2.10, 3.24, 3.76, 3.92, 4.08, 4.20, 4.36, 4.32, 4.08, 3.96, 3.80, 3.40, 3.04, 2.60, 2.12, 1.72, 1.22, 1, 0.44, 0.22, 0.0576] #volt
amp_Vr_err = [0.004, 0.02, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.04, 0.04, 0.02, 0.0008] #volt
fase_Vr = [85, 73, 61, 38, 28, 16, 6.5, 2, -4, -13, -19, -30, -35, -47, -56, -68, -75, -83, -95, -105, -116, -123, -125] #gradi
fase_Vr_err = [2, 1, 1, 3, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 4, 5, 1, 3, 5, 4, 5, 1] #gradi

#una sonda ai capi del generatore e una sonda tra L e C (ai capi di C + R)
#poi differenza tra 2 segnali delle 2 sonde per trovare segnale ai capi di L
frequenza = [0.5, 1, 2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, 350] #kilo hertz
amp_Vg = [4.12, 4.16, 4.12, 4.12, 4.16, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.32, 4.36, 4.36, 4.36, 4.36, 4.36, 4.36, 4.36] #volt
amp_Vg_err = 0.04 #volt
amp_Vl = [0.28, 0.28, 0.36, 0.68, 0.96, 1.40, 1.88, 2.28, 2.68, 3.0, 3.56, 3.96, 4.20, 4.36, 4.52, 4.60, 4.72, 4.60, 4.44, 4.40, 4.28] #volt
amp_Vl_err = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.20, 0.04, 0.04, 0.04, 0.02, 0.04, 0.04] #volt
# NOI TUTTI CI ABBIAMO PROVATO, MICHELE PIù DEGLI ALTRI, MA LA FASE NON SI TROVA "POTA"

#una sonda ai capi del generatore e una sonda tra L e C scambiate di posizione rispettoa prima (ai capi di L + R)
#poi differenza tra 2 segnali delle 2 sonde per trovare segnale ai capi di L
frequenza = [40, 100, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 9000, 12000, 16000, 22000, 30000, 40000, 70000, 100000] #hertz
amp_Vg = [4.12, 4.12, 4.16, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.08, 4.32, 4.28, 4.28, 4.28, 4.32, 4.32, 4.28, 4.28, 4.32] #volt
amp_Vg_err = 0.04 #volt
amp_Vc = [4.20, 4.20, 4, 3.80, 3.64, 3.24, 2.84, 2.52, 2.2, 2.0, 1.84, 1.52, 1.36, 1.44, 1.24, 1.0, 0.8, 0.76, 0.72, 0.6, 0.5, 0.5] #volt
amp_Vc_err = [0.04, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08 , 0.04, 0.08, 0.04, 0.04, 0.04, 0.08, 0.04, 0.08, 0.08] #volt
print([len(i) for i in [frequenza, amp_Vg, amp_Vc, amp_Vc_err]])


