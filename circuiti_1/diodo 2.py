import numpy as np
import math
from liblab import *

"""configurazione 1 (voltometro in parallelo col diodo)"""

I = np.array([0.03, 0.05, 0.20, 2.46, 53, 720, 15]) * 10**(-6)
I = {0.03:0.01, 0.05:0.01, 0.20:0.02, 2.13:0.01, 53:1, 620:2, 14600:50, 125000:2000, 482000:2000, 53400:500, 2345:3} #micro ampere
V = np.array([0.100, 0.212, 0.309, 0.396, 0.501, 0.599, 0.712, 0.795, 0.838, 0.758, 0.651])
R_eq = V/I

"""configurazione 2 (voltometro in parallelo col generatore)"""
I = {62.40:0.30, 0.03:0.01, 0.03:0.01, 0.15:0.01, 4.59:0.01, 133.3:0.3, 386:1, 990:1.5, 1705:2, 2130:2, 2615:1, 3614:2} #micro ampere
V = np.array([0.519, 0.151, 0.212, 0.309, 0.422, 0.555, 0.621, 0.718, 0.810, 0.861, 0.918, 1.030])