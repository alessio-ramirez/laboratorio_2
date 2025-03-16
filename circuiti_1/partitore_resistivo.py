import numpy as np
from liblab import test_comp
from guezzi import *
import math

"""per avere V_out = 0.5 V_in imponiamo R_1 = R_2 (metà della caduta di potenziale si ha dopo R_1).
Per avere la caduta di tensione su R_load indipendente da R_load imponiamo R_load molto grande,
infatti la corrente segue l'altro percorso e non si ha caduta di potenziale dovuta a R_load"""

R_1 = 21.67
R_2 = 21.65

"""un po di corrente fluisce in R_load perche V_out tende ad aumentare"""
V_in = np.array([2.164, 2.164, 2.164, 2.164, 2.164]) #volt
V_out = np.array([1.069, 1.075, 1.077, 1.078, 1.078]) #volt
R_load = np.array([1, 2, 3, 4, 5])*10**6 #mega ohm

