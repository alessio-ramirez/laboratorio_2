import numpy as np
import math
from liblab import *
import matplotlib.pyplot as plt

"""configurazione 1 (voltometro in parallelo col diodo)"""

#I = np.array([0.03, 0.05, 0.20, 2.46, 53, 720, 15]) * 10**(-6)
I = {0.03:0.01, 0.05:0.01, 0.20:0.02, 3.41:0.01, 76.88:0.1, 910:2, 10500:400, 109000:1000, 490000:1000, 253:3, 2764:2, 4653:2, 32900:100, 64000:500, 253000:1000, 214000:1000, 1145:10, 4.59:0.01, 1661:1, 18100:100, 55100:1000} #micro ampere
I_ = np.array(list(I.keys()))
V = np.array([0.100, 0.212, 0.309, 0.396, 0.501, 0.600, 0.700, 0.796, 0.880, 0.559, 0.655, 0.675, 0.746, 0.773, 0.836, 0.823, 0.624, 0.421, 0.638, 0.724, 0.762])
R_eq = np.sort(V/I_)

print(R_eq)
fig, ax = plt.subplots()
ax.scatter(V, I_)
plt.show()

"""configurazione 2 (voltometro in parallelo col generatore)"""
I = {0.021:0.010, 0.02:0.01, 0.031:0.010, 0.03:0.01, 0.05:0.01, 0.12:0.01, 0.65:0.01, 2.37:0.02, 9.64:0.02, 49.30:0.1, 120.20:0.2, 358.60:1, 550:2, 951:2, 1214:2, 1658:2, 3.88:0.02, 19.50:0.1, 78.00:0.1, 176.70:0.2, 410:1}#micro ampere
V = np.array([0.018, 0.105, 0.0156, 0.207, 0.258, 0.305, 0.360, 0.401, 0.447, 0.509, 0.550, 0.616, 0.652, 0.713, 0.749, 0.805, 0.418, 0.473, 0.529, 0.570, 0.626])
I_ = np.array(list(I.keys()))
R_eq = np.sort(V/I_)
print(R_eq)
fig, ax = plt.subplots()
ax.scatter(V, I_)
plt.show()