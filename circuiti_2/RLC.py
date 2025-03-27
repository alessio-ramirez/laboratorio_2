from guezzi import *
import numpy as np

#regime sottosmorzato
R_nota = 7
R_nota_err = R_nota/100
C_nota = 10 * 10**(-9)
L_stima = 0.07

V_r_sotto = [1, 2] #millivolt
tempi_sotto = [0, 8] #microsecondi