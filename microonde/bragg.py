import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from burger_lib.guezzi import *
import numpy as np

d = 4 #cm #spessore strato
angolo =   [20,   23,   25,   27,  30,   32,   35,   38,  40,   43,   45 ,  48,   50,    55,   60] #angolo di rotazione del cubo rispetto all'emettitore
tensione = [0.03, 0.03, 0.03, 0.04, 0.09, 0.18, 0.21, 0.23, 0.28, 0.32, 0.30, 0.15, 0.14, 0.02, 0.05] #in volt è vero
tens_err = [0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01]
#ultime misure poco sensate perchè il rilevatore è inclinato al punto che il segnale dal trasmettitore ci finisce direttamente dentro senza fare riflessione
