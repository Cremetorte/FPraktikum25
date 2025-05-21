import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


R_rest = 2.118
dR_rest = 0.015

# linear fit parameters
m = 1.19238e-02
dm = 8.91499e-04
c = -1.06131e+00
dc = 1.77094e-01

# Kritisches Feld
B_c2 = (R_rest - c) / m
dB_c2 = np.sqrt((dm * (R_rest - c) / m**2)**2 + (dc / m)**2 + (dR_rest / m)**2)

print(f"B_c2 = {B_c2:.4e} ± {dB_c2:.4e}")


phi_0 = 2.067833848e-15
rho_rest = 