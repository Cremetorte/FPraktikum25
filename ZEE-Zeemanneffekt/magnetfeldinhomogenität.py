import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for plotting
from scipy.optimize import curve_fit



data = np.loadtxt("Daten/Magnetfeldinhomogenität.txt", skiprows=1)
z = data[:, 0].flatten()  # mm
B = data[:, 1].flatten()  # mT

delta_B_abs = np.max(B) - np.min(B)
d = z[np.where(B == np.min(B))[0][0]] - z[np.where(B == np.max(B))[0][0]]
print(d)
dB = delta_B_abs / d
print(f"Magnetfeldinhomogenität: ΔB = {dB:.4e} mT/mm")

plt.plot(z, B, 'o', label='Messdaten')
plt.xlabel('z (mm)')
plt.ylabel('B (mT)')
plt.title('Magnetfeldinhomogenität')
plt.grid()
plt.legend()
plt.show()