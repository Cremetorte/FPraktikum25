import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for plotting


file = "Daten/Radien.dat"
data = np.loadtxt(file, skiprows=1)

p = data[:, 0].flatten()
sigmaminus = data[:, 1].flatten()  # m
pi = data[:, 2].flatten()  # m
sigmaplus = data[:, 3].flatten()  # m


dR_sq_pi = np.array([pi[i+1]**2 - pi[i]**2 for i in range(len(pi)-1)])
dR_pi = np.array([pi[i+1] - pi[i] for i in range(len(pi)-1)])

plt.plot(p[:-1], dR_sq_pi*1e12, 'x', label=r'$\Delta R^2_\pi$')
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R^2_\pi$ [μm²]')
plt.title('Differenzen der Radien im Quadrat')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien_quadrat.png")

plt.cla()
plt.plot(p[:-1], dR_pi*1e6, 'x', label=r'$\Delta R_\pi$')
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R_\pi$ [μm]')
plt.title('Differenzen der Radien')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien.png")