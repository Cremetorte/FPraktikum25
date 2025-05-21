import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
matplotlib.use('TkAgg')

data = np.loadtxt("Daten/A2/FitParameter.dat", delimiter="\t", skiprows=1)

def linear(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear, data[:, 0], data[:, 1])
error = np.sqrt(np.diag(pcov))
x_fit = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
y_fit = linear(x_fit, *popt)

print(f"Fitparameter: a = {popt[0]:.5e} ± {error[0]:.5e}, b = {popt[1]:.5e} ± {error[1]:.5e}")


plt.plot(x_fit, y_fit, label=fr"Linear Fit: $R_{{FF}}$ = {popt[0]:.3e}$\cdot$B - {-1*popt[1]:.3e}")
plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='x', label="Fitparameter m")
plt.xlabel("Spulenmagnetfeld B [mT]")
plt.ylabel(r"Flux-Flow-Widerstand R_{FF} [$\Omega$]")
plt.title("Fitergebnisse des Flux-Flow-Widerstandes")
plt.legend()
plt.grid()
plt.savefig("Plots/FitParameter.png", dpi=300)