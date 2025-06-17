import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for plotting


data = np.loadtxt("Daten/Magnetfeldkalibrierung.txt", skiprows=1)
# print(data)
I = data[:, 0].T
B = data[:, 1].T # mT

def poly(x, *par):
    y = par[0] * x**(len(par)-1)
    for i in range(1, len(par)):
        y += par[i] * x**(len(par)-i-1)
    return y   
        
poly_deg = 3

popt, pcov = np.polyfit(I, B, poly_deg, cov=True)
errors = np.sqrt(np.diag(pcov))

# string for printing
str = ""
for i in range(l := len(popt)):
    str += fr"({popt[i]:.4f}\pm {errors[i]:.4f}) \cdot I^{l-i-1} + "


print(f"Fit Function: B = {str}")
# print(f"\n{popt}")


chi_squared_red = np.sum((B - poly(I, *popt)) ** 2/(poly(I, *popt)))
print(f"Reduziertes Chi-Quadrat: {chi_squared_red:.4f}")

# Generate data for plotting
I_fit = np.linspace(min(I), max(I), 100)
B_fit = poly(I_fit, *popt)
# Plotting
plt.plot(I, B, 'x', label='Messdaten')
plt.plot(I_fit, B_fit, label='Quadratischer Fit', color='orange')
plt.xlabel('Strom I (A)')
plt.ylabel('Magnetfeld B (mT)')
plt.title('Magnetfeldkalibrierung')
plt.legend()
plt.grid()
# plt.savefig("Plots/Magnetfeldkalibrierung.png")
plt.show()
