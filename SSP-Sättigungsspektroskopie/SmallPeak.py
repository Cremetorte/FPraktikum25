import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate

filename = "Daten/calibrated_data.csv"

data = np.loadtxt(filename, delimiter=",", skiprows=1)

STEP_SIZE = 100

MIN = int(len(data[:,0]) * 0.09)
MAX = int(len(data[:,0]) * 0.15)
print(MIN, MAX)
t,S = data[MIN:MAX:STEP_SIZE,0], data[MIN:MAX:STEP_SIZE,1]

# plt.plot(t, S, label='data')
# plt.xlabel('t')
# plt.ylabel('S')
# plt.title('Sättigungsspektrum')
# plt.legend()
# plt.grid()
# plt.show()

def fit_func(x, *params):
    m, c, mu, amp, sigma = params
    return (m*x + c +
            amp * sigma**2/4 /((x - mu)**2 + sigma**2 /4)
    )

def lin(x, m, c):
    return m*x + c

plin,pc = curve_fit(lin, t, S, p0=[0.1, 0.1])


p0 = [*plin] + [ 8.35e8, 0.0005, 0.1e8]

popt, pcov = curve_fit(fit_func, t, S, p0=p0)
perr = np.sqrt(np.diag(pcov))

print("Fitted parameters:")
print(tabulate([["m", "c", "mu", "amp", "sigma"], popt, perr], headers=["parameter", "Value", "Error"], floatfmt=".4e"))

# plt.plot(t, S, label='data')
# plt.plot(t, fit_func(t, *popt), label='fit', color='red')
# plt.xlabel('Frequenz [Hz]')
# plt.ylabel('U [V]')
# plt.title('Zoom auf den kleinen Peak')
# plt.legend()
# plt.grid()
# plt.show()
# plt.savefig("Plots/KleinerPeak.png")

peak = popt[2]
freq = peak + 384.230894048e12

freq_lit = 384.230484468e12 - 229.8518e6
print(f"Peak frequency: {freq} Hz")
print(f"Fehler = {perr[2]} Hz")
print(f"Literatur frequency: {freq_lit:.9f} Hz")
print(f"Abweichung: {freq - freq_lit:.9f} Hz")

sigma = popt[4]
sigma_lit = 1/freq_lit
print(f"Linienbreite: {2* sigma:.9f} Hz")
print(f"Literatur Linienbreite: {sigma_lit:.9e} Hz")