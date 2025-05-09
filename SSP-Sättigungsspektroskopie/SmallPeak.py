import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate

filename = "Daten/sättigungabsorptionsspektrum.csv"

data = np.loadtxt(filename, delimiter=",", skiprows=1)

STEP_SIZE = 100

MIN = int(len(data[:,0]) * 0.09)
MAX = int(len(data[:,0]) * 0.15)
print(MIN, MAX)
t,S = data[MIN:MAX:STEP_SIZE,0], data[MIN:MAX:STEP_SIZE,1]

""" plt.plot(t, S, label='data')
plt.xlabel('t')
plt.ylabel('S')
plt.title('Sättigungsspektrum')
plt.legend()
plt.grid()
plt.show() """

def fit_func(x, *params):
    m, c, mu, amp, sigma = params
    return (m*x + c +
            amp * sigma**2/4 /((x - mu)**2 + sigma**2 /4)
    )

p0 = [-5, 0.02,  0.00188, 0.0005, 0.0000546]

popt, pcov = curve_fit(fit_func, t, S, p0=p0)
perr = np.sqrt(np.diag(pcov))

print("Fitted parameters:")
print(tabulate([["m", "c", "mu", "amp", "sigma"], popt, perr], headers=["parameter", "Value", "Error"], floatfmt=".4e"))

plt.plot(t, S, label='data')
plt.plot(t, fit_func(t, *popt), label='fit', color='red')
plt.xlabel('t')
plt.ylabel('S')
plt.title('Sättigungsspektrum')
plt.legend()
plt.grid()
plt.show()