import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import glob
import pandas as pd
from scipy import optimize

t_a = 600   # Aktivierungszeit
file = "AG_csv/Ag_" + str(t_a) + "s.csv"


# -------------------------------------------------------------- Daten einlesen

df = pd.read_csv(file, delimiter=';', decimal=',')
data = df.to_numpy().transpose()

std_err = np.sqrt(data[1])

data[0] += 5.31
    
# print(data)

# ---------------------------------------------------------------------------- Funktionen

def exponential(t, amp, lamb):
    return amp*np.exp(-t*lamb)

def sum_exp(t, amp_1, amp_2, lamb_1, lamb_2, y_offset):
    return exponential(t, amp_1, lamb_1) + exponential(t, amp_2, lamb_2) + y_offset

start_values = [10, 10, 0.5, 0.5,1]

popt, pcov = optimize.curve_fit(sum_exp, data[0], data[1], p0=start_values)

perr = np.sqrt(np.diag(pcov))
print(popt)
print(perr)


# --------------------------------------------------------------- plot
plt.plot(data[0], data[1], marker="x", label="Data")
plt.plot(data[0], sum_exp(data[0], *popt), label="Sum of Exponentials")
plt.plot(data[0], exponential(data[0], popt[0], popt[2]), label="Exponential 1")
plt.plot(data[0], exponential(data[0], popt[1], popt[3]), label="Exponential 2")

# plt.yscale("log")
# plt.xlabel("Time")
# plt.ylabel("Counts (log scale)")
plt.legend()

plt.savefig("test.png")