import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

t_a = 5   # Aktivierungszeit
file = "AG_csv/Ag_" + str(t_a) + "s.csv"

# 1. CSV laden (mit ; als Trenner und , als Dezimalzeichen)
df = pd.read_csv(file, sep=';', decimal=',')
data = df.to_numpy().transpose()

# 2. Daten extrahieren
t = data[0] + 7.05
N = data[1]
std_error = np.sqrt(N)

# 3. Modellfunktion: Zwei exponentielle Zerfälle
def doppel_exp(t, A1, l1, A2, l2):
    return A1 * np.exp(-l1 * t) + A2 * np.exp(-l2 * t)


def einzel_exp(t, A, l):
    return A*np.exp(-l*t)

# 4. Erste Parameter-Schätzungen: A1, λ1, A2, λ2
p0 = [max(N), 0.01, max(N)/2, 0.001]

# 5. Curve Fit
params, covariance = curve_fit(doppel_exp, t, N, p0=p0, sigma=std_error, absolute_sigma=True)

# 6. Ergebnisse
A1, l1, A2, l2 = params
dA1, dl1, dA2, dl2 = np.sqrt(np.diag(covariance))
print(f"A1 = {A1:.3f} ± {dA1:.3f}, λ1 = {l1:.5f} ± {dl1:.3f}")
print(f"A2 = {A2:.3f} ± {dA2:.3f}, λ2 = {l2:.5f} ± {dl2:.3f}")

# 7. Plot
t_fine = np.linspace(0, max(t), 1000)
N_fit = doppel_exp(t_fine, *params)
exp_1 = einzel_exp(t_fine, A1, l1)
exp_2 = einzel_exp(t_fine, A2, l2)

# Messdaten
plt.plot(t, N, 'x', label="Messdaten")

# Gesamtfit
plt.plot(t_fine, N_fit, '-', label="Fit")

# einzelne Exponetialfunktionen
plt.plot(t_fine, exp_1, label="$^{{110}}$Ag")
plt.plot(t_fine, exp_2, label="$^{{108}}$Ag")


plt.xlabel("Zeit [s]")
plt.ylabel("Zerfallsrate [1/s]")
plt.legend()
plt.grid()
plt.savefig("fit_" + str(t_a) + "s.png")