import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def fit_func(x, m, c):
    """Linear function for curve fitting."""
    return 1/m * x + c

files = ["JJ1_IVC_002_avg=100.dat", "JJ2_IVC_002_avg=100.dat", "JJ3_IVC_002_avg=100.dat"]
labels = ["Josephsonkontakt 1", "Josephsonkontakt 2", "Josephsonkontakt 3"]

lin_region = [
    (4e-3,7e-3),
    (1e-3,2.4e-3),
    (0.3e-3,0.45e-3)
]

files = ["Daten/" + file for file in files]


data_dict = {}

for file in files:
    data = np.loadtxt(file, skiprows=0)
    data_dict[file] = data



for file in files:
    U = data_dict[file][:, 0]  # Convert to mV
    I = data_dict[file][:, 1]  # Convert to µA

    r = []
    dr_ls = []

    mask_pos = (U > lin_region[files.index(file)][0]) & (U < lin_region[files.index(file)][1])
    U_pos = U[mask_pos]
    I_pos = I[mask_pos]
    popt_pos, pcov_pos = curve_fit(fit_func, U_pos, I_pos)
    r.append(popt_pos[0])
    dr_ls.append(np.sqrt(np.diag(pcov_pos))[0])

    mask_neg = (U < -lin_region[files.index(file)][0]) & (U > -lin_region[files.index(file)][1])
    U_neg = U[mask_neg]
    I_neg = I[mask_neg]
    popt_neg, pcov_neg = curve_fit(fit_func, U_neg, I_neg)
    r.append(popt_neg[0])
    dr_ls.append(np.sqrt(np.diag(pcov_neg))[0])


    plt.figure(figsize=(8, 5))
    plt.plot(U, I, label="Messdaten")
    plt.plot(U_pos, I_pos, 'r.', label="Fit-Bereich (+)")
    plt.plot(U_neg, I_neg, 'r.', label="Fit-Bereich (-)")

    # Fit lines
    U_fit_pos = np.linspace(0, U_pos.max(), 100)
    I_fit_pos = fit_func(U_fit_pos, *popt_pos)
    plt.plot(U_fit_pos, I_fit_pos, 'b--', label="Fit (+)")

    U_fit_neg = np.linspace(U_neg.min(), 0, 100)
    I_fit_neg = fit_func(U_fit_neg, *popt_neg)
    plt.plot(U_fit_neg, I_fit_neg, 'g--', label="Fit (-)")

    plt.xlabel("U [V]")
    plt.ylabel("I [A]")
    plt.title(labels[files.index(file)])
    plt.legend()
    plt.grid()
    plt.show()

    
    dr = max((max(r) - min(r))/2, np.max(dr_ls))
    r = np.mean(r)
    print(f"R = {r:.4e} ± {dr:.4e} Ω for {file}")
