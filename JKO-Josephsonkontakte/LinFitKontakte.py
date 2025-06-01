import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy.optimize import curve_fit


def linear_fit(x, m, c):
    """Linear function for curve fitting."""
    return m * x + c


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
    


for i, file in enumerate(files):
    x_data = data_dict[file][:, 0]  # Convert to mV
    y_data = data_dict[file][:, 1]  # Convert to µA

    mask = (np.abs(x_data) > lin_region[i][0]) & (np.abs(x_data) < lin_region[i][1])  # Filter out non-positive values
    x_data_cut = x_data[mask]
    y_data_cut = y_data[mask]
    
    x_data_other = x_data[~mask]
    y_data_other = y_data[~mask]

    # Fit the data
    popt, pcov = curve_fit(linear_fit, x_data_cut, y_data_cut)
    m, c = popt
    dm, dc = np.sqrt(np.diag(pcov))
    
    print(fr"R = {m:.4e} \pm {dm:.4e} \Omega")

    # Create a linear fit line
    fit_line = linear_fit(x_data, m, c)

    # Plot the data and the fit
    plt.plot(x_data_other*1e3, y_data_other*1e6, label=labels[i], marker="x", markersize=1, linestyle=":", linewidth=0.5,
             color="black", markerfacecolor="black", markeredgecolor="black")
    plt.plot(x_data_cut*1e3, y_data_cut*1e6, label=f"{labels[i]} (Fitted Region)", marker="x", markersize=1, linestyle="", color="red")
    plt.plot(x_data*1e3, fit_line*1e6, label=f"Fit: {m:.2f}x + {c:.2f}", color='blue')

    plt.xlabel('U [mV]')
    plt.ylabel('I [µA]')
    plt.title(f'IV-Kennlinie des Josephson-Kontaktes {i+1}')
    plt.grid()
    plt.legend()
    
    # plt.show()
    
    # plt.savefig(f"Plots/IVC_{labels[i]}_linfit.png", dpi=300)
    plt.clf()