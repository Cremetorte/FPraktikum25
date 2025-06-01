import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
# import matplotlib
# matplotlib.use('TkAgg')



def sinc(x, a, b, c):
    """Sinc function for fitting."""
    return a * np.abs(np.sinc(b * x)) + c


files = ["JJ1_IcH_001.dat", "JJ1_IcH_001.dat", "JJ1_IcH_001.dat"]
labels = ["Josephsonkontakt 1", "Josephsonkontakt 2", "Josephsonkontakt 3"]

files = ["Daten/" + file for file in files]

data = {}


for file in files:
    data_temp = np.loadtxt(file, skiprows=0)
    
    # Extracting the first two columns
    I_sp, I_c = data_temp[:, 0].T, data_temp[:, 1].T

    # convert to H Field
    mu_0 = 1.2566373e-6
    k = 0.2234
    H = k/mu_0 * I_sp
    
    # put into data dictionary
    data[file] = (H, I_c)
    
plt.title("Sinc-Fit der Josephson-Kontakte")
plt.xlabel("H [kA/m]")
plt.ylabel("I_c [µA]")
plt.grid()
plt.legend()

for i, file in enumerate(files):
    plt.cla()
    H, I_c = data[file]
    
    # Fit the data
    popt, pcov = curve_fit(sinc, H, I_c, p0=[1e-5, 1/7000, 0])
    errors = np.sqrt(np.diag(pcov))
    
    # Generate x values for the fit line
    x_fit = np.linspace(np.min(H), np.max(H), 1000)
    y_fit = sinc(x_fit, *popt)
    
    print(f"Fit parameters for {labels[i]}: {popt}, errors: {errors}")
    
    # Plot the data and the fit
    plt.plot(H *1e-3, I_c * 1e6, marker='x', label=f"{labels[i]} - Daten")
    plt.plot(x_fit *1e-3, y_fit * 1e6, linestyle='-', label=f"{labels[i]} - Fit")
    plt.title(f"Sinc-Fit an {labels[i]}")
    plt.xlabel("$H$ [kA/m]")
    plt.ylabel("$I_c$ [µA]")
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig("Plots/SincFit_" + labels[i].replace(" ", "_") + ".png")