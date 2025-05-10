import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate

filename = "Daten/calibrated_data.csv"

data = np.loadtxt(filename, delimiter=",", skiprows=1)

STEP_SIZE = 100
MAX = int(len(data[:,0]) * 0.9)
freq, S = data[:MAX:STEP_SIZE,0], data[:MAX:STEP_SIZE,1]

# plt.plot(freq, S, label='data')
# plt.show()


def peakfunc(x, mu, amp, sigma):
    # return amp * np.exp(-(x-mu)**2/(2*sigma**2))
    return amp * sigma**2/4 /((x - mu)**2 + sigma**2 /4)

def gauss(x, mu, amp, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2))


def fit_func(x, *params):
    m, c, mu_bg1, a_bg1, sigma_bg1, mu_bg2, a_bg2, sigma_bg2, mu1, a1, sigma1, mu2, a2, sigma2, mu3, a3, sigma3, mu4, a4, sigma4, mu5, a5, sigma5 = params
    return (m*x + c -
            gauss(x, mu_bg1, a_bg1, sigma_bg1) -
            gauss(x, mu_bg2, a_bg2, sigma_bg2) +
            peakfunc(x, mu1, a1, sigma1) +      
            peakfunc(x, mu2, a2, sigma2) +
            # peakfunc(x, mu2, a2, sigma2) +
            peakfunc(x, mu3, a3, sigma3) +
            peakfunc(x, mu4, a4, sigma4) +
            peakfunc(x, mu5, a5, sigma5)
    )


p0_old = [-0.05, 0.03,
      0.0034, 0.043, 0.002244/2, 
      0.004, 0.044, 0.001844/2,
    #   0.00188, 0.0005, 0.0000546,
      0.00224, 0.00221, 0.0000563,
      0.0026007, 0.0023902, 0.0000542,
      0.0028637, 0.00685, 0.0000665,
      0.0032253, 0.01374, 0.0000828,
      0.0038320, 0.003161, 0.0000624
      ]

p0 = [-2e-11, 0.05,
      1.266e9, 0.06, 0.2e9,
      1.06e9, 0.025, 0.3e9,
      9.12e8, 0.002, 0.1e8,
      0.99e9, 0.002, 0.1e8,
      1.048e9, 0.005, 0.022e8,
      1.127e9, 0.01, 0.02e9,
      1.26e9, 0.007, 0.04e8
      ]

popt, pcov = curve_fit(fit_func, freq, S, p0=p0, maxfev=10000)
errors = np.sqrt(np.diag(pcov))



plt.plot(freq, S, label='Messdaten')
plt.plot(freq, fit_func(freq, *popt), label='Fit', color='red')
plt.xlabel('Frequenz [Hz]')
plt.ylabel('U [V]')
plt.legend()
plt.grid()
plt.title('Fit der kalibrierten Daten')
plt.savefig("fit.png")
plt.show()
print("Fit parameters:")


# Prepare data for the table
table_data = []

# Add background peaks
for i in range(2):  # Two background peaks
    peak_number = f"Background {i + 1}"
    mu = f"{popt[1 + 1 + i * 3]:.4e} ± {errors[1 + 1 + i * 3]:.4e}"
    a = f"{popt[2 + 1 + i * 3]:.4e} ± {errors[2 + 1 + i * 3]:.4e}"
    sigma = f"{popt[3 + 1 + i * 3]:.4e} ± {errors[3 + 1 + i * 3]:.4e}"
    table_data.append([peak_number, mu, a, sigma])

# Add main peaks
for i in range(5):  # Adjust range if more peaks are added
    peak_number = f"Peak {i + 1}"
    mu = f"{popt[7 + 1 + i * 3]:.4e} ± {errors[7 + 1 + i * 3]:.4e}"
    a = f"{popt[8 + 1 + i * 3]:.4e} ± {errors[8 + 1 + i * 3]:.4e}"
    sigma = f"{popt[9 + 1 + i * 3]:.4e} ± {errors[9 + 1 + i * 3]:.4e}"
    table_data.append([peak_number, mu, a, sigma])

# Define headers
headers = ["peak number", "$\\mu$ [s]", "a [V]", "\\sigma [s]"]

# Print the table
print(tabulate(table_data, headers=headers, tablefmt="latex_raw"))

# Prepare data for the new table
new_table_data = []

# # Add background peaks with updated calculations
# for i in range(2):  # Two background peaks
#     peak_number = f"Background {i + 1}"
#     mu = popt[1 + 1 + i * 3]
#     mu_error = errors[1 + 1 + i * 3]
#     frequ = mu + 384.230894048e12
#     frequ_error = mu_error
#     a = f"{popt[2 + 1 + i * 3]:.12e} ± {errors[2 + 1 + i * 3]:.6e}"
#     sigma = popt[3 + 1 + i * 3]
#     sigma_error = errors[3 + 1 + i * 3]
#     gamma = 1 / (2 * sigma)
#     gamma_error = sigma_error / (2 * sigma**2)
#     new_table_data.append([peak_number, f"{frequ:.12e} ± {frequ_error:.6e}", a, f"{gamma:.12e} ± {gamma_error:.6e}"])

# Add main peaks with updated calculations
for i in range(5):  # Adjust range if more peaks are added
    peak_number = f"Peak {i + 1}"
    mu = popt[7 + 1 + i * 3]
    mu_error = errors[7 + 1 + i * 3]
    frequ = mu + 384.230894048e12
    frequ_error = mu_error
    a = f"{popt[8 + 1 + i * 3]:.12e} ± {errors[8 + 1 + i * 3]:.6e}"
    sigma = popt[9 + 1 + i * 3]
    sigma_error = errors[9 + 1 + i * 3]
    gamma = 1 / (2 * sigma)
    gamma_error = sigma_error / (2 * sigma**2)
    new_table_data.append([peak_number, f"{frequ:.12e} ± {frequ_error:.6e}", a, f"{gamma:.6e} ± {gamma_error:.4e}"])

# Print the new table
print(tabulate(new_table_data, headers=headers, tablefmt="latex_raw"))



freq_D2 = 384.230484468e12
stoerungen = [-229.8518e6, -72.9112e6, 193.7407e6]

# Add frequency shifts for crossover peaks
crossover_shifts = [(stoerungen[i] + stoerungen[j]) / 2 for i in range(len(stoerungen)) for j in range(i + 1, len(stoerungen))]
stoerungen.extend(crossover_shifts)

# # Print all frequency shifts
# print("Frequency shifts (including crossover peaks):")
# for shift in stoerungen:
#     print(f"{shift:.6e}")

stoerungen = [s + freq_D2 for s in stoerungen]

print("Frequency shifts (including crossover peaks):")
for shift in stoerungen:
    print(f"{shift:.11e}")


