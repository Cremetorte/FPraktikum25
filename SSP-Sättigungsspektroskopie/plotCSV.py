import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate

filename = "Daten/sättigungabsorptionsspektrum.csv"

data = np.loadtxt(filename, delimiter=",", skiprows=1)

STEP_SIZE = 100
MAX = int(len(data[:,0]) * 0.9)
t,S = data[:MAX:STEP_SIZE,0], data[:MAX:STEP_SIZE,1]



def peakfunc(x, mu, amp, sigma):
    # return amp * np.exp(-(x-mu)**2/(2*sigma**2))
    return amp * sigma**2/4 /((x - mu)**2 + sigma**2 /4)

def gauss(x, mu, amp, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2))


def fit_func_6_peaks(x, *params):
    if len(params) != 1 + 3*8:
        print("wrong number of arguments")
        
    
        
    c, mu_bg1, a_bg1, sigma_bg1, mu_bg2, a_bg2, sigma_bg2, mu1, a1, sigma1, mu2, a2, sigma2, mu3, a3, sigma3, mu4, a4, sigma4, mu5, a5, sigma5, mu6, a6, sigma6 = params
    return (c -
            gauss(x, mu_bg1, a_bg1, sigma_bg1) -
            gauss(x, mu_bg2, a_bg2, sigma_bg2) +
            peakfunc(x, mu1, a1, sigma1) +      
            peakfunc(x, mu2, a2, sigma2) +
            peakfunc(x, mu2, a2, sigma2) +
            peakfunc(x, mu3, a3, sigma3) +
            peakfunc(x, mu4, a4, sigma4) +
            peakfunc(x, mu5, a5, sigma5) +
            peakfunc(x, mu6, a6, sigma6)
    )
    
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
    

p0 = [-0.05, 0.03,
      0.0034, 0.043, 0.002244/2, 
      0.004, 0.044, 0.001844/2,
    #   0.00188, 0.0005, 0.0000546,
      0.00224, 0.00221, 0.0000563,
      0.0026007, 0.0023902, 0.0000542,
      0.0028637, 0.00685, 0.0000665,
      0.0032253, 0.01374, 0.0000828,
      0.0038320, 0.003161, 0.0000624
      ]

lower_bounds = [-np.inf] * len(p0)
lower_bounds[7] = 0.00188
lower_bounds[8] = 0.000225
upper_bounds = [np.inf] * len(p0)
upper_bounds[7] = 0.00189
upper_bounds[8] = 0.000608


popt, pcov = curve_fit(fit_func, t, S, p0=p0)

errors = np.sqrt(np.diag(pcov))

print("Fit parameters:")
print(fr"linear offset: ({popt[0]:.4e} ± {errors[0]:.4e}) x + ({popt[1]:.4e} ± {errors[1]:.4e})")
print(fr"Background Peak 1: \mu = {popt[1+1]:.4e} ± {errors[1+1]:.4e}, a = {popt[2+1]:.4e} ± {errors[2+1]:.4e}, \sigma = {popt[3+1]:.4e} ± {errors[3+1]:.4e}")
print(fr"Background Peak 2: \mu = {popt[4+1]:.4e} ± {errors[4+1]:.4e}, a = {popt[5+1]:.4e} ± {errors[5+1]:.4e}, \sigma = {popt[6+1]:.4e} ± {errors[6+1]:.4e}")
print(fr"Peak 1: \mu = {popt[7+1]:.4e} ± {errors[7+1]:.4e}, a = {popt[8+1]:.4e} ± {errors[8+1]:.4e}, \sigma = {popt[9+1]:.4e} ± {errors[9+1]:.4e}")
print(fr"Peak 2: \mu = {popt[10+1]:.4e} ± {errors[10+1]:.4e}, a = {popt[11+1]:.4e} ± {errors[11+1]:.4e}, \sigma = {popt[12+1]:.4e} ± {errors[12+1]:.4e}")
print(fr"Peak 3: \mu = {popt[13+1]:.4e} ± {errors[13+1]:.4e}, a = {popt[14+1]:.4e} ± {errors[14+1]:.4e}, \sigma = {popt[15+1]:.4e} ± {errors[15+1]:.4e}")
print(fr"Peak 4: \mu = {popt[16+1]:.4e} ± {errors[16+1]:.4e}, a = {popt[17+1]:.4e} ± {errors[17+1]:.4e}, \sigma = {popt[18+1]:.4e} ± {errors[18+1]:.4e}")
print(fr"Peak 5: \mu = {popt[19+1]:.4e} ± {errors[19+1]:.4e}, a = {popt[20+1]:.4e} ± {errors[20+1]:.4e}, \sigma = {popt[21+1]:.4e} ± {errors[21+1]:.4e}")
# print(fr"Peak 6: \mu = {popt[22]:.4e} ± {errors[22]:.4e}, a = {popt[23]:.4e} ± {errors[23]:.4e}, \sigma = {popt[24]:.4e} ± {errors[24]:.4e}")




peak_labels = [
    "[2,1]", 
    "Crossover:\n[2,1], [2,2]",
    "[2,2]",
    "Crossover:\n[2,1], [2,3]",
    "Crossover:\n[2,2], [2,3]",
    "[2,3]"
]

plt.plot(t, S, label="Messdaten")
plt.plot(t, fit_func(t, *popt), label="Fit", color="red")
plt.annotate(peak_labels[0], xy=(peak_x := 0.00188, peak_y := fit_func(peak_x, *popt)), xytext=(peak_x+0.0002, peak_y+0.002),
                 arrowprops=dict(facecolor='black', arrowstyle='-'), fontsize=8)
for i in range(5):
    peak_x = popt[8 + i * 3]
    peak_y = fit_func(peak_x, *popt)
    plt.annotate(peak_labels[i+1], xy=(peak_x, peak_y), xytext=(peak_x+0.0002, peak_y + (2-i)*0.003),
                 arrowprops=dict(facecolor='black', arrowstyle='-'), fontsize=8)

plt.xlabel("Time [s]")
plt.ylabel("U [V]")
plt.title("Sättigungsspektroskopie")
plt.grid()
plt.legend()

plt.show()
plt.clf()

plt.plot(t, S, label="Messrohdaten")
plt.xlabel("Time [s]")
plt.ylabel("U [V]")
plt.title("Rohdaten der Sättigungsspektroskopie")
plt.grid()
plt.legend()
plt.savefig("Plots/Sättigungsspektroskopie_Rohdaten.png")


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
