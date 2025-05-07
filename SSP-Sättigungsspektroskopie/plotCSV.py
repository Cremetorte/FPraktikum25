import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = "Daten/sättigungabsorptionsspektrum.csv"

data = np.loadtxt(filename, delimiter=",", skiprows=1)

STEP_SIZE = 100
MAX = int(len(data[:,0]) * 0.9)
t,S = data[:MAX:STEP_SIZE,0], data[:MAX:STEP_SIZE,1]



def gauss(x, mu, amp, sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2))


def fit_func(x, *params):
    if len(params) != 1 + 3*8:
        print("wrong number of arguments")
        
    
        
    c, mu_bg1, a_bg1, sigma_bg1, mu_bg2, a_bg2, sigma_bg2, mu1, a1, sigma1, mu2, a2, sigma2, mu3, a3, sigma3, mu4, a4, sigma4, mu5, a5, sigma5, mu6, a6, sigma6 = params
    return (c -
            gauss(x, mu_bg1, a_bg1, sigma_bg1) -
            gauss(x, mu_bg2, a_bg2, sigma_bg2) +
            gauss(x, mu1, a1, sigma1) +      
            gauss(x, mu2, a2, sigma2) +
            gauss(x, mu2, a2, sigma2) +
            gauss(x, mu3, a3, sigma3) +
            gauss(x, mu4, a4, sigma4) +
            gauss(x, mu5, a5, sigma5) +
            gauss(x, mu6, a6, sigma6)
    )
    

p0 = [0.03,
      0.0034, 0.043, 0.002244/2, 
      0.004, 0.044, 0.001844/2,
      0.0018937, 0.000473, 0.0000546/2,
      0.002244, 0.00221, 0.0000563/2,
      0.0026007, 0.0023902, 0.0000542/2,
      0.0028637, 0.00685, 0.0000665/2,
      0.0032253, 0.01374, 0.0000828/2,
      0.0038320, 0.003161, 0.0000624/2
      ]


popt, pcov = curve_fit(fit_func, t, S, p0=p0)

errors = np.sqrt(np.diag(pcov))

print("Fit parameters:")
print("Constant: ", popt[0], "±", errors[0])
print(fr"Background Peak 1: \mu = {popt[1]:.4e} ± {errors[1]:.4e}, a = {popt[2]:.4e} ± {errors[2]:.4e}, \sigma = {popt[3]:.4e} ± {errors[3]:.4e}")
print(fr"Background Peak 2: \mu = {popt[4]:.4e} ± {errors[4]:.4e}, a = {popt[5]:.4e} ± {errors[5]:.4e}, \sigma = {popt[6]:.4e} ± {errors[6]:.4e}")
print(fr"Peak 1: \mu = {popt[7]:.4e} ± {errors[7]:.4e}, a = {popt[8]:.4e} ± {errors[8]:.4e}, \sigma = {popt[9]:.4e} ± {errors[9]:.4e}")
print(fr"Peak 2: \mu = {popt[10]:.4e} ± {errors[10]:.4e}, a = {popt[11]:.4e} ± {errors[11]:.4e}, \sigma = {popt[12]:.4e} ± {errors[12]:.4e}")
print(fr"Peak 3: \mu = {popt[13]:.4e} ± {errors[13]:.4e}, a = {popt[14]:.4e} ± {errors[14]:.4e}, \sigma = {popt[15]:.4e} ± {errors[15]:.4e}")
print(fr"Peak 4: \mu = {popt[16]:.4e} ± {errors[16]:.4e}, a = {popt[17]:.4e} ± {errors[17]:.4e}, \sigma = {popt[18]:.4e} ± {errors[18]:.4e}")
print(fr"Peak 5: \mu = {popt[19]:.4e} ± {errors[19]:.4e}, a = {popt[20]:.4e} ± {errors[20]:.4e}, \sigma = {popt[21]:.4e} ± {errors[21]:.4e}")
print(fr"Peak 6: \mu = {popt[22]:.4e} ± {errors[22]:.4e}, a = {popt[23]:.4e} ± {errors[23]:.4e}, \sigma = {popt[24]:.4e} ± {errors[24]:.4e}")






plt.plot(t, S, label="Absorptionsspektrum")
plt.plot(t, fit_func(t, *popt))
plt.show()
