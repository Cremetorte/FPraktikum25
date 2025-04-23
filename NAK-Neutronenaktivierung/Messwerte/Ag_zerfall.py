import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import glob
import pandas as pd
from scipy import optimize

t_a = 5   # Aktivierungszeit
file = "AG_csv/Ag_" + str(t_a) + "s.csv"


# -------------------------------------------------------------- Daten einlesen

df = pd.read_csv(file, delimiter=';', decimal=',')
data = df.to_numpy().transpose()

std_err = np.sqrt(data[1])

# data[0] += 7.05
    
# print(data)

# ---------------------------------------------------------------------------- Funktionen

def exponential(t, amp, lamb):
    return amp*np.exp(-t*lamb)

def sum_exp(t, amp_1, amp_2, lamb_1, lamb_2, y_offset):
    return exponential(t, amp_1, lamb_1) + exponential(t, amp_2, lamb_2) + y_offset



def chi_sq(x):
    amp_1, amp_2, lamb_1, lamb_2, y_offset = x
    fraction = (data[1] - sum_exp(data[0], amp_1, amp_2, lamb_1, lamb_2, y_offset)) / std_err
    # fraction = np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0)  # Handle invalid values
    return np.sum(np.square(fraction))


def chi_sq_one_exp(x):
    amp, lamb = x
    fraction = (data[1] - exponential(data[0], amp, lamb)) / std_err
    return np.sum(np.square(fraction))

# -------------------------------------------------------------- Minimiere Chi^2
            # amp_1, amp_2, lambda_1, lambda_2, y_offset                
start_values = [1, 58, 0.00485, 0.0282,1]
bound_values = [(0,100000), (0,100000), (0,1), (0,1), (0,10)]

res = optimize.minimize(chi_sq, start_values, bounds=bound_values)
opt_par = res.x
# cov = res.hess_inv.todense()
print(opt_par)
# # print(cov)

# red_chi_sq = chi_sq(opt_par)/(len(data[0]) - 5)
# print(red_chi_sq)
# err = np.sqrt(np.diag(cov*red_chi_sq))

# print(err)


# start_values = [60,0.005]
# opt_par = optimize.minimize(chi_sq_one_exp, start_values).x
# print(opt_par)


# --------------------------------------------------------------- plot
plt.plot(data[0], data[1])
plt.plot(data[0], sum_exp(data[0], *opt_par))
plt.savefig("test.png")