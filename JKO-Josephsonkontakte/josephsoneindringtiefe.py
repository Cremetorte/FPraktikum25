import numpy as np
import tabulate

I_0 = np.array([82.025, 112.53, 113.83]) * 1e-6
dI_0 = np.array([0.275, 0.18, 0.29]) * 1e-6

a = 4e-6
A = a**2

j_0 = I_0 / A
d_j_0 = dI_0 / A


# table = list(zip(j_0, d_j_0))
# headers = ["j_0 (A/m^2)", "d_j_0 (A/m^2)"]
# print(tabulate.tabulate(table, headers=headers, floatfmt=".5g", tablefmt="latex_raw"))

lambda_L = np.array([88.550e-9, 96.579e-9, 99.804e-9])
dlambda_L = np.array([4.090e-9, 4.731e-9, 4.623e-9])

phi_0 = 2.067833848e-15  # Wb, magnetic flux quantum
mu_0 = 1.256637061e-6  # H/m, permeability of free space


inner_part = phi_0 / (4*np.pi * mu_0 * lambda_L * j_0)
lambda_J = np.sqrt(inner_part)
# Gauss error propagation for lambda_J
# lambda_J = sqrt(phi_0 / (4 * pi * mu_0 * lambda_L * j_0))
# d(lambda_J) = (1/2) * lambda_J * sqrt( (d_lambda_L/lambda_L)^2 + (d_j_0/j_0)^2 )

dlambda_J = 0.5 * lambda_J * np.sqrt((dlambda_L / lambda_L)**2 + (d_j_0 / j_0)**2)

table = list(zip(lambda_J, dlambda_J))
headers = ["lambda_J (m)", "d_lambda_J (m)"]
print(tabulate.tabulate(table, headers=headers, floatfmt=".5g", tablefmt="latex_raw"))