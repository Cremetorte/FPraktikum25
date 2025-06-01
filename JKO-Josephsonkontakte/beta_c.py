import numpy as np
import tabulate

I_r = 48.706e-6
dI_r = 1.67e-6

I_0 = 112.53e-6
dI_0 = 0.18e-6


beta_c = np.square(4*I_0 / (np.pi * I_r))
d_beta_c = 2 * 4*I_0 / (np.pi * I_r) * np.sqrt((4 * dI_0/(np.pi * I_r))**2 + (4*I_0 * dI_r  / (np.pi * I_r**2))**2)

# print(f"beta_c = {beta_c:.5g} ± {d_beta_c:.5g}")
phi_0 = 2.067833848e-15  # Wb, magnetic flux quantum
mu_0 = 1.256637061e-6  # H/m, permeability of free space

R = np.array([6.604, 12.458, 15.770]) 

dR = np.array([0.011, 0.563, 4.3]) # m, uncertainty in R

capacity = beta_c * phi_0 / (2 * np.pi * I_0 * R[1]**2)
d_capacity = np.sqrt(
    (d_beta_c * phi_0 / (2 * np.pi * I_0 * R[1]**2)) ** 2 +
    (dI_0 * beta_c * phi_0 / (2 * np.pi * I_0**2 * R[1]**2)) ** 2
)
print(f"capacity = {capacity:.5g} ± {d_capacity:.5g} F/m²")
A = 4*4e-12  # m², cross-sectional area of the Josephson junction

cpa = capacity / A
dcpa = d_capacity / A

print(f"cpa = {cpa:.5g} ± {dcpa:.5g} F/m²")


I_0 = np.array([82.025, 112.53, 113.83]) * 1e-6
dI_0 = np.array([0.275, 0.18, 0.29]) * 1e-6

a = 4e-6
A = a**2



beta_c = 2 * np.pi * I_0 * R**2 * cpa * A / phi_0
d_beta_c = beta_c * np.sqrt(
    (dI_0 / I_0) ** 2 +
    (2 * dR / R) ** 2 +
    (dcpa / cpa) ** 2
)

print(f"beta_c = {beta_c} ± {d_beta_c}")