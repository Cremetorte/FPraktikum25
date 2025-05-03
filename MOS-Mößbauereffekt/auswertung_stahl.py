import numpy as np



I_star = 3/2
I = 0.5
lambda_gamma = 1.24e-6 / 14.4e3 * 100
alpha = 9.7
t_A = 2.54e-3 # cm


sigma_0 = lambda_gamma**2/(2*np.pi) * (2*I_star + 1)/(2*I + 1) / (1 + alpha)
# cm^3



f_prime = 0.8
n_A = 8.84e22 #Kerne/cm^3
a_A = 0.022

T_A = f_prime * n_A * a_A * sigma_0 * t_A

print(f"T_A = {T_A}")
print(lambda_gamma)



gamma_exp_v = 0.423843e-3
err_gamma_exp_v = 0.02748e-3
c=299_792_458

E = 14.4e3 * gamma_exp_v/c
dE = 14.4e3 * err_gamma_exp_v/c

print(f"dE = {E:.4e} \pm {dE:.4e}")




#fehler
# Gamma / faktor
error_gamma = np.sqrt((dE/4.2)**2 + (E/4.2**2 * 0.1)**2)
print(f"Gamma real: {E/4.2:.4e} \pm {error_gamma:.4e}")

h_bar = 4.135_667_696e-15

print(f"tau = {h_bar/E:.4e} \pm {np.abs(h_bar/E**2 * dE)}")

# Literaturwert
tau_lit = h_bar/4.7e-9
print(f"Tau_lit = {tau_lit}")