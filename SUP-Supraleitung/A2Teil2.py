import numpy as np
import matplotlib.pyplot as plt

R_rest = 2.118
dR_rest = 0.015

# linear fit parameters
m = 1.19238e-02 * 1e3
dm = 8.91499e-04    *1e3
c = -1.06131e+00
dc = 1.77094e-01

# Kritisches Feld
B_c2 = (R_rest - c) / m
dB_c2 = np.sqrt((dm * (R_rest - c) / m**2)**2 + (dc / m)**2 + (dR_rest / m)**2)

print(f"B_c2 = {B_c2:.4e} ± {dB_c2:.4e}")

# B_c2 = B_c2 * 1e-3


phi_0 = 2.067833848e-15
rho_rest = 2.056e-9
d_rho_rest = 2.7e-11

eta = phi_0 * B_c2 /rho_rest
d_eta = np.sqrt((phi_0 * dB_c2 / rho_rest)**2 + (phi_0 * B_c2 * d_rho_rest / rho_rest**2)**2)
print(f"eta = {eta:.4e} ± {d_eta:.4e}")




xi = np.sqrt(  np.sqrt(3) * 0.95 * phi_0 / np.pi**2 / B_c2)
d_xi = 1/(2* np.sqrt(  np.sqrt(3) * 0.95 * phi_0 / np.pi**2 / B_c2)) * np.sqrt(3) * 0.95 * phi_0 / np.pi**2 / B_c2 * dB_c2
print(f"xi = {xi:.4e} ± {d_xi:.4e}")

kappa = 0.931
dkappa = 1.171e-3
lambda_l = kappa*xi
d_lambda = np.sqrt((kappa * d_xi)**2 + (xi * dkappa)**2)

print(f"lambda_l = {lambda_l:.4e} ± {d_lambda:.4e}")#

x_val = 4*lambda_l
x = np.linspace(-x_val, x_val, 100)

B = np.exp(-x**2 / (lambda_l**2))
phi = 1 - np.exp(-x**2 / (xi**2))

plt.plot(x, B, color="r", label=r"$\frac{B(x)}{B(x=0)}$")
plt.plot(x, phi, color="b", label=r"$\frac{\Phi(x)}{\Phi(x=0)}$")
# plt.axhline(1/np.e, color="black", lw=0.5, linestyle="--")
plt.axvline(lambda_l, color="r", lw=1, linestyle="dotted", label=r"$\lambda_l$")
plt.axvline(-lambda_l, color="r", lw=1, linestyle="dotted")
plt.axvline(xi, color="b", lw=1, linestyle="dotted", label=r"$\xi$")
plt.axvline(-xi, color="b", lw=1, linestyle="dotted")

plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$\frac{B(x)}{B(x=0)}$, $\frac{\Phi(x)}{\Phi(x=0)}$")
plt.title(r"Magnetfeld und Ordnungsparameter")
plt.legend()
plt.grid()
plt.savefig("A2Teil2.png", dpi=300)