import numpy as np

Is = [3.94, 3.87, 3.82, 3.99, 3.81, 3.89, 3.28, 3.65, 3.77, 3.51, 3.85, 3.83, 3.89, 3.93, 3.84]

def B(I):
    B = -0.2924*I**3 + 3.5679*I**2 + 62.3677*I + 94.2098
    return B/1000

Bs = [B(I) for I in Is]

B_mean = np.mean(Bs)
B_std = np.std(Bs)
print(f"B = {B_mean*1000:.1f} ± {B_std*1000:.1f} mT")

mu_B = 9.274e-24  # Bohr magneton in J/T
h = 6.626e-34  # Planck's constant in J·s
DeltaNu = 3.3595e10 # Hz

g = DeltaNu * h / (4*mu_B*B_mean)  # g-factor calculation
g_err = DeltaNu * h / (4*mu_B*B_mean**2) * B_std  # Error in g-factor
print(f"g = {g:.4f} ± {g_err:.4f}")




print(r"\begin{tabular}{cc}")
print(r"$I$ / A & $B$ / T \\")
print(r"\hline")
for I_val, B_val in zip(Is, Bs):
    print(f"{I_val:.2f} & {B_val:.4f} \\\\")
print(r"\end{tabular}")