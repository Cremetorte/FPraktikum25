import numpy as np

peaks = [-5.340506e-3, -3.124833e-3, -0.831054e-3, 0.831055e-3, 3.073734e-3, 5.309494e-3]
err_peaks = [0.008946e-3, 0.008350e-3, 0.011093e-3, 0.010350e-3, 0.008502e-3, 0.009200e-3]

E_gamma = 14.4e3
c = 299_792_458

E_G = E_gamma/c * 0.5 * (peaks[3] - peaks[1]  +
                         peaks[4] - peaks[2]  )

print(peaks[3] - peaks[1])
print(peaks[4] - peaks[2])

dE_G = E_gamma/c * 0.5 * np.sqrt(
    err_peaks[3]**2 + err_peaks[1]**2  +
    err_peaks[4]**2 + err_peaks[2]**2
)
print(fr"E_G = {E_G:.4e} \pm {dE_G:.4e}")


E_A = E_gamma/c * 0.25 * (
    peaks[1] - peaks[0] +
    peaks[2] - peaks[1] +
    peaks[4] - peaks[3] +
    peaks[5] - peaks[4]
)
print(peaks[1] - peaks[0])
print(peaks[2] - peaks[1])
print(peaks[4] - peaks[3])
print(peaks[5] - peaks[4])


dE_A = E_gamma/c * 0.25 * (
    peaks[1]**2 + peaks[0]**2 +
    peaks[2]**2 + peaks[1]**2 +
    peaks[4]**2 + peaks[3]**2 +
    peaks[5]**2 + peaks[4]**2
)




print(fr"E_A = {E_A:.4e} \pm {dE_A:.4e}")



mu_k = 3.152_451_254_17e-8
mu_g = 0.0903 * mu_k
dmu_g = 0.0007* mu_k

B = E_G / (2 * mu_g)
dB = np.sqrt((dE_G / (2 * mu_g))**2 + (E_G * dmu_g / (2 * mu_g**2))**2)

print(fr"B = {B:.4e} \pm {dB:.4e}")



mu_a = 3/2 * E_A / B
dmu_a = 3/2 * np.sqrt(
    (dE_A / B)**2 +
    (E_A * dB / B**2)**2
)
print(fr"mu_a = {mu_a:.4e} \pm {dmu_a:.4e}")