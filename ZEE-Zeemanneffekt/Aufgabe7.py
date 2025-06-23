import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")  # Use TkAgg backend for plotting


file = "Daten/Radien.dat"
data = np.loadtxt(file, skiprows=1)

p = data[:, 0].flatten()
sigmaminus = data[:, 1].flatten()  # m
pi = data[:, 2].flatten()  # m
sigmaplus = data[:, 3].flatten()  # m


dR_sq_pi = np.array([pi[i+1]**2 - pi[i]**2 for i in range(len(pi)-1)])
dR_pi = np.array([pi[i+1] - pi[i] for i in range(len(pi)-1)])

dR_sq_pi_sigmaminus = np.array([sigmaminus[i]**2 - pi[i]**2 for i in range(len(sigmaminus))])
dR_pi_sigmaminus = np.array([sigmaminus[i] - pi[i] for i in range(len(sigmaminus))])

dR_sq_pi_sigmaplus = np.array([sigmaplus[i]**2 - pi[i]**2 for i in range(len(sigmaplus))])
dR_pi_sigmaplus = np.array([sigmaplus[i] - pi[i] for i in range(len(sigmaplus))])

""" 
plt.plot(p[:-1], dR_sq_pi*1e6, 'x', label=r'$\Delta R^2_\pi$')
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R^2_\pi$ [mm²]')
plt.title('Differenzen der Radien im Quadrat')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien_quadrat.png")

plt.cla()
plt.plot(p[:-1], dR_pi*1e6, 'x', label=r'$\Delta R_\pi$')
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R_\pi$ [μm]')
plt.title('Differenzen der Radien')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien.png")

plt.cla()
plt.plot(p, dR_sq_pi_sigmaminus*1e6, "x", label=r"$\delta^2 R_{\sigma^-,\pi}$")
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R^2_{\sigma^-,\pi}$ [mm²]')
plt.title('Differenzen der Radien im Quadrat')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien_quadrat,sigmaminus_pi.png")
print(fr"\delta^2 R_{{\sigma^-,\pi}} = {np.mean(dR_sq_pi_sigmaminus)} \pm {np.std(dR_sq_pi_sigmaminus)}")

plt.cla()
plt.plot(p, dR_pi_sigmaminus*1e6, "x", label=r"$\delta R_{\sigma^-,\pi}$")
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R_{\sigma^-,\pi}$ [μm]')
plt.title('Differenzen der Radien')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien,sigmaminus_pi.png")
print(fr"\\delta R_{{\\sigma^-,\\pi}} = {np.mean(dR_pi_sigmaminus)} \\pm {np.std(dR_pi_sigmaminus)}")

plt.cla()
plt.plot(p, dR_pi_sigmaplus*1e6, "x", label=r"$\delta R_{\sigma^+,\pi}$")
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R_{\sigma^+,\pi}$ [μm]')
plt.title('Differenzen der Radien')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien,sigmaplus_pi.png")
print(fr"\\delta R_{{\\sigma^+,\\pi}} = {np.mean(dR_pi_sigmaplus)} \\pm {np.std(dR_pi_sigmaplus)}")

plt.cla()
plt.plot(p, dR_sq_pi_sigmaplus*1e6, "x", label=r"$\delta^2 R_{\sigma^+,\pi}$")
plt.xlabel('Invertierte Ordnung $p$')
plt.ylabel(r'$\Delta R^2_{\sigma^+,\pi}$ [mm²]')
plt.title('Differenzen der Radien im Quadrat')
plt.legend()
plt.grid()
plt.savefig("Plots/differenzen_radien_quadrat,sigmaplus_pi.png")
print(fr"\delta^2 R_{{\sigma^-,\pi}} = {np.mean(dR_sq_pi_sigmaplus)} \pm {np.std(dR_sq_pi_sigmaplus)}")
 """


freier_spektralbereich = 33.496e9
delta_nu_splus = freier_spektralbereich * np.mean(dR_sq_pi_sigmaplus)/np.mean(dR_sq_pi)
err_delta_nu_splus = freier_spektralbereich * np.sqrt((np.std(dR_sq_pi_sigmaplus)/np.mean(dR_sq_pi))**2 + (np.mean(dR_sq_pi_sigmaplus)/np.mean(dR_sq_pi)**2 * np.std(dR_sq_pi))**2)
print(fr"delta_nu_minus = {delta_nu_splus:.5e} ± {err_delta_nu_splus:.5e} Hz")

delta_nu_sminus = freier_spektralbereich * np.mean(dR_sq_pi_sigmaminus)/np.mean(dR_sq_pi)
err_delta_nu_sminus = freier_spektralbereich * np.sqrt((np.std(dR_sq_pi_sigmaminus)/np.mean(dR_sq_pi))**2 + (np.mean(dR_sq_pi_sigmaminus)/np.mean(dR_sq_pi)**2 * np.std(dR_sq_pi))**2)
print(fr"delta_nu_plus = {delta_nu_sminus:.5e} ± {err_delta_nu_sminus:.5e} Hz")


lambda_0 = 643.8e-9  # m
n = 1.4560
d = 3e-3
delta_lambda = lambda_0**2 / (2*n*d)
print(fr"delta_lambda = {delta_lambda:.5e} m")


print("Im Wellenlängenbereich:")
freier_spektralbereich = delta_lambda
delta_nu_splus = freier_spektralbereich * np.mean(dR_sq_pi_sigmaplus)/np.mean(dR_sq_pi)
err_delta_nu_splus = freier_spektralbereich * np.sqrt((np.std(dR_sq_pi_sigmaplus)/np.mean(dR_sq_pi))**2 + (np.mean(dR_sq_pi_sigmaplus)/np.mean(dR_sq_pi)**2 * np.std(dR_sq_pi))**2)
print(fr"delta_nu_minus = {delta_nu_splus:.5e} ± {err_delta_nu_splus:.5e} Hz")

delta_nu_sminus = freier_spektralbereich * np.mean(dR_sq_pi_sigmaminus)/np.mean(dR_sq_pi)
err_delta_nu_sminus = freier_spektralbereich * np.sqrt((np.std(dR_sq_pi_sigmaminus)/np.mean(dR_sq_pi))**2 + (np.mean(dR_sq_pi_sigmaminus)/np.mean(dR_sq_pi)**2 * np.std(dR_sq_pi))**2)
print(fr"delta_nu_plus = {delta_nu_sminus:.5e} ± {err_delta_nu_sminus:.5e} Hz")
