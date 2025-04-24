import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")
import pandas as pd
from scipy import optimize
import tabulate

# -------------------------------------------------------------- Daten einlesen

df = pd.read_csv("Daten.csv", delimiter=',', decimal='.')
data = df.to_numpy().transpose()

t_a,A_0_Ag_108,dA_0_Ag_108,lambda_Ag_108,dlambda_Ag_108,A_0_Ag_110,dA_0_Ag_110,lambda_Ag_110,dlambda_Ag_110 = data



#--------------------------------------------------------------- Erwartungen

def act_func(x, A_0, l):
    return A_0*(1-np.exp(-l*x))
    
smooth_t = np.linspace(0, np.max(t_a), 1000)    # um schöne funktionen zu plotten

# -------------------------------------------------------------- Konstanten

lambda_108 = 0.0048499
lambda_110 = 0.0282226

# -------------------------------------------------------------- Anfangsaktivitäten fitten
p0_108 = [400, lambda_108]
ag108_popt, ag108_pcov = optimize.curve_fit(act_func, t_a, A_0_Ag_108, sigma=dA_0_Ag_108, absolute_sigma=True, p0=p0_108)
ag108_errors = np.sqrt(np.diag(ag108_pcov))

p0_110 = [100, lambda_110]
ag110_popt, ag110_pcov = optimize.curve_fit(act_func, t_a, A_0_Ag_110, sigma=dA_0_Ag_110, absolute_sigma=True, p0=p0_110)
ag110_errors = np.sqrt(np.diag(ag110_pcov))

plt.errorbar(t_a, A_0_Ag_108, yerr=dA_0_Ag_108, color="black", 
             label=r"Gemessene Anfangsaktivitäten von $^{{108}}$Ag", linestyle="",
             marker="x", capsize=3, ecolor="red")
plt.plot(smooth_t, act_func(smooth_t, *ag108_popt), label="Gefittete Anfangsaktivität")
plt.legend()
plt.title("Anfangsaktivitäten von $^{{108}}$Ag")
plt.xlabel("Aktivierungszeit $t_a$ [s]")
plt.ylabel("Anfangsaktivität A$_0$[s$^{{-1}}$]")
plt.grid()

plt.savefig("Ag108_anfangsaktiv.png")


plt.clf()


plt.errorbar(t_a, A_0_Ag_110, yerr=dA_0_Ag_110, color="black", 
             label=r"Gemessene Anfangsaktivitäten von $^{{110}}$Ag", linestyle="",
             marker="x", capsize=3, ecolor="red")
plt.plot(smooth_t, act_func(smooth_t, *ag110_popt), label="Gefittete Anfangsaktivität")
plt.legend()
plt.title("Anfangsaktivitäten von $^{{110}}$Ag")
plt.xlabel("Aktivierungszeit $t_a$ [s]")
plt.ylabel("Anfangsaktivität A$_0$[s$^{{-1}}$]")
plt.grid()

# plt.show()
plt.savefig("Ag110_anfangsaktiv.png")


table = [[r"$^{{108}}$Ag"],[r"$^{{110}}$Ag"]]
for i in range(len(ag108_popt)):
    table[0].append(ag108_popt[i])
    table[0].append(ag108_errors[i])
    table[1].append(ag110_popt[i])
    table[1].append(ag110_errors[i])

print(tabulate.tabulate(table, tablefmt="latex_raw", headers=[r"Silberisotop", "Sättigungsaktivität $A_0$", "$\sigma_{{A_0}}$", "$\lambda$", "$\sigma_\lambda$"]))