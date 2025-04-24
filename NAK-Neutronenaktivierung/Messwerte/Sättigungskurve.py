import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd

# -------------------------------------------------------------- Daten einlesen

df = pd.read_csv("Daten.csv", delimiter=',', decimal='.')
data = df.to_numpy().transpose()

t_a,A_0_Ag_108,dA_0_Ag_108,lambda_Ag_108,dlambda_Ag_108,A_0_Ag_110,dA_0_Ag_110,lambda_Ag_110,dlambda_Ag_110 = data



#--------------------------------------------------------------- Erwartungen

def act_func(x, A_0, l):
    return A_0*(1-np.exp(-l*x))
    
smooth_t = np.linspace(0, np.max(t_a), 1000)    # for smooth fuctions

# -------------------------------------------------------------- Konstanten

lambda_108 = 0.0048499
lambda_110 = 0.0282226

# -------------------------------------------------------------- Anfangsaktivitäten

plt.errorbar(t_a, A_0_Ag_108, yerr=dA_0_Ag_108, color="black", 
             label=r"Gemessene Anfangsaktivitäten von $^{{108}}$Ag", linestyle="",
             marker="x", capsize=3, ecolor="red")
plt.legend()



plt.clf()


plt.errorbar(t_a, A_0_Ag_110, yerr=dA_0_Ag_110, color="black", 
             label=r"Gemessene Anfangsaktivitäten von $^{{110}}$Ag", linestyle="",
             marker="x", capsize=3, ecolor="red")

plt.savefig("Test")
