import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -------------------------------------------------------------- Daten einlesen

FeSO4 = pd.read_csv("Daten/FeSO4.csv", delimiter=',', decimal='.')
FeSO4_velocity = FeSO4["Velocity"].to_numpy()
FeSO4_count = FeSO4["Count"].to_numpy()

Stahl = pd.read_csv("Daten/Stahl.csv", delimiter=',', decimal='.')
stahl_velocity = Stahl["Velocity"].to_numpy()
stahl_count = Stahl["Count"].to_numpy()

Weicheisen = pd.read_csv("Daten/Weicheisen.csv", delimiter=',', decimal='.')
weicheisen_velocity = Weicheisen["Velocity"].to_numpy()
weicheisen_count = Weicheisen["Count"].to_numpy()


smooth_weicheisen_velocity = np.linspace(np.min(weicheisen_velocity), np.max(weicheisen_velocity), 1000)

smooth_FeSO4_velocity = np.linspace(np.min(FeSO4_velocity), np.max(FeSO4_velocity), 1000)

smooth_stahl_velocity = np.linspace(np.min(stahl_velocity), np.max(stahl_velocity), 1000)

# -------------------------------------------------------------- Funktionen

def velocity_to_energy(velocitys):
    return velocitys

def lorentz_1_peak(x, c, mu, a, b):
    return c + a/((x-mu)**2 + b**2)

def lorentz_2_peaks(x, c, mu1, a1, b1, mu2, a2, b2):
    return c + a1/((x-mu1)**2 + b1**2) + a2/((x-mu2)**2 + b2**2)

def lorentz_6_peaks(x, c, mu1, a1, b1, mu2, a2, b2, mu3, a3, b3,
                        mu4, a4, b4, mu5, a5, b5, mu6, a6, b6):
    return (c + a1/((x-mu1)**2 + b1**2) + 
            a2/((x-mu2)**2 + b2**2) +
            a3/((x-mu3)**2 + b3**2) +
            a4/((x-mu4)**2 + b4**2) +
            a5/((x-mu5)**2 + b5**2) + 
            a6/((x-mu6)**2 + b6**2))


# -------------------------------------------------------------- Fit
# Fit the data
popt_1_peak, pcov_1_peak = curve_fit(lorentz_1_peak, stahl_velocity, stahl_count,
                          p0=[700, 250, -50, 0.5])

popt_2_peaks, pcov_2_peaks = curve_fit(lorentz_2_peaks, FeSO4_velocity, FeSO4_count,
                          p0=[1200, 250, -50, 0.5, 350, -50, 0.5])

# popt_6_peaks, pcov_6_peaks = curve_fit(lorentz_6_peaks, stahl_velocity, stahl_count,
#                           p0=[1300, 80, -100, 500, 
#                                     150, -100, 500, 
#                                     230, -100, 500, 
#                                     280, -100, 500, 
#                                     350, -100, 500, 
#                                     420, -100, 500],
#                           bounds=([1150  ,  70, -20000, 0,    145, -20000, 0,   220, -10000, 0,
#                                            270, -10000, 0,    340, -20000, 0,   410, -20000, 0], 
#                                   [1301, 100,  0, 600,    170,  0, 600,   230,  0, 600,
#                                            280,  0, 600,    360,  0, 600,   425,  0, 600])
#                           )


# print("Fit parameters for 6 peaks (FeSO4):")
# print(f"Background: {popt_6_peaks[0]:.2f}")
# print(f"Peak 1: {popt_6_peaks[1]:.2f}, {popt_6_peaks[2]:.2f}, {popt_6_peaks[3]:.2f}")
# print(f"Peak 2: {popt_6_peaks[4]:.2f}, {popt_6_peaks[5]:.2f}, {popt_6_peaks[6]:.2f}")
# print(f"Peak 3: {popt_6_peaks[7]:.2f}, {popt_6_peaks[8]:.2f}, {popt_6_peaks[9]:.2f}")
# print(f"Peak 4: {popt_6_peaks[10]:.2f}, {popt_6_peaks[11]:.2f}, {popt_6_peaks[12]:.2f}")
# print(f"Peak 5: {popt_6_peaks[13]:.2f}, {popt_6_peaks[14]:.2f}, {popt_6_peaks[15]:.2f}")
# print(f"Peak 6: {popt_6_peaks[16]:.2f}, {popt_6_peaks[17]:.2f}, {popt_6_peaks[18]:.2f}")



plt.plot(FeSO4_velocity, FeSO4_count)
plt.plot(smooth_FeSO4_velocity, lorentz_2_peaks(smooth_FeSO4_velocity, *popt_2_peaks), color="red", label="Fit")
plt.legend()
plt.grid()
plt.savefig("Plots/FeSO4_Fit.png")
plt.clf()



plt.plot(stahl_velocity, stahl_count)
plt.plot(smooth_stahl_velocity, lorentz_1_peak(smooth_stahl_velocity, *popt_1_peak), color="red", label="Fit")
plt.legend()
plt.grid()
plt.savefig("Plots/Stahl_Fit.png")
plt.clf()





# ------------------------------------------------------------------------ Prepare 6 peaks

cuts = [0, 125-30, 195-30, 250-30, 310-30, 385-30, len(weicheisen_velocity)]

p0 = [1200, ]

for i in range(1, len(cuts)):
    cut_we_velocity = weicheisen_velocity[cuts[i-1]:cuts[i]]
    cut_we_count = weicheisen_count[cuts[i-1]:cuts[i]]
    p0_temp = [1200, 0.5*(cuts[i]-cuts[i-1]), -50, 0.5]
    popt, pcov = curve_fit(lorentz_1_peak, cut_we_velocity, cut_we_count, p0=p0_temp)
    p0.append(popt[1])
    p0.append(popt[2])
    p0.append(popt[3])
    plt.plot(cut_we_velocity, cut_we_count)
    plt.plot(smooth_weicheisen_velocity, lorentz_1_peak(smooth_weicheisen_velocity, *popt), color="red", label="Fit")
    plt.legend()
    plt.grid()
    plt.savefig(f"Plots/Weicheisen_Fit_peak{i}.png")
    plt.clf()
    
    
    
print(p0)


popt_6_peaks, pcov_6_peaks = curve_fit(lorentz_6_peaks, stahl_velocity, stahl_count,
                          p0=p0
                          )



plt.plot(weicheisen_velocity, weicheisen_count)
plt.plot(smooth_weicheisen_velocity, lorentz_6_peaks(smooth_weicheisen_velocity, *popt_6_peaks), color="red", label="Fit")
plt.legend()
plt.grid()
plt.savefig("Plots/Weicheisen_Fit.png")