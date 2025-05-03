import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate


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

def real_velocity(velocitys):
    return 0.03151089353095451 * velocitys - 8.011193385475183

def lorentz_1_peak(x, c, mu, a, b):
    return c + a/((x-mu)**2 + b**2)

def lorentz_2_peaks(x, c, mu1, a1, b1, mu2, a2, b2):
    return c + a1/((x-mu1)**2 + b1**2/4) + a2/((x-mu2)**2 + b2**2/4)

def lorentz_6_peaks(x, c, mu1, a1, b1, mu2, a2, b2, mu3, a3, b3,
                        mu4, a4, b4, mu5, a5, b5, mu6, a6, b6):
    return (c + 
            a1/((x-mu1)**2 + b1**2/4) + 
            a2/((x-mu2)**2 + b2**2/4) +
            a3/((x-mu3)**2 + b3**2/4) +
            a4/((x-mu4)**2 + b4**2/4) +
            a5/((x-mu5)**2 + b5**2/4) + 
            a6/((x-mu6)**2 + b6**2/4))
    
    
popt_1_peak, pcov_1_peak = curve_fit(lorentz_1_peak, stahl_velocity, stahl_count,
                          p0=[700, 0, -10000, 0.5])
error_1_peak = np.sqrt(np.diag(pcov_1_peak))
print("\nFit parameters for 1 peak (Stahl):")
print(f"Background: {popt_1_peak[0]:.6f} ± {error_1_peak[0]:.6f}")
print(f"Peak 1: mu={popt_1_peak[1]:.6f}±{error_1_peak[1]:.6f}, a={popt_1_peak[2]:.6f}±{error_1_peak[2]:.6f}, b={popt_1_peak[3]:.6f}±{error_1_peak[3]:.6f}")


popt_2_peaks, pcov_2_peaks = curve_fit(lorentz_2_peaks, FeSO4_velocity, FeSO4_count,
                          p0=[1200, -0.5, -10000, 0.5, 3, -50, 0.5])
error_2_peaks = np.sqrt(np.diag(pcov_2_peaks))
print("\nFit parameters for 2 peaks (FeSO4):")
print(f"Background: {popt_2_peaks[0]:.6f} ± {error_2_peaks[0]:.6f}")
print(f"Peak 1: mu={popt_2_peaks[1]:.6f}±{error_2_peaks[1]:.6f}, a={popt_2_peaks[2]:.6f}±{error_2_peaks[2]:.6f}, b={popt_2_peaks[3]:.6f}±{error_2_peaks[3]:.6f}")
print(f"Peak 2: mu={popt_2_peaks[4]:.6f}±{error_2_peaks[4]:.6f}, a={popt_2_peaks[5]:.6f}±{error_2_peaks[5]:.6f}, b={popt_2_peaks[6]:.6f}±{error_2_peaks[6]:.6f}")



"""Peak 1: 84.75, -57533.49, 21.20
Peak 2: 155.07, -48684.49, 19.29
Peak 3: 227.86, -16945.64, 14.18
Peak 4: 280.61, -18672.95, 14.33
Peak 5: 351.78, -49717.72, 19.60
Peak 6: 422.73, -49716.86, 20.23"""

#transform to real velocity
const = 1273
peaks = [[84.75, -57533.49, 21.20], 
         [155.07, -48684.49, 19.29],
         [227.86, -16945.64, 14.18],
         [280.61, -18672.95, 14.33],
         [351.78, -49717.72, 19.60],
         [422.73, -49716.86, 20.23]]

m = 0.03151089353095451
c = 8.011193385475183


p_0 = [const]

for peak in peaks:
    peak[0] = real_velocity(peak[0])
    peak[1] = peak[1] * m/30
    peak[2] = peak[2] * m
    p_0.append(peak[0])
    p_0.append(peak[1])
    p_0.append(peak[2])
# print(p_0)
    




# # old_popt_6_peaks = [1300, 84.75, -57533.49, 21.20
# #                         155.07, -48684.49, 19.29
# #                         227.86, -16945.64, 14.18
# #                         280.61, -18672.95, 14.33
# #                         351.78, -49717.72, 19.60
# #                         422.73, -49716.86, 20.23]

# m = 0.03151089353095451
# b_start = 16*m**2
# a_start = -30000*m**2

# print("a_start:", a_start)
# print("b_start:", b_start)


popt_6_peaks, pcov_6_peaks = curve_fit(lorentz_6_peaks, weicheisen_velocity, weicheisen_count,
                          p0=p_0)

error_6_peaks = np.sqrt(np.diag(pcov_6_peaks))

print("\nFit parameters for 6 peaks (Weicheisen):")
print(f"Background: {popt_6_peaks[0]:.6f} ± {error_6_peaks[0]:.6f}")
print(f"Peak 1: mu={popt_6_peaks[1]:.6f}±{error_6_peaks[1]:.6f}, a={popt_6_peaks[2]:.6f}±{error_6_peaks[2]:.6f}, b={popt_6_peaks[3]:.6f}±{error_6_peaks[3]:.6f}")
print(f"Peak 2: mu={popt_6_peaks[4]:.6f}±{error_6_peaks[4]:.6f}, a={popt_6_peaks[5]:.6f}±{error_6_peaks[5]:.6f}, b={popt_6_peaks[6]:.6f}±{error_6_peaks[6]:.6f}")
print(f"Peak 3: mu={popt_6_peaks[7]:.6f}±{error_6_peaks[7]:.6f}, a={popt_6_peaks[8]:.6f}±{error_6_peaks[8]:.6f}, b={popt_6_peaks[9]:.6f}±{error_6_peaks[9]:.6f}")
print(f"Peak 4: mu={popt_6_peaks[10]:.6f}±{error_6_peaks[10]:.6f}, a={popt_6_peaks[11]:.6f}±{error_6_peaks[11]:.6f}, b={popt_6_peaks[12]:.6f}±{error_6_peaks[12]:.6f}")
print(f"Peak 5: mu={popt_6_peaks[13]:.6f}±{error_6_peaks[13]:.6f}, a={popt_6_peaks[14]:.6f}±{error_6_peaks[14]:.6f}, b={popt_6_peaks[15]:.6f}±{error_6_peaks[15]:.6f}")
print(f"Peak 6: mu={popt_6_peaks[16]:.6f}±{error_6_peaks[16]:.6f}, a={popt_6_peaks[17]:.6f}±{error_6_peaks[17]:.6f}, b={popt_6_peaks[18]:.6f}±{error_6_peaks[18]:.6f}")


plt.title("Daten und Fit der FeSO4-Probe")
plt.plot(FeSO4_velocity, FeSO4_count)
plt.plot(smooth_FeSO4_velocity, lorentz_2_peaks(smooth_FeSO4_velocity, *popt_2_peaks), color="red", label="Lorentz-Peak Fit")
plt.legend()
plt.grid()
plt.xlabel("Geschwindigkeit [mm/s]")
plt.ylabel("Counts")
plt.savefig("Plots/FeSO4_Fit.png")
plt.clf()


plt.title("Daten und Fit der Stahl-Probe")
plt.plot(stahl_velocity, stahl_count)
plt.plot(smooth_stahl_velocity, lorentz_1_peak(smooth_stahl_velocity, *popt_1_peak), color="red", label="Lorentz-Peak Fit")
plt.legend()
plt.grid()
plt.xlabel("Geschwindigkeit [mm/s]")
plt.ylabel("Counts")
plt.show()
plt.savefig("Plots/Stahl_Fit.png")
plt.clf()





plt.title("Daten und Fit der Weicheisen-Probe")
plt.plot(weicheisen_velocity, weicheisen_count)
plt.plot(smooth_weicheisen_velocity, lorentz_6_peaks(smooth_weicheisen_velocity, *popt_6_peaks), color="red", label="Lorentz-Peak Fit")
plt.legend()
plt.grid()
plt.xlabel("Geschwindigkeit [mm/s]")
plt.ylabel("Counts")
plt.savefig("Plots/Weicheisen_Fit.png")



# -------------------------------------------------------------- Ergebnisse in Tabelle

# Prepare data for the table
results = []

# Stahl (1 peak)
results.append(["Stahl", f"{popt_1_peak[0]:.6f} ± {error_1_peak[0]:.6f}", 
                f"{popt_1_peak[1]:.6f} ± {error_1_peak[1]:.6f}", 
                f"{popt_1_peak[2]:.6f} ± {error_1_peak[2]:.6f}", 
                f"{popt_1_peak[3]:.6f} ± {error_1_peak[3]:.6f}"])

# FeSO4 (2 peaks)
results.append(["FeSO4", f"{popt_2_peaks[0]:.6f} ± {error_2_peaks[0]:.6f}", 
                f"{popt_2_peaks[1]:.6f} ± {error_2_peaks[1]:.6f}", 
                f"{popt_2_peaks[2]:.6f} ± {error_2_peaks[2]:.6f}", 
                f"{popt_2_peaks[3]:.6f} ± {error_2_peaks[3]:.6f}"])
results.append(["", f"{popt_2_peaks[0]:.6f} ± {error_2_peaks[0]:.6f}", 
                f"{popt_2_peaks[4]:.6f} ± {error_2_peaks[4]:.6f}", 
                f"{popt_2_peaks[5]:.6f} ± {error_2_peaks[5]:.6f}", 
                f"{popt_2_peaks[6]:.6f} ± {error_2_peaks[6]:.6f}"])

# Weicheisen (6 peaks)
for i in range(6):
    results.append(["Weicheisen" if i==0 else None, f"{popt_6_peaks[0]:.6f} ± {error_6_peaks[0]:.6f}", 
                    f"{popt_6_peaks[1 + i * 3]:.6f} ± {error_6_peaks[1 + i * 3]:.6f}", 
                    f"{popt_6_peaks[2 + i * 3]:.6f} ± {error_6_peaks[2 + i * 3]:.6f}", 
                    f"{popt_6_peaks[3 + i * 3]:.6f} ± {error_6_peaks[3 + i * 3]:.6f}"])

# Print the table
headers = ["Probe", "c", "mu", "a", "b"]
print("\nFit Results:")
print(tabulate(results, headers=headers, tablefmt="latex_raw", floatfmt=".4f"))
