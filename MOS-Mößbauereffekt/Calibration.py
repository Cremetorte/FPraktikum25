import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Weicheisen = pd.read_csv("Daten/Weicheisen.csv", delimiter=',', decimal='.')
weicheisen_velocity = Weicheisen["Velocity"].to_numpy()
weicheisen_count = Weicheisen["Count"].to_numpy()


smooth_weicheisen_velocity = np.linspace(np.min(weicheisen_velocity), np.max(weicheisen_velocity), 1000)


def lorentz_6_peaks(x, c, mu1, a1, b1, mu2, a2, b2, mu3, a3, b3,
                        mu4, a4, b4, mu5, a5, b5, mu6, a6, b6):
    return (c + 
            a1 * 1/((x - mu1)**2  + b1**2/4) +
            a2 * 1/((x - mu2)**2  + b2**2/4) +
            a3 * 1/((x - mu3)**2  + b3**2/4) +
            a4 * 1/((x - mu4)**2  + b4**2/4) + 
            a5 * 1/((x - mu5)**2  + b5**2/4) +
            a6 * 1/((x - mu6)**2  + b6**2/4))
    
    
lower_bounds = [
    1000,           # y_offset
    40,  -1e6,   1, # Peak 1: mu1, a1, b1
    100, -1e6,   1, # Peak 2
    180, -1e6,   1, # Peak 3
    240, -1e6,   1, # Peak 4
    320, -1e6,   1, # Peak 5
    400, -1e6,   1  # Peak 6
]
upper_bounds = [
    1400,          # y_offset
    100,   0,   100, # Peak 1: mu1, a1, b1
    160,  0,   100, # Peak 2
    240,  0,   100, # Peak 3
    300,  0,   100, # Peak 4
    380,  0,   100, # Peak 5
    460,  0,   100  # Peak 6
]

low=0

popt_6_peaks, pcov_6_peaks = curve_fit(lorentz_6_peaks, weicheisen_velocity[low::], weicheisen_count[low::],
                          p0=[1300, 60, -30000, 14, 
                                    140, -30000, 20, 
                                    210, -30000, 20, 
                                    270, -30000, 20, 
                                    360, -30000, 20, 
                                    430, -30000, 20], 
                          bounds=(lower_bounds, upper_bounds))

# print(popt_6_peaks)


print("Fit parameters for 6 peaks (Weicheisen):")
print(f"Background: {popt_6_peaks[0]:.2f}")
print(f"Peak 1: {popt_6_peaks[1]:.2f}, {popt_6_peaks[2]:.2f}, {popt_6_peaks[3]:.2f}")
print(f"Peak 2: {popt_6_peaks[4]:.2f}, {popt_6_peaks[5]:.2f}, {popt_6_peaks[6]:.2f}")
print(f"Peak 3: {popt_6_peaks[7]:.2f}, {popt_6_peaks[8]:.2f}, {popt_6_peaks[9]:.2f}")
print(f"Peak 4: {popt_6_peaks[10]:.2f}, {popt_6_peaks[11]:.2f}, {popt_6_peaks[12]:.2f}")
print(f"Peak 5: {popt_6_peaks[13]:.2f}, {popt_6_peaks[14]:.2f}, {popt_6_peaks[15]:.2f}")
print(f"Peak 6: {popt_6_peaks[16]:.2f}, {popt_6_peaks[17]:.2f}, {popt_6_peaks[18]:.2f}")
print("")


peaks = [popt_6_peaks[1], popt_6_peaks[4], popt_6_peaks[7], popt_6_peaks[10], popt_6_peaks[13], popt_6_peaks[16]]
peak_values = [lorentz_6_peaks(p, *popt_6_peaks) for p in peaks]
peak_descriptions = [
    fr"$\mu_{i+1}={p:.1f}$ " for i, (p, v) in enumerate(zip(peaks, peak_values))
]

plt.plot(weicheisen_velocity[low::], weicheisen_count[low::], label="Data")
plt.plot(smooth_weicheisen_velocity, lorentz_6_peaks(smooth_weicheisen_velocity, *popt_6_peaks), label="Fit", color="red")
plt.scatter(peaks, peak_values, color='green', label="Peaks", marker="x")
for i, (p, v) in enumerate(zip(peaks, peak_values)):
    plt.annotate(peak_descriptions[i], xy=(p, v), xytext=(p-50, v-50),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.margins(y=0.1)
plt.title("Kalibierungsfit Weicheisen")
plt.xlabel("Cassy-Signal")
plt.ylabel("Counts")
plt.legend(loc='center right')
plt.grid()

plt.savefig("Plots/Genetic_6_peaks.png")



# calibrate x axis
# v(K) = m*K + c
# m*(mu1 - mu2) != 10.65
m = 10.65/(popt_6_peaks[16] - popt_6_peaks[1])

# c -> 0.5* (mu4 - mu3) - c !=0

c = m * (popt_6_peaks[7] +  0.5 * (popt_6_peaks[10] - popt_6_peaks[7]))

print(f"v(K) = {m}*K - {c}")

weicheisen_velocity = m*weicheisen_velocity - c

smooth_weicheisen_velocity = np.linspace(np.min(weicheisen_velocity), np.max(weicheisen_velocity), 1000)

plt.clf()
plt.plot(weicheisen_velocity[low::], weicheisen_count[low::])
# plt.plot(smooth_weicheisen_velocity, lorentz_6_peaks(smooth_weicheisen_velocity, *popt_6_peaks))

plt.savefig("Plots/Genetic_6_peaks_cal.png")