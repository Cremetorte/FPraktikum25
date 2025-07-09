import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("qtagg")

# file = "Daten/l=0,l=1.csv"
files = ["Daten/l=0.csv", "Daten/l=1.csv"]
file = files[1]  # Change to the desired file


print(f"Processing file: {file}")

data = np.loadtxt(file, delimiter=',', skiprows=4)

t = data[:, 0].flatten()  # Time in s
channel_1 = data[:, 1].flatten()  # Kanal 1
channel_1 /= np.max(channel_1)
channel_2 = data[:, 2].flatten()  # Kanal 2
channel_3 = data[:, 3].flatten()  # Kanal 3



peak_1_bounds = [0, 0.0018]
peak_1_prime_bounds = [0.02, 0.025]


peak_1_mask = (t >= peak_1_bounds[0]) & (t <= peak_1_bounds[1])
peak_1_idx = np.argmax(channel_1[peak_1_mask])
peak_1 = t[peak_1_mask][peak_1_idx]

peak_1_prime_mask = (t >= peak_1_prime_bounds[0]) & (t <= peak_1_prime_bounds[1])
peak_1_prime_idx = np.argmax(channel_1[peak_1_prime_mask])
peak_1_prime = t[peak_1_prime_mask][peak_1_prime_idx]




peaks_low_bounds = [0.008, 0.012, 0.015, 0.0189]
peaks_high_bounds = [0.0087, 0.0124, 0.01575, 0.019]
peaks = []

for low, high in zip(peaks_low_bounds, peaks_high_bounds):
    # print(f"Searching for peak in range: {low} to {high}")
    peak_mask = (t >= low) & (t <= high)
    peak_idx = np.argmax(channel_1[peak_mask])
    peak = t[peak_mask][peak_idx]
    peaks.append(peak)

peaks_labels = ["(4,1)", "(3,1)", "(2,1)", "(1,1)"]


umrechnugsfaktor = 5.2230e+11
# URF * t(peak_1) + c = 0 => c = -URF * t(peak_1)
c = -umrechnugsfaktor * peak_1

t = umrechnugsfaktor * t + c  # Convert time to frequency in GHz


peaks = [umrechnugsfaktor * peak + c for peak in peaks]
peak_1 = umrechnugsfaktor * peak_1 + c
peak_1_prime = umrechnugsfaktor * peak_1_prime + c


# export for table generation
print("Peak Dictionary:")
string = f'{{"(0,0)": {peak_1}, '
for i in range(l := len(peaks)):
    string += f'"{peaks_labels[l-i-1]}": {peaks[l-i-1]}, '
string += f'"(0,0)_prime": {peak_1_prime}}}'
print(string)
print("")

print(f"Umrechnung: {umrechnugsfaktor} Hz/s * t + {c} Hz\n")










# umrechnugsfaktor /= 1e9  # Convert frequency to GHz for plotting
factor = 1e-9

plt.plot(t*factor, channel_1, label='Kanal 1', color='blue')
plt.vlines(peak_1*factor, ymin=min(channel_1), ymax=max(channel_1*1.2), color='red', linestyle='--', label='(0,1)')
plt.vlines(peak_1_prime*factor, ymin=min(channel_1), ymax=max(channel_1*1.2), color='red', linestyle='--')
plt.annotate("(0,1)", xy=(peak_1*factor, max(channel_1)), xytext=(3, 10), textcoords='offset points', 
                 arrowprops=dict(arrowstyle='-', color='red'), color='red', fontsize=10)
plt.annotate("(0,1)", xy=(peak_1_prime*factor, max(channel_1)), xytext=(3, 10), textcoords='offset points', 
                 arrowprops=dict(arrowstyle='-', color='red'), color='red', fontsize=10)



for i, peak in enumerate(peaks):
    plt.vlines(peak*factor, ymin=min(channel_1), ymax=max(channel_1*1.2), color='green', linestyle='--', label=f'{peaks_labels[i]}')
    plt.annotate(peaks_labels[i], xy=(peak*factor, max(channel_1)), xytext=(3, 10), textcoords='offset points', 
                 arrowprops=dict(arrowstyle='-', color='green'), color='green', fontsize=10)

plt.xlabel('Frequency (GHz)')
plt.ylabel('Relative Intensity')
plt.title('Resonator spectrum for l=1') 
plt.legend(loc='upper right')
plt.grid()


plt.show()