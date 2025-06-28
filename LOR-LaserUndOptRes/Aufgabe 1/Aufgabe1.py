# import matplotlib
# matplotlib.use('Agg')  # Remove or comment out this line
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("qtagg")

file = "Daten/Messung1.csv"

data = np.loadtxt(file, delimiter=',', skiprows=3)

t = data[:, 0].flatten()  # Time in s
channel_1 = data[:, 1].flatten()  # Kanal 1
channel_2 = data[:, 2].flatten()  # Kanal 2
channel_3 = data[:, 3].flatten()  # Kanal 3


peak_1_bounds = [0.0074, 0.0076]
peak_1_prime_bounds = [0.0258, 0.026]


peak_1_mask = (t >= peak_1_bounds[0]) & (t <= peak_1_bounds[1])
peak_1_idx = np.argmax(channel_1[peak_1_mask])
peak_1 = t[peak_1_mask][peak_1_idx]

peak_1_prime_mask = (t >= peak_1_prime_bounds[0]) & (t <= peak_1_prime_bounds[1])
peak_1_prime_idx = np.argmax(channel_1[peak_1_prime_mask])
peak_1_prime = t[peak_1_prime_mask][peak_1_prime_idx]


peak_2_bounds = [0.0092, 0.0094]

peak_2_mask = (t >= peak_2_bounds[0]) & (t <= peak_2_bounds[1])
peak_2_idx = np.argmax(channel_1[peak_2_mask])
peak_2 = t[peak_2_mask][peak_2_idx]

delta_t_nonprime = peak_2 - peak_1


peak_2_prime_bounds = [0.027, 0.0272]

peak_2_prime_mask = (t >= peak_2_prime_bounds[0]) & (t <= peak_2_prime_bounds[1])
peak_2_prime_idx = np.argmax(channel_1[peak_2_prime_mask])
peak_2_prime = t[peak_2_prime_mask][peak_2_prime_idx]

delta_t_prime = peak_2_prime - peak_1_prime

delta_t = (delta_t_nonprime + delta_t_prime) / 2




time_resolution = (t[100] - t[0]) / 100  # Assuming uniform time steps

distance_between_peaks = peak_1_prime - peak_1
error_distance_between_peaks = t
print(f"Abstand zwischen den Peaks: {distance_between_peaks:.6f} ± {time_resolution:.6f} s")



# calibrate time axis to frequency axis
length_of_cavity = (18 - (5.04 - 2.68)) *1e-3 #m  = 18 - 5.04 + 2.68
d_length_of_cavity =  np.sqrt(0.0002**2 + 0.0002**2) # Effective length of the cavity
print(f"Länge des Resonators: {length_of_cavity:.4e} ± {d_length_of_cavity:.4e} m")


c = 299_792_458  # Speed of light in m/s
nu_fsr = c / (2 * length_of_cavity)
d_nu_fsr = nu_fsr * (d_length_of_cavity / length_of_cavity)
print(f"FSR: {nu_fsr:.4e} ± {d_nu_fsr:.4e} Hz")


umrechnugsfaktor =  nu_fsr / distance_between_peaks
error_umrechnungsfaktor = d_nu_fsr / distance_between_peaks
print(f"Umrechnungsfaktor: {umrechnugsfaktor:.4e} ± {error_umrechnungsfaktor:.4e} s/Hz")


frequency_d = delta_t * umrechnugsfaktor
error_frequency_d = time_resolution * error_umrechnungsfaktor
print(f"Frequenzdifferenz: {frequency_d:.4e} ± {error_frequency_d:.4e} Hz")



d_laser = nu_fsr / frequency_d * length_of_cavity
d_d_laser = d_laser * (error_frequency_d / frequency_d)
print(f"Laserlänge: {d_laser:.4e} ± {d_d_laser:.4e} m")

umrechnugsfaktor /= 1e9 # Convert to s/GHz for better readability
# umrechnugsfaktor = 1

# Plotting the data
plt.plot(t * umrechnugsfaktor, channel_1, label='Channel 1', color='blue')
plt.vlines(peak_1*umrechnugsfaktor, ymin=min(channel_1), ymax=max(channel_1*1.2), color='red', linestyle='--', label='Peak 1')
plt.vlines(peak_1_prime*umrechnugsfaktor, ymin=min(channel_1), ymax=max(channel_1*1.2), color='red', linestyle='--')
plt.vlines(peak_2*umrechnugsfaktor, ymin=min(channel_1), ymax=max(channel_1*1.2), color='green', linestyle='--', label='Peak 2')
plt.vlines(peak_2_prime*umrechnugsfaktor, ymin=min(channel_1), ymax=max(channel_1*1.2), color='green', linestyle='--')
plt.annotate(
    '', 
    xy=(peak_1*umrechnugsfaktor, max(channel_1)*1.1), 
    xytext=(peak_1_prime*umrechnugsfaktor, max(channel_1)*1.1),
    arrowprops=dict(arrowstyle='<->', color='red', lw=2)
)
plt.text(
    (peak_1*umrechnugsfaktor + peak_1_prime*umrechnugsfaktor) / 2,
    max(channel_1)*1.11,
    r'$\nu_{fsr}$',
    color='red',
    ha='center',
    va='bottom'
)

plt.annotate(
    '', 
    xy=(peak_1*umrechnugsfaktor, max(channel_1)*1.2), 
    xytext=(peak_2*umrechnugsfaktor, max(channel_1)*1.2),
    arrowprops=dict(arrowstyle='<->', color='green', lw=2)
)
plt.text(
    (peak_1*umrechnugsfaktor + peak_2*umrechnugsfaktor) / 2,
    max(channel_1)*1.21,
    r'$\nu_{laser}$',
    color='green',
    ha='center',
    va='bottom'
)

plt.annotate(
    '', 
    xy=(peak_1_prime*umrechnugsfaktor, max(channel_1)*1.2), 
    xytext=(peak_2_prime*umrechnugsfaktor, max(channel_1)*1.2),
    arrowprops=dict(arrowstyle='<->', color='green', lw=2)
)
plt.text(
    (peak_1_prime*umrechnugsfaktor + peak_2_prime*umrechnugsfaktor) / 2,
    max(channel_1)*1.21,
    r'$\nu_{laser}$',
    color='green',
    ha='center',
    va='bottom'
)

plt.xlabel('Frequency (GHz)')
plt.ylabel('Voltage (V)')
plt.title('Calibrated Channel 1 Data')
plt.grid()
plt.legend()
# plt.savefig("Plots/messung1.png")  # This will display the plot interactively
plt.show()


