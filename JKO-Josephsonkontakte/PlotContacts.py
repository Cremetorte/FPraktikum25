import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

files = ["JJ1_IVC_002_avg=100.dat", "JJ2_IVC_002_avg=100.dat", "JJ3_IVC_002_avg=100.dat"]
labels = ["Josephsonkontakt 1", "Josephsonkontakt 2", "Josephsonkontakt 3"]

files = ["Daten/" + file for file in files]


data = {}
for file in files:
    data[file] = np.loadtxt(file, skiprows=0)

for i, file in enumerate(files):
    plt.plot(
        data[file][:, 0]*10**3, data[file][:, 1]*10**6,
        label=labels[i],
        marker="x",
        markersize=1,
        linestyle=":",
        linewidth=0.5,
        color="black",
        markerfacecolor="red",
        markeredgecolor="red"
    )
    plt.xlabel('U [mV]')
    plt.ylabel('I [µA]')
    plt.title(f'IV-Kennlinie des Josephson-Kontaktes {i+1}')
    plt.grid()
    plt.legend()
    # plt.savefig(f"Plots/IVC_{labels[i]}.png", dpi=300)
    plt.show()
    plt.clf()