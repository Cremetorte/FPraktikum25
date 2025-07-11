import numpy
from tabulate import tabulate

res_folder = "results/Nb_fit_result"

freqs = [3.24, 3.61, 4.05, 5.22]
files = [f"{res_folder}/fit_result_{f}GHz.csv" for f in freqs]

means_q = []
stds_q = []
means_dk = []
stds_dk = []

for file in files:
    data = numpy.genfromtxt(file, delimiter=',', names=True)

    q = data['Q_load']
    means_q.append(numpy.mean(q))
    stds_q.append(numpy.std(q))

    d_k = data['d_k']
    means_dk.append(numpy.mean(d_k))
    stds_dk.append(numpy.std(d_k))

# Prepare table data
table = []
for i, freq in enumerate(freqs):
    table.append([
        f"{freq} GHz",
        f"{means_q[i]} ± {stds_q[i]}",
        f"{means_dk[i]} ± {stds_dk[i]}"
    ])

headers = ["Frequency", "Q", "Linewidth $d_k$"]
print(tabulate(table, headers=headers, tablefmt="latex_raw", floatfmt=".4e"))