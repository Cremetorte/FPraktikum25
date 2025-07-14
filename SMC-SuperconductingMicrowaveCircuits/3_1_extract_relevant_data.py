import numpy
from tabulate import tabulate

res_folder = "results/Nb_fit_result"

freqs = [3.24, 3.61, 4.05, 5.22]
files = [f"{res_folder}/fit_result_{f}GHz.csv" for f in freqs]

means_freq = []
stds_freq = []

means_qint = []
stds_qint = []

means_qext = []
stds_qext = []

means_linewidth = []
stds_linewidth = []

for file in files:
    data = numpy.genfromtxt(file, delimiter=',', names=True)

    f_0 = data["f_0"]
    qint = data["Q_int"]
    qext = data["Q_ext"]

    linewidth = f_0 * (1/ qint + 1 / qext)

    means_freq.append(numpy.mean(f_0))
    stds_freq.append(numpy.std(f_0))

    means_qint.append(numpy.mean(qint))
    stds_qint.append(numpy.std(qint))

    means_qext.append(numpy.mean(qext))
    stds_qext.append(numpy.std(qext))

    means_linewidth.append(numpy.mean(linewidth))
    stds_linewidth.append(numpy.std(linewidth))


# Prepare table data
table = []
for i, freq in enumerate(freqs):
    table.append([
        f"{means_freq[i]*1e-9:.3f}",
        f"{means_qint[i]:.1f} ± {stds_qint[i]:.1f}",
        f"{means_qext[i]:.1f} ± {stds_qext[i]:.1f}",
        f"{means_linewidth[i]*1e-6:.5f} ± {stds_linewidth[i]*1e-6:.5f}"
    ])

headers = ["Frequency [GHz]", "Q$_\\text{int}$", "Q$_\\text{ext}$", "Linewidth [MHz]"]
print(tabulate(table, headers=headers, tablefmt="latex_raw", floatfmt=".4e"))