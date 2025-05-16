import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import SpanSelector


I_list = [0, 1000, 3000, 4000, 5000, 6000, 7000, 8000, 11000, 14000, 4500, 4700, 5300, 5500, 6500]

data_dict = {}

for I in I_list:
    filename = f"Daten/A2/Spulenstrom{I:05d}mA.dat"
    data = np.loadtxt(filename, delimiter="\t", skiprows=1)
    data_dict[I] = data
    print(f"Loaded {filename}")


# Cut to positive I only
for I, data in data_dict.items():
    data_dict[I] = data[data[:, 0] > 0]

""" # Plotting the data
for I, data in data_dict.items():
    plt.plot(data[:, 0], data[:, 1], label=f"{I} mA")

plt.xlabel("Stromstärke [A]")
plt.ylabel("Voltage [V]")
plt.title("Mit Magnetfeld")
plt.legend()
plt.grid()
plt.show() """





for I in data_dict.keys():

    fig, ax = plt.subplots()
    ax.set_title(f"Linear Fit of red points for {I} mA")
    ax.set_xlabel("Stromstärke [A]")
    ax.set_ylabel("Voltage [V]")

    data = data_dict[I]
    ax.plot(data[:, 0], data[:, 1], label=f"{I} mA")
    ax.legend()
    ax.grid()


    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(data[:, 0], (xmin, xmax))
        indmax = min(len(data) - 1, indmax)

        region_x = data[indmin:indmax, 0]
        other_x = data[:indmin, 0]
        other_x = np.append(other_x, data[indmax:, 0])

        region_y = data[indmin:indmax, 1]
        other_y = data[:indmin, 1]
        other_y = np.append(other_y, data[indmax:, 1])
        ax.set_data(region_x, region_y, 'r', lw=2, label="Selected Region")
        ax.plot(other_x, other_y, 'b', lw=2, label="Unselected Region")

        popt, pcov = np.polyfit(region_x, region_y, 1, cov=True)
        perr = np.sqrt(np.diag(pcov))
        print(f"Linear Fit: m: {popt[0]:.4e} ± {perr[0]:.4e}, c: {popt[1]:.4e} ± {perr[1]:.4e}")
        
    

    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,interactive=True,
                        props=dict(alpha=0.5, facecolor='red'))
    plt.show()


    # plt.plot(data[:, 0], data[:, 1], label=f"{I} mA")
    # plt.title("Linear Fit of red points")