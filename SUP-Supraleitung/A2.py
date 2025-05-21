import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.widgets import SpanSelector, Button



I_list = [0, 1000, 3000, 4000, 5000, 6000, 7000, 8000, 11000, 14000, 4500, 4700, 5300, 5500, 6500]
# I_list = [0, 4000,7000]

data_dict = {}

for I in I_list:
    filename = f"Daten/A2/Spulenstrom{I:05d}mA.dat"
    data = np.loadtxt(filename, delimiter="\t", skiprows=1)
    data_dict[I] = data
    print(f"Loaded {filename}")


# Cut to positive I only
for I, data in data_dict.items():
    data_dict[I] = data[data[:, 0] > 0]

# # Plotting the data
# for I, data in data_dict.items():
#     plt.plot(data[:, 0], data[:, 1], label=f"{I} mA")

# plt.xlabel("Probenstrom [A]")
# plt.ylabel("Spannung [V]")
# plt.title("Mit Magnetfeld")
# plt.legend()
# plt.grid()
# plt.show()



# plt.rcParams['figure.dpi'] = 300

fit_res = {}

for I in data_dict.keys():

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_title(f"Linearer Fit der rot gekennzeichneten Punkte bei {I} mA Spulenstrom")
    ax.set_xlabel("Probenstrom [A]")
    ax.set_ylabel("Spannung [V]")

    data = data_dict[I]
    ax.plot(data[:, 0], data[:, 1], label=f"{I} mA")
    ax.legend()
    ax.grid()
    
    bounds = (np.min(data[:, 0]), np.max(data[:, 0]))
    
    fit_temp = [None]
    def onselect(xmin, xmax):
        # indmin, indmax = np.searchsorted(data[:, 0], (xmin, xmax))
        # indmax = min(len(data) - 1, indmax)
        
        indmin = int((xmin/np.max(data[:,0])) * len(data))
        indmax = int((xmax/np.max(data[:,0])) * len(data))
        print(f"Selected indices: {indmin}, {indmax}")
        print(f"Selected range: {xmin:.4f} to {xmax:.4f}")
        
        # indmin = int(xmin)
        # indmax = int(xmax)

        region_x = data[indmin:indmax, 0]
        other_x = data[:indmin, 0]
        other_x = np.append(other_x, data[indmax:, 0])

        region_y = data[indmin:indmax, 1]
        other_y = data[:indmin, 1]
        other_y = np.append(other_y, data[indmax:, 1])
        ax.cla()
        ax.plot(region_x, region_y, 'r', marker=".", linestyle="", lw=1, label="Berücksichtigte Region")
        ax.plot(other_x, other_y, 'b', marker=".", linestyle="", lw=1, label="Nicht berücksichtigte Region")

        popt, pcov = np.polyfit(region_x, region_y, 1, cov=True)
        perr = np.sqrt(np.diag(pcov))
        
        
        fit_temp[0] = [I, popt[0], perr[0], popt[1], perr[1]]


        # Print output in tab-separated format for easy copy to Excel
        print(f"{I}\t{float(popt[0]):.6e}\t{float(perr[0]):.6e}\t{float(popt[1]):.6e}\t{float(perr[1]):.6e}\n")
        
        
        ax.plot(data[:, 0], popt[0] * data[:, 0] + popt[1], 'g--', label="Linearer Fit")
        ax.set_title(f"Linearer Fit der rot gekennzeichneten Punkte bei {I} mA Spulenstrom")
        ax.set_xlabel("Probenstrom [A]")
        ax.set_ylabel("Spannung [V]")
        ax.grid()
        ax.legend()
        ax.set_xlim(bounds)
        ax.set_ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        plt.draw()

    i = 0
    fit_res[I] = []
    def on_button_clicked(event):
        if fit_temp[0] == None:
            print("No fit data available. Please select a region first.")
            return
        global i
        fit_res[I].append([*fit_temp, i])
        i += 1
        print(f"Saved fit data for {I} mA")
    

    ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
    button = Button(ax_button, "Fit ausgeben")
    button.on_clicked(on_button_clicked)
    
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,interactive=True,
                        props=dict(alpha=0.2, facecolor='red'))
    plt.show()


    # plt.plot(data[:, 0], data[:, 1], label=f"{I} mA")
    # plt.title("Linear Fit of red points")
    
    
print("Fit results:")
for I, res_list in fit_res.items():
    for i, res in enumerate(res_list):
        res = res[0]
        print(f"{I}\t{float(res[0]):.6e}\t{float(res[1]):.6e}\t{float(res[2]):.6e}\t{float(res[3]):.6e}\t{float(res[4]):.6e}")
        