import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector, Button


filename = "Daten/JJ3_IcH_001.dat"
titlestr = "Josephsonkontakt 3: $I_c$ vs $H$"

data = np.loadtxt(filename, skiprows=0)

I_sp, I_c = data[:, 0].T, data[:, 1].T

mu_0 = 1.2566373e-6
k = 0.2234
H = k/mu_0 * I_sp

# plt.plot(H*1e3, I_c*1e6, label="Messdaten")
# plt.show()


def default_view(axes):
    axes.clear()
    axes.set_title(titlestr)
    axes.set_xlabel("$H$ [kA/m]")
    axes.set_ylabel("$I_c$ [µA]")
    axes.plot(H*1e-3, I_c*1e6, label="Messdaten")
    axes.legend()
    axes.grid()
    plt.draw()
    plt.title(titlestr)
    print("Set/Reset view to default.")
    
    
last_fit_params = None

def onselect(xmin, xmax):
    xmin = xmin*1e3
    xmax = xmax*1e3
    mask = (H >= xmin) & (H <= xmax)
    xdata = H[mask]
    ydata = I_c[mask]
    
    xrange = np.linspace(np.min(xdata), np.max(xdata), 100)
    
    popt, pcov = np.polyfit(xdata, ydata, 2, cov=True)
    
    global last_fit_params
    last_fit_params = (popt, pcov)
    
    ax.plot(xrange*1e-3, np.polyval(popt, xrange)*1e6, label="Fit", color="red")
    plt.draw()
    
    
def print_root(x):
    global last_fit_params
    # print("Fit parameters:")
    if last_fit_params is not None:
        popt, pcov = last_fit_params
        errors = np.sqrt(np.diag(pcov))
        minimum = -popt[1] / (2 * popt[0])
        delta_min = np.sqrt((errors[1] / (2 * popt[0]))**2 + (errors[0] * popt[1] / (2*popt[0]**2))**2)
        print(f"{minimum:.4e}\t{delta_min:.4e}")
    else:
        print("No fit parameters available. Please select a range first.")

    
    
fig, ax = plt.subplots()
plt.subplots_adjust(right=0.8)
default_view(ax)


span = SpanSelector(ax, onselect, 'horizontal', useblit=True,interactive=True,
                        props=dict(alpha=0.2, facecolor='red'))


reset_button_ax = plt.axes([0.85, 0.40, 0.1, 0.075])
reset_button = Button(reset_button_ax, 'Reset View')
reset_button.on_clicked(lambda event: default_view(ax))

print_button_ax = plt.axes([0.85, 0.6, 0.1, 0.075])
print_button = Button(print_button_ax, 'Print Root')
print_button.on_clicked(print_root)

plt.show()


