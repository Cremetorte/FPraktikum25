import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import Slider

Weicheisen = pd.read_csv("Daten/Weicheisen.csv", delimiter=',', decimal='.')
weicheisen_velocity = Weicheisen["Velocity"].to_numpy()
weicheisen_count = Weicheisen["Count"].to_numpy()


smooth_weicheisen_velocity = np.linspace(np.min(weicheisen_velocity), np.max(weicheisen_velocity), 1000)


def lorentz_1_peak(x, c, mu, a, b):
    return c + a/((x-mu)**2 + b**2)


display_params = [1000, 100, -1000, 10]

# Plot vorbereiten
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)
data_plot = ax.scatter(weicheisen_velocity, weicheisen_count, label='Daten')
model_line, = ax.plot(smooth_weicheisen_velocity, lorentz_1_peak(smooth_weicheisen_velocity, *display_params), color='red', label='Modell')
ax.legend()
ax.set_title("Manuelles Fitting mit Slidern")

# Slider-Achsen
slider_axes = [
    plt.axes([0.25, 0.35 - i * 0.05, 0.65, 0.03])
    for i in range(4)
]

# Slider erstellen
slider_mu = Slider(slider_axes[0], 'mu', 0.1, 5.0, valinit=display_params[0])
slider_a = Slider(slider_axes[1], 'a', 0.1, 2.0, valinit=display_params[1])
slider_b = Slider(slider_axes[2], 'b', -1.0, 2.0, valinit=display_params[2])
slider_c = Slider(slider_axes[3], 'c', -2.0, 3.0, valinit=display_params[3])


# Update-Funktion
def update(val):
    a = slider_a.val
    b = slider_b.val
    c = slider_c.val
    mu = slider_mu.val
    model_line.set_ydata(lorentz_1_peak(smooth_weicheisen_velocity, c, mu, a, b))
    fig.canvas.draw_idle()

# Slider-Callbacks
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_c.on_changed(update)
slider_mu.on_changed(update)

plt.show()

