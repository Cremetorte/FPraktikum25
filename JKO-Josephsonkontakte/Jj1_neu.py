import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

IVC = np.loadtxt('Daten/JJ1_IVC_002_avg=100.dat', skiprows=0)
U, I = np.hsplit(IVC, 2)


mask = (U>-1e-5) & (U < 3e-2)

U = U[mask]
U = U[:int(len(U)/2):]
# U = U[::-1]
I = I[mask]
I = I[:int(len(I)/2):]#
# I = I[::-1]
sort_idx = np.argsort(U.flatten())
U = U[sort_idx]
I = I[sort_idx]

electorn_charge = 1.602176634e-19  # in Coulombs

# plt.plot(U, I, label='IV-Kennlinie', color='blue', marker="x", markersize=1, linestyle="-", linewidth=0.5)
# plt.xlabel('U [V]')
# plt.ylabel('I [A]')
# plt.title('IV-Kennlinie des Josephson-Kontaktes')
# plt.grid()
# plt.legend()
# plt.show()
selected_span = None  # To store the Rectangle patch

def onselect(xmin, xmax):
    global selected_span
    midpoint = (xmin + xmax) / 2
    width = abs(xmax - xmin)
    print(f"Selected range: xmin={xmin:.5g}, xmax={xmax:.5g}")
    print(f"Midpoint: {midpoint:.5g}, Width: {width:.5g}")

    print(f"Delta = {midpoint/2} pm {width/2} eV")

    # Remove previous span if it exists
    if selected_span is not None:
        selected_span.remove()
    # Add a red rectangle to highlight the selected range
    selected_span = ax.axvspan(xmin, xmax, color='red', alpha=0.3, label='Berücksichtigte Region')
    fig.canvas.draw_idle()
    ax.legend()

fig, ax = plt.subplots()
ax.plot(U, I, label='IV-Kennlinie', color='blue', marker="x", markersize=1, linestyle="-", linewidth=0.5)
ax.set_xlabel('U [V]')
ax.set_ylabel('I [A]')
ax.set_title('IV-Kennlinie des Josephson-Kontaktes')
ax.grid()
ax.legend()

span = SpanSelector(ax, onselect, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor='red'))

plt.show()