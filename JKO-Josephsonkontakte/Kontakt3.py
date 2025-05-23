import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


IVC = np.loadtxt('Daten/JJ3_IVC_002_avg=100.dat', skiprows=0)
U, I = np.hsplit(IVC, 2)


mask = (U>-1e-6) & (U < 3e-3)

U = U[mask]
U = U[int(len(U)/2):-int(len(U)/4):]
# U = U[::-1]
I = I[mask]
I = I[int(len(I)/2):-int(len(I)/4):]
# I = I[::-1]
# sort_idx = np.argsort(U.flatten())
# U = U[sort_idx]
# I = I[sort_idx]


# def onselect(eclick, erelease):
#     x1, x2 = eclick.xdata, erelease.xdata
#     y1, y2 = eclick.ydata, erelease.ydata
#     if None in (x1, x2, y1, y2):
#         print("Selection out of bounds.")
#         return
#     xmin, xmax = sorted([x1, x2])
#     ymin, ymax = sorted([y1, y2])
#     mask_sel = (U.flatten() >= xmin) & (U.flatten() <= xmax) & (I.flatten() >= ymin) & (I.flatten() <= ymax)
#     indices = np.where(mask_sel)[0]
#     print(f"Ausgewählte Indizes: {indices}")
#     print(f"Ausgewählte U: {U.flatten()[indices]}")
#     print(f"Ausgewählte I: {I.flatten()[indices]}")

# fig, ax = plt.subplots()
# ax.plot(U, I, label='IV-Kennlinie', color='blue', marker="x", markersize=1, linestyle="-", linewidth=0.5)
# toggle_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=1e-12, minspany=1e-12, spancoords='data', interactive=True)


rel_indices = [332, 333, 334, 335,336, 337, 338, 339]
U_crop = U[rel_indices]
I_crop = I[rel_indices]
popt, pcov = np.polyfit(U_crop.flatten(), I_crop.flatten(), 1, cov=True)
err = np.sqrt(np.diag(pcov))

# rel_indices_2 = [338, 339, 340, 341]
# U_crop_2 = U[rel_indices_2]
# I_crop_2 = I[rel_indices_2]
# popt_2, pcov_2 = np.polyfit(U_crop_2.flatten(), I_crop_2.flatten(), 1, cov=True)

# I_0 = np.abs(popt[1] + popt_2[1]) / 2
# dI_0 = np.abs(popt[1] - popt_2[1]) / 2

U_plot = np.linspace(-0.0025e-3, 0.004e-3)

print(fr"I_0 = {popt[1]} \pm {err[1]} A")

plt.plot(U*1e3, I*1e6, label='IV-Kennlinie', color='blue', marker="x", markersize=1, linestyle="-", linewidth=0.5)
plt.plot(U_plot*1e3, (popt[0] * U_plot + popt[1])*1e6, label='Linearer Fit ', color='red', linestyle='--')
# plt.plot(U_plot*1e3, (popt_2[0] * U_plot + popt_2[1])*1e6, label='Linearer Fit 2', color='green', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('U [mV]')
plt.ylabel('I [µA]')
plt.title('IV-Kennlinie des Josephson-Kontaktes 3 mit Fit')
plt.show()
