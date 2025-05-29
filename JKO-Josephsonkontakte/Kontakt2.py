import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib
matplotlib.use('TkAgg')


IVC = np.loadtxt('Daten/JJ2_IVC_002_avg=100.dat', skiprows=0)
U, I = np.hsplit(IVC, 2)

mask = (I>0) & (U<1e-3)

U = U[mask]
U = U.flatten()[:int(len(U)/2):]
U = U[::-1]
I = I[mask]
I = I.flatten()[:int(len(I)/2):]
I = I[::-1]
# sort_idx = np.argsort(U.flatten())
# U = U[sort_idx]
# I = I[sort_idx]


for i in range(len(I)):
    if U[i] > 6e-5:
        print(f"U[{i}] = {U[i]} V")

        I_0 = (I[i] + I[i-1])/2
        dI_0 = (I[i] - I[i-1])/2
        print(f"I_0 = {I_0} \pm {dI_0} A")
        
        # linfit
        popt, pcov = np.polyfit(U[i:i+100], I[i:i+100], 1, cov=True)
        print(f"popt = {popt}")
        U_min = (I_0 - dI_0 - popt[1]) / popt[0]
        U_max = (I_0 + dI_0 - popt[1]) / popt[0]
        dU = (U_max - U_min) / 2
        print(fr"U_g = {U[i]} \pm {dU} V")


        

        break


plt.xlabel('U [V]')
plt.ylabel('I [A]')
plt.title('IV-Kennlinie des Josephson-Kontaktes')
plt.plot(U, I, label='IV-Kennlinie', color='blue', marker="x", markersize=1, linestyle="-", linewidth=0.5)
plt.grid()
plt.legend()
plt.show()