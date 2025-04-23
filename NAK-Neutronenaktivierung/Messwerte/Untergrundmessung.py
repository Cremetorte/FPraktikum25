import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Untergrundmessung.csv', delimiter=';', decimal=',')
data = df.to_numpy().transpose()

# print(data.shape)

data[0] = (data[0] - 12) * 60 *60


mean_rate = np.mean(data[1])
std_dev = np.std(data[1])

mean_array = mean_rate*np.ones_like(data[0])




# Plot
plt.plot(data[0], data[1], color="blue", marker="x", linestyle="", label="Messpunkte der Hintergrundstrahlung")
plt.plot(data[0], mean_array, color="red", linestyle="-.", label=fr"Mittelwert der Messdaten: ({mean_rate:.3f}$\pm${std_dev:.3f})/s")
plt.fill_between(data[0], mean_array-std_dev, mean_array+std_dev, color="lightgrey", label="Standardabweichung der Messdaten")

plt.legend()
plt.xlabel(fr"Zeit [s]")
plt.ylabel(fr"Anzahl der Zerfälle [s$^{{-1}}$]")
plt.title("Messung der Untergrundstrahlung")

plt.savefig("Untergrundmessung.png", dpi=300)
