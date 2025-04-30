import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import geneticalgorithm.geneticalgorithm as ga

Weicheisen = pd.read_csv("Daten/Weicheisen.csv", delimiter=',', decimal='.')
weicheisen_velocity = Weicheisen["Velocity"].to_numpy()
weicheisen_count = Weicheisen["Count"].to_numpy()


smooth_weicheisen_velocity = np.linspace(np.min(weicheisen_velocity), np.max(weicheisen_velocity), 1000)


def lorentz_6_peaks(x, params):
    



plt.plot(weicheisen_velocity, weicheisen_count)
plt.plot(smooth_weicheisen_velocity, lorentz_6_peaks(smooth_weicheisen_velocity, results))

plt.savefig("Plots/Genetic_6_peaks")