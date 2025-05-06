import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "Daten/sättigungabsorptionsspektrum.csv"

data = np.loadtxt(filename, delimiter=",", skiprows=1)
t,S = data[:,0], data[:,1]

# plt.plot(t, S, label="Absorptionsspektrum")
# plt.show()


def fit_func(x, *params):
    