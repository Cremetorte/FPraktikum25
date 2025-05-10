import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate

filename = "Daten/sättigungabsorptionsspektrum.csv"

data = np.loadtxt(filename, delimiter=",", skiprows=1)

t, U = data[:,0], data[:,1]

def calibrate(t):
    return 2.1981e11 * t + 418.45e6

freq = calibrate(t)

output_filename = "Daten/calibrated_data.csv"
calibrated_data = np.column_stack((freq, U))
np.savetxt(output_filename, calibrated_data, delimiter=",", header="Frequency,Voltage", comments="")