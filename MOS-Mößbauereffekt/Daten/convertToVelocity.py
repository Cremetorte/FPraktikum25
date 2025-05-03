import numpy as np
import pandas as pd

files = ["Daten/Weicheisen.csv", "Daten/FeSO4.csv", "Daten/Stahl.csv"]

for file in files:
    data = pd.read_csv(file, delimiter=',', decimal='.')
    data["Velocity"] = 0.03151089353095451 * data["Velocity"] - 8.011193385475183
    data.to_csv(file, index=False)
    print(f"Converted {file} to velocity units.")