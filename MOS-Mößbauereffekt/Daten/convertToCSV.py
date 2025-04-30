import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


data = np.genfromtxt("Daten/Daten.txt").T

_1, _2, weicheisen_channel_old, weicheisen_count_old, FeSO4_channel_old, FeSO4_count_old, stahl_channel_old, stahl_count_old = data

weicheisen_channel = weicheisen_channel_old[32:487]
weicheisen_count = weicheisen_count_old[32:487]
FeSO4_channel = FeSO4_channel_old[32:487]
FeSO4_count = FeSO4_count_old[32:487]
stahl_channel = stahl_channel_old[32:487]
stahl_count = stahl_count_old[32:487]




# weicheisen_channel = []
# weicheisen_count = []
# FeSO4_channel = []
# FeSO4_count = []
# stahl_channel = []
# stahl_count = []
# # Filter out zero values
# for i in range(len(weicheisen_channel_old)):
#     if weicheisen_count_old[i] != 0:
#         weicheisen_channel.append(weicheisen_channel_old[i])
#         weicheisen_count.append(weicheisen_count_old[i])

# for i in range(len(FeSO4_channel_old)):
#     if FeSO4_count_old[i] != 0:
#         FeSO4_channel.append(FeSO4_channel_old[i])
#         FeSO4_count.append(FeSO4_count_old[i])
        
# for i in range(len(stahl_channel_old)):
#     if stahl_count_old[i] != 0:
#         stahl_channel.append(stahl_channel_old[i])
#         stahl_count.append(stahl_count_old[i])


# Save filtered data to CSV files
weicheisen_data = np.column_stack((weicheisen_channel, weicheisen_count))
FeSO4_data = np.column_stack((FeSO4_channel, FeSO4_count))
stahl_data = np.column_stack((stahl_channel, stahl_count))
np.savetxt("Daten/Weicheisen.csv", weicheisen_data, delimiter=",", header="Velocity,Count", comments="")
np.savetxt("Daten/FeSO4.csv", FeSO4_data, delimiter=",", header="Velocity,Count", comments="")
np.savetxt("Daten/Stahl.csv", stahl_data, delimiter=",", header="Velocity,Count", comments="")
