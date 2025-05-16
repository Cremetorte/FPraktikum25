import matplotlib.pyplot as plt
import numpy as np
import os

strom_dict = {"0A": 0, 
              "1A": 1,
              "3A": 3,
              "4A": 4,
              "5A": 5,
              "6A": 6,
                "7A": 7,
                "8A": 8,
                "11A": 11,
                "14A": 14,
                "45A": 4.5,
                "47A": 4.7,
                "53A": 5.3,
                "55A": 5.5,
                "65A": 6.5,
              }

# for i in strom_dict.keys():
#     os.rename(f"Daten/A2/Spulenstrom{i}.dat", f"Daten/A2/Spulenstrom{int(strom_dict[i]*1000):05d}mA.dat")

I_in_mA = [I*1000 for I in strom_dict.values()]
print(I_in_mA)