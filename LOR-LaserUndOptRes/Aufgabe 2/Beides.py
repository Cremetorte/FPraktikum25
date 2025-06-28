import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tabulate import tabulate
matplotlib.use("qtagg")



# l = 0
peaks_l0 = {"(0,0)": 0.0, "(1,0)": 9388342500.0, "(2,0)": 7142452500.0, "(3,0)": 4765987500.0, "(4,0)": 2298120000.0, "(0,0)_prime": 11568945000.0}

nu_l0 = sorted(peaks_l0.values())

fsr_l0 = nu_l0[-1] - nu_l0[0] 

nu_t = fsr_l0 * 1/(2*3.1415) * np.acos(1 - 2*15.64/18)



# l = 1
peaks_l1 = {"(0,1)": 0.0, "(1,1)": 9218595000.0, "(2,1)": 7403602500.0, "(3,1)": 5627782500.0, "(4,1)": 3747502500.0, "(0,1)_prime": 10903012500.0}

nu_l1 = sorted(peaks_l1.values())


peaks_labels = ["(0,0)", "(1,0)", "(2,0)", "(3,0)", "(4,0)", "(0,1)", "(1,1)", "(2,1)", "(3,1)", "(4,1)"]


# experimental stuff
nu_exp = {}
for i in peaks_labels:
    if i in peaks_l0.keys():
        nu_exp[i] = (peaks_l0[i] / fsr_l0)
    elif i in peaks_l1.keys():
        nu_exp[i] = (peaks_l1[i] / fsr_l0)
    else:
        print(f"Warning: {i} not found in peaks_l0 or peaks_l1")


def theoretical_nu(dp, dl, q):
    l = dl
    p = dp
    # print(f"Calculating theoretical nu for ({l},{p} with q={q}")
    return (q*fsr_l0 + nu_t*(2 * p + l))/fsr_l0
    

#theoretical stuff
nu_theo = {"(0,0)": 0,}
q_dict = {"(0,0)": 0,}
last_pl = [0,0]
for i in peaks_labels:
    if i == "(0,0)":
        continue
    if i == "(0,1)":
        nu_theo["(0,1)"] = 0
        q_dict["(0,1)"] = 0
        last_pl = [0, 1]
        continue

    p = int(i[1])
    l = int(i[3])
    dp, dl = p - last_pl[0], l - last_pl[1]
    
    last_pl_string = f"({last_pl[0]},{last_pl[1]})"

    best_q = None
    best_diff = float('inf')
    best_nu = None

    for q in range(-5, 6):
        theo = (theoretical_nu(dp, dl, q) + nu_theo[last_pl_string]) * 1.
        exp = nu_exp[i]
        diff = abs(theo - exp)
        if diff < best_diff:
            best_diff = diff
            best_q = q
            best_nu = theo

    nu_theo[i] = best_nu
    q_dict[i] = best_q
    last_pl = [p, l]





    
print(nu_theo)


table_data = []
for label in peaks_labels:
    exp_val = nu_exp.get(label, "")
    theo_val = nu_theo.get(label, "")
    q_val = q_dict.get(label, "")
    table_data.append([label, exp_val, theo_val, q_val])

headers = ["Peak Label", "nu_exp", "nu_theo", "q"]
print(tabulate(table_data, headers=headers, tablefmt="latex_raw", floatfmt=".4f"))

