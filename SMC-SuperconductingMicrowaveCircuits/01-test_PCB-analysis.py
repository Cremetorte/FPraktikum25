'''#This script is for evaluating and fitting the small PCB with an inductance, capacitance and LC-circuit.
To get this running, look for the ares marked like this:
    ##### code here ##### 
    #xyz
#   #####################
# (they also will be in the testPCB_module file, which is loaded in the beginning)

Overall your task will be to edit these areas, e.g. by adding data paths, changing values of variables or editing formulas. 
After editing those areas, the code will plot the data and generate the needed fits (of course you can edit the plots to your liking)
'''



from src.modules.testPCB_module import *

############ code here ############
file_list =[
    '../data/your_C_data/your_C_data.dat',
    '../data/your_L_data/your_C_data.dat',
    '../data/your_LC_data/your_C_data.dat',
]
# Select what to fit
FIT_C = False
FIT_L = False
FIT_LC = False
###################################

# Load and format data
DATA = [load_data(file) for file in file_list]

# Run fits selectively
if FIT_C:
    fit_and_plot(
        ############ code here ############
        guess=[0],
        bounds=([0], [0]),
        fit_func=0,
        ref_func=0,
        freqs=(0, 0),
        ###################################
        idx=0,
        plot_title="Fitting C",
        plot_file="001_C.png",
        param_labels=["C"],
        dat=DATA
    )

if FIT_L:
    fit_and_plot(
        ############ code here ############
        guess=[0, 0],
        bounds=([0, 0], [0, 0]),
        fit_func=0,
        ref_func=0,
        freqs=(0, 0),
        ###################################
        idx=1,
        plot_title="Fitting L",
        plot_file="002_L.png",
        param_labels=["L", "R"],
        dat=DATA
    )

if FIT_LC:
    fit_and_plot(
        ############ code here ############
        guess=[0, 0, 0],
        bounds=([0, 0, 0], [0, 0, 0]),
        fit_func=0,
        ref_func=0,
        freqs=(0, 0),
        ###################################
        idx=2,
        plot_title="Fitting LC",
        plot_file="003_LC.png",
        param_labels=["L", "C", "R"],
        dat=DATA
    )
