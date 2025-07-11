"""
This script is for fitting S-parameters of the niobium resonators in the linear regime.
Please provide the parameters set in the user input section.
The user input section contains parameters as lists, so you can fit multiple datasets in one go.
- "file_list" contains the paths to the measurement files.
- "guess_dict_list" contains the initial parameter guesses for the fit.
- "bg_cutoff_freqs_list" contains the cutoff frequencies for the background fit.
- "ftype" contains the type of fit. Possible values are "refl_open", "refl_short", "refl_through", "trans_through" and "trans_side". "ftype" is not a list since it will be the same for all datasets.
"""

from modules.sparam_fit_module import fit
from modules import data_module, plot_module
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")
import glob


##################
### User Input ###
##################
file_list =  ['Data/3_1/Group_041_2025_07_01_14.55.45_n3_VNA_res1_lin_20_3.24GHz/Group_041_2025_07_01_14.55.45_n3_VNA_res1_lin_20_3.24GHz.dat',
 'Data/3_1/Group_041_2025_07_01_15.12.53_n3_VNA_res2_lin_20_3.61GHz/Group_041_2025_07_01_15.12.53_n3_VNA_res2_lin_20_3.61GHz.dat',
 'Data/3_1/Group_041_2025_07_01_15.24.40_n3_VNA_res3_lin_20_4.05GHz/Group_041_2025_07_01_15.24.40_n3_VNA_res3_lin_20_4.05GHz.dat',
 'Data/3_1/Group_041_2025_07_01_15.41.51_n3_VNA_res4_lin_21_5.22GHz/Group_041_2025_07_01_15.41.51_n3_VNA_res4_lin_21_5.22GHz.dat']

guess_dict_list = [
    {'f_0 (Hz)': 3.24272e9, 'kappa_0/2pi (Hz)': 0.53e6},
    {'f_0 (Hz)': 3.61e9, 'kappa_0/2pi (Hz)': 1359013.5316951978},
    {'f_0 (Hz)': 4.05e9, 'kappa_0/2pi (Hz)': 1880016.458535551},
    {'f_0 (Hz)': 5.22e9, 'kappa_0/2pi (Hz)': 10e6},
]

# bg_cutoff_freqs_list = [
#     (None, None),
# ]
bg_cutoff_freqs_list = [
    (3.23999e9, 3.24532e9),
    (3.60676e9, 3.611e9),
    (4.05e9, 4.057e9),
    (5.23e9, 5.218e9),
]


ftype = 'trans_side'

##################
##################
##################

measurement_parameter = 'Power (dBm)'
material = 'Nb'

for file, guess_dict, bg_cutoff_freqs in zip(file_list, guess_dict_list, bg_cutoff_freqs_list):
    # load data
    # .dat files generated with the stuelab package contain measurement datasets seperated by a blank line
    # the readdata module translates that into a list of dataframes with each list entry corresponding to one measurement.
    print(f'\rLoading {file}...')
    dataset_list = data_module.readdat_pd(file)

    # fit data
    params_list = []
    for idx, dataset in enumerate(dataset_list):
        print(f'\rFit dataset {idx+1}/{len(dataset_list)}...',end='',flush=True)

        S21 = dataset['S21re ()'] + 1j * dataset['S21im ()']
        f = dataset['Frequency (Hz)']

        params, _, _, finalresult = fit(f=f, sparam=S21,
                                             params_guess=guess_dict,
                                             ftype=ftype,
                                             bg_cutoff_freqs=bg_cutoff_freqs)
        params_list.append(params)
        # plot fit
        ylabel = 'S21 [dB]'
        plot_module.make_fit_plot(freq=f, SXX=S21,
                                  fitparams=params,
                                  power=dataset['Power (dBm)'][0],
                                  idx=idx,
                                  bg_cutoff=bg_cutoff_freqs,
                                  ylabel=ylabel,
                                  ftype=ftype,
                                  file=file,
                                  material=material)

    # store and plot fit summary
    fit_result = []
    for params in params_list:
        row = {name: par.value for name, par in params.items()}
        fit_result.append(row)
    fit_result_df = pd.DataFrame(fit_result)

    measurement_parameter_values = [df_i[measurement_parameter].iloc[0] for df_i in dataset_list]
    fit_result_df[measurement_parameter] = measurement_parameter_values

    os.makedirs('../results', exist_ok=True)
    os.makedirs('../results/'+material+'_fit_result', exist_ok=True)
    fit_result_df.to_csv('../results/'+material+'_fit_result/fit_result_'+file[-11:-4]+'.csv', index=False)