"""
This script is for fitting S-parameters of the copper resonators in the linear regime.
Please provide the parameters set in the user input section.
The user input section contains parameters as lists, so you can fit multiple datasets in one go.
- "file_list" contains the paths to the measurement files.
- "sparam_type_list" contains the type of S-parameter to fit. Possible values are "S21", "S11" and "S22".
- "guess_dict_list" contains the initial parameter guesses for the fit.
- "bg_cutoff_freqs_list" contains the cutoff frequencies for the background fit.
- "ftype" contains the type of fit. Possible values are "refl_open", "refl_short", "refl_through", "trans_through" and "trans_side".
"""

from modules.sparam_fit_module import fit
from modules import data_module, plot_module
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")

file_list =[
    '../data/your_data/your_data.dat',
]

sparam_type_list = [
    '',
]

guess_dict_list = [
    {'f_0 (Hz)': None, 'kappa_0/2pi (Hz)': None},
]

bg_cutoff_freqs_list = [
    (None, None),
]

ftype_list = [
    '',
]
##################
##################
##################

measurement_parameter = 'Power (dBm)'
material = 'Cu'

for file, guess_dict, ftype, sparam_type, bg_cutoff_freqs, in zip(file_list, guess_dict_list, ftype_list, sparam_type_list,
                                                                  bg_cutoff_freqs_list):
    # load data
    # .dat files generated with the stuelab package contain measurement datasets seperated by a blank line
    # the readdata module translates that into a list of dataframes with each list entry corresponding to one measurement.
    print(f'\rLoading {file}...')
    dataset_list = data_module.readdat_pd(file)

    # fit data
    params_list = []
    for idx, dataset in enumerate(dataset_list):
        print(f'\rFit dataset {idx + 1}/{len(dataset_list)}...', end='', flush=True)

        SXX = dataset[sparam_type+'re ()'] + 1j * dataset[sparam_type+'im ()']
        f = dataset['Frequency (Hz)']

        params,_,_,_= fit(f=f, sparam=SXX, ftype=ftype, params_guess=guess_dict,
                          bg_cutoff_freqs=bg_cutoff_freqs)
        params_list.append(params)
        # plot fit
        if ftype != 'trans_through':
            ylabel = sparam_type+' [dB]'
        else:
            ylabel = '|' +sparam_type + '|'

        plot_module.make_fit_plot(freq=f, SXX=SXX,
                                  fitparams=params,
                                  power=dataset['Power (dBm)'][0],
                                  idx=idx,
                                  ylabel=ylabel,
                                  ftype=ftype,
                                  file=file,
                                  material=material,
                                  bg_cutoff=bg_cutoff_freqs,)
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
        fit_result_df.to_csv('../results/'+material+'_fit_result/fit_result_' + file[-11:-4] + '.csv', index=False)