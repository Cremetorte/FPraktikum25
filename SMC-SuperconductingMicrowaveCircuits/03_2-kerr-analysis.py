"""
This script is for fitting S-parameters of resonators in the non-linear regime..
Please provide the parameters set in the user input section.
The user input section contains parameters as lists, so you can fit multiple datasets in one go.
- "kerr_file_list" contains the paths to the measurement files.
- "bg_fit_file_list" contains the paths to the corresponding linear regime fit results.
- "P_lims_list" contains the power range for each dataset. If you want to fit all datasets, set the entries to (0,0).
"""

from modules import kerr_fit_module, data_module, plot_module
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize
import time

##################
### User Input ###
##################
kerr_file_list =[
    '../data/your_data/your_data.dat',
            ]

bg_fit_file_list = [
    '../results/Nb_fit_result/your_fit.csv',
                ]

guess_dict_list = [
    {'kappa_0': {'value': None, 'min': None, 'max': None, 'vary': None},
     'kappa_1': {'value': None, 'min': None, 'max': None, 'vary': None},
     'kappa_2': {'value': None, 'min': None, 'max': None, 'vary': None},
     'Kerr': {'value': None, 'min': None, 'max': None, 'vary': None}},
]

P_lims_list=[
    (None, None),
           ]

##################
##################
##################

for kerr_file, bg_fit_file, guess_dict, P_lims in zip(kerr_file_list, bg_fit_file_list, guess_dict_list, P_lims_list):
    ### load datasets
    print(f'\rLoading {kerr_file}...')
    dataset_list = data_module.readdat_pd(kerr_file)

    # get attenuation from leftmost absolute S21 value since its far from resonance an is basically the background level
    bg_dataset = dataset_list[0]
    attenuation = np.abs(bg_dataset['S21dB (dB)'][0])
    print(f'Attenuation: {attenuation} dB')

    # filter data accourding to P_min, P_max
    if P_lims != (0,0):
        dataset_list_filtered = [df for df in dataset_list
                         if P_lims[0] <= df['Power (dBm)'].iloc[0] <= P_lims[1]]

    ### remove background
    # load fit and average the 20 fit results
    bg_fit_results = pd.read_csv(bg_fit_file)
    bg_fit_result = bg_fit_results.mean()

    # pick the omega_0 / kappa from the averaged data in the linear regime
    omega_0_linear = 2 * np.pi * bg_fit_result['f_0']
    kappa_ext_linear = omega_0_linear / bg_fit_result['Q_ext']
    kappa_0_linear = omega_0_linear / bg_fit_result['Q_int'] + omega_0_linear / bg_fit_result['Q_ext']

    print('\nCorrecting data with fit')
    data_list_corrected = kerr_fit_module.background_correct_data(dataset_list_filtered, bg_fit_result)

    print('\nNow performing kerr fit')
    duffing_fit_start = time.time()

    ### fit background corrected data with kerr / duffing expression
    params = Parameters()
    params.add('kappa_0',
               value=guess_dict['kappa_0']['value'],
               min=guess_dict['kappa_0']['min'],
               max=guess_dict['kappa_0']['max'],
               vary=guess_dict['kappa_0']['vary'])
    params.add('kappa_1',
               value=guess_dict['kappa_1']['value'],
               min=guess_dict['kappa_1']['min'],
               max=guess_dict['kappa_1']['max'],
               vary=guess_dict['kappa_1']['vary'])

    # lmfit has problems with small numbers, if you have min=0 and max<=1e-10 it thinks the two are equal.
    # To circumvent that one can seperate the small value into a bigger value and a scaling factor.
    if guess_dict['kappa_2']['max'] <= 1e10:
        scaling_factor = 1e-10
        guess_dict['kappa_2']['max'] = guess_dict['kappa_2']['max'] / 1e10
    elif guess_dict['kappa_2']['max'] <= 1e20:
        scaling_factor = 1e-20
        guess_dict['kappa_2']['max'] = guess_dict['kappa_2']['max'] / 1e20

    params.add('kappa_2',
               value=guess_dict['kappa_2']['value'],
               min=guess_dict['kappa_2']['min'],
               max=guess_dict['kappa_2']['max'],
               vary=guess_dict['kappa_2']['vary'])

    params.add('Kerr',
               value=guess_dict['Kerr']['value'],
               min=guess_dict['Kerr']['min'],
               max=guess_dict['Kerr']['max'],
               vary=guess_dict['Kerr']['vary'])
    result = minimize(kerr_fit_module.Kerr_residual, params, args=(data_list_corrected, omega_0_linear, kappa_ext_linear, attenuation, scaling_factor))
    print('\n')
    params = result.params

    # save fit parameters to csv
    data_module.save_params_to_csv(params, kerr_file)

    duffing_fit_end = time.time()
    print(f'Fitting took {(duffing_fit_end - duffing_fit_start)/60:.2f} minutes.')

    # Create plots for each dataset
    print('Generating plots...')

    kappa_0_param = result.params['kappa_0']
    kappa_1_param = result.params['kappa_1']
    kappa_2_param = result.params['kappa_2']
    Kerr_param = result.params['Kerr']

    plot_module.make_kerr_fit_plot(data_list_corrected, kerr_file,
                                   omega_0_linear, kappa_ext_linear, attenuation,
                                   kappa_0_param, kappa_1_param, kappa_2_param, Kerr_param,
                                   scaling_factor)