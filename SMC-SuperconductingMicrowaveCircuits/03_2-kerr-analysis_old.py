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

freqs = [3.24, 3.61, 4.05, 5.22]
ω_0 = [2 * np.pi * f * 1e9 for f in [3.243, 3.609, 4.053, 5.224]]  # Convert GHz to rad/s

kerr_file_list =[
    "/home/dennis/fprakt2025/FPraktikum25/SMC-SuperconductingMicrowaveCircuits/Data/3_2/Group_041_2025_07_01_14.39.49_n3_VNA_general_1.00-1.00GHz/Group_041_2025_07_01_14.39.49_n3_VNA_general_1.00-1.00GHz.dat",
    "/home/dennis/fprakt2025/FPraktikum25/SMC-SuperconductingMicrowaveCircuits/Data/3_2/Group_041_2025_07_01_15.14.59_n3_VNA_res2_sweep_31_3.61GHz/Group_041_2025_07_01_15.14.59_n3_VNA_res2_sweep_31_3.61GHz.dat",
    "/home/dennis/fprakt2025/FPraktikum25/SMC-SuperconductingMicrowaveCircuits/Data/3_2/Group_041_2025_07_01_15.26.31_n3_VNA_res3_sweep_24_4.05GHz/Group_041_2025_07_01_15.26.31_n3_VNA_res3_sweep_24_4.05GHz.dat",
    "/home/dennis/fprakt2025/FPraktikum25/SMC-SuperconductingMicrowaveCircuits/Data/3_2/Group_041_2025_07_01_15.43.50_n3_VNA_res4_sweep_24_5.22GHz/Group_041_2025_07_01_15.43.50_n3_VNA_res4_sweep_24_5.22GHz.dat"

            ]


bg_fit_file_list = [f"results/Nb_fit_result/fit_result_{f}GHz.csv" for f in freqs]

# guess_dict_list = [
#     {'kappa_0': {'value': 0.3e9, 'min': 0, 'max': np.inf, 'vary': None},
#      'kappa_1': {'value': 0.6e9, 'min': 0, 'max': np.inf, 'vary': None},
#      'kappa_2': {'value': 1.5e9, 'min': 0, 'max': np.inf, 'vary': None},
#      'Kerr': {'value': -170, 'min': -np.inf, 'max': 0, 'vary': None}},
# ]

# guess_dict_list = [
#     {'kappa_0': {'value': 9.8122, 'min': 0, 'max': 50, 'vary': True},
#      'kappa_1': {'value': 990.7507e-9, 'min': 0, 'max': 1e-4, 'vary': True},
#      'kappa_2': {'value': 2.0915e-21, 'min': 0, 'max': 1e-15, 'vary': True},
#      'Kerr': {'value': -90.9757e-9, 'min': -1e-3, 'max': 0, 'vary': True}},] + [
#     {'kappa_0': {'value': 0.005*w, 'min': 0, 'max': 5*w, 'vary': None},
#      'kappa_1': {'value': 0.02*w, 'min': 0, 'max': 2*w, 'vary': None},
#      'kappa_2': {'value': 0.02*w, 'min': 0, 'max': 5*w, 'vary': None},
#      'Kerr': {'value': -0.01*0.01*w, 'min': -w, 'max': 0, 'vary': None}}
#      for w in freqs[1:]]

guess_dict_list = [
    {
        'kappa_0': {'value': 5e6 , 'min': 1e6 , 'max': 100e6 , 'vary': True},  # ~61.6 rad/s (9.81 Hz)
        'kappa_1': {'value': 1 , 'min': 1e-9 , 'max': 1e2 , 'vary': True},  # ~6.23e-6 rad/s
        'kappa_2': {'value': 1e-10 , 'min': 0, 'max': 1e-1 , 'vary': False},  # Vernachlässigbar
        'Kerr': {'value': -0.05e6 , 'min': -1e6 , 'max': -1e-9 , 'vary': True}  # ~-5.72e-7 rad/s
    }
    for w in [3.243e9, 3.609e9, 4.053e9, 5.224e9]  # ω₀-Werte in Hz
]

P_lims_list=[
    (-22, -19),
           ]*4

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
    else:
        dataset_list_filtered = dataset_list

    ### remove background
    # load fit and average the 20 fit results
    bg_fit_results = pd.read_csv(bg_fit_file)
    bg_fit_result = bg_fit_results.mean().to_frame().T

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
    # To circumvent that one can separate the small value into a bigger value and a scaling factor.
    if guess_dict['kappa_2']['max'] < 1e-10:
        scaling_factor = 1e-10
        guess_dict['kappa_2']['max'] = guess_dict['kappa_2']['max'] / scaling_factor
    elif guess_dict['kappa_2']['max'] < 1e-20:
        scaling_factor = 1e-20
        guess_dict['kappa_2']['max'] = guess_dict['kappa_2']['max'] / scaling_factor
    else:
        scaling_factor = 1


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