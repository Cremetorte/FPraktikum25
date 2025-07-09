from matplotlib import pyplot as plt
import numpy as np
from . import sparam_fit_module, kerr_fit_module
import os
import math

def format_exponent(value, decimals=4):
    if value == 0:
        mantissa_str = f"0.{'0' * decimals}"
        return fr"$ {mantissa_str} \times 10^{{0}} $"

    exponent = int(math.floor(math.log10(abs(value))))
    engineering_exponent = 3 * (exponent // 3)
    mantissa = value / (10 ** engineering_exponent)
    mantissa_str = f"{mantissa:.{decimals}f}"
    return fr"$ {mantissa_str} \times 10^{{{engineering_exponent}}} $"

def make_fit_plot(freq, SXX, fitparams, power, ylabel, ftype, file, material, idx="0", bg_cutoff=(0,0), freq_cutoff=(0,0), freq_exclude=(0,0)):
    """
    %todo: write docstring
    """
    if freq_exclude != (0, 0):
        f_min, f_max = freq_exclude
        mask = (freq <= f_min) | (freq >= f_max)
        freq = freq[mask].reset_index(drop=True)
        SXX = SXX[mask].reset_index(drop=True)

    if freq_cutoff != (0,0):
        f_min, f_max = freq_cutoff
        mask = (freq >= f_min) & (freq <= f_max)
        freq = freq[mask].reset_index(drop=True)
        SXX = SXX[mask].reset_index(drop=True)



    fig, axes = plt.subplots(1,2,figsize=(12,5))

    if ftype != 'trans_through':
        axes[0].scatter(freq / 1e9, 20 * np.log10(np.abs(SXX)), s=4, color="blue", zorder=5,
                   label="data")
    else:
        axes[0].scatter(freq / 1e9, np.abs(SXX), s=4, color="blue", zorder=5,
                        label="data")

    if bg_cutoff != (0,0):
        axes[0].axvline(x=bg_cutoff[0] / 1e9, color="pink", linestyle="--", label="bg cutoff")
        axes[0].axvline(x=bg_cutoff[1] / 1e9, color="pink", linestyle="--")
        idx_min = np.abs(freq - bg_cutoff[0]).argmin()
        idx_max = np.abs(freq - bg_cutoff[1]).argmin()

    pltXlim = axes[0].get_xlim()
    pltYlim = axes[0].get_ylim()

    xfit = np.linspace(pltXlim[0], pltXlim[1], 1000)
    yfit = sparam_fit_module.S11full(xfit * 1e9, fitparams, ftype=ftype)
    background_fit = sparam_fit_module.backmodel(xfit * 1e9, fitparams)

    if ftype != 'trans_through':
        axes[0].plot(xfit, 20 * np.log10(np.abs(yfit)), color="red", zorder=10, label="fit")
        axes[0].plot(xfit, 20 * np.log10(np.abs(background_fit)), color="gray", label="background")
    else:
        axes[0].plot(xfit, np.abs(yfit), color="red", zorder=10, label="fit")
        axes[0].plot(xfit, np.abs(background_fit), color="gray", label="background")

    axes[0].set_xlim(pltXlim)
    axes[0].set_ylim(pltYlim)
    if ftype != 'trans_through':
        axes[0].set_title('absolute in dB scale')
        axes[0].annotate('Power ='+str(round(power,2)) +' dBm   Qint = '+format_exponent(fitparams['Q_int'].value)+'   Qext = '+format_exponent(fitparams['Q_ext'].value)+'   f0 = '+str(round(fitparams['f_0'].value/1e9,4))+' GHz', xy=(0.05, 1.1), xycoords='axes fraction',)
    else:
        axes[0].set_title('absolute ')
        axes[0].annotate('Power ='+str(round(power,2)) +' dBm   Qload = '+str(round(fitparams['Q_load'].value/1e3,2))+'x10^3   f0 = '+str(round(fitparams['f_0'].value/1e9,4))+' GHz $d_k=', xy=(0.05, 1.1), xycoords='axes fraction',)

    axes[0].set_xlabel("Frequency [GHz]")
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(np.real(SXX), np.imag(SXX), 'o', markersize=4, color="blue", zorder=5)
    if bg_cutoff != (0,0):
        axes[1].plot(np.real(SXX[idx_min]), np.imag(SXX[idx_min]), 'x', markersize=15, color="pink",zorder=4)
        axes[1].plot(np.real(SXX[idx_max]), np.imag(SXX[idx_max]), 'x', markersize=15, color="pink",zorder=4)
    axes[1].plot(np.real(yfit), np.imag(yfit), '-', markersize=4, color="red", zorder=10)
    axes[1].plot(np.real(background_fit), np.imag(background_fit), '-', markersize=4, color="gray", zorder=1)

    axes[1].set_xlabel("Re()")
    axes[1].set_ylabel("Im()")
    axes[1].set_title('complex')
    axes[1].grid(True)
    axes[1].set_aspect('equal')

    index =  f"image_{idx:03d}"

    os.makedirs('../plots', exist_ok=True)
    os.makedirs('../plots/'+material+'_fit_plots', exist_ok=True)
    os.makedirs('../plots/'+material+'_fit_plots/'+ file[-11:-4]+'/', exist_ok=True)
    plt.savefig('../plots/'+material+'_fit_plots/'+ file[-11:-4]+'/'+ index + '.png', dpi=300)
    plt.close()

    return fig

def make_kerr_fit_plot(data_list_corrected, data_file,
                       omega_0_linear, kappa_ext_linear, attenuation,
                       kappa_0_param, kappa_nl_1_param, kappa_nl_2_param, Kerr_param,
                       scaling_factor):
    kappa_nl_2_param = kappa_nl_2_param * scaling_factor
    # Create figures for each dataset
    for i, dataset in enumerate(data_list_corrected):
        print(f'\rPlotting dataset {i+1}/{len(data_list_corrected)}', end='', flush=True)
        # Get frequency data
        freq = dataset['Frequency (Hz)']

        # Plot data
        S21_data = dataset['S21re ()'] + 1j * dataset['S21im ()']
        S21mag_data = np.abs(S21_data)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(freq, S21mag_data, label='data')

        # Calculate model
        omega = 2 * np.pi * freq
        power_dBm = dataset['Power (dBm)']

        # Get model magnitude (we only need the 4th return value)
        _, _, S21_model, S21mag_model, _, _ = kerr_fit_module.Kerr_fit(
            omega, power_dBm, omega_0_linear, kappa_ext_linear, attenuation,
            kappa_0_param, kappa_nl_1_param, kappa_nl_2_param, Kerr_param
        )
        axes[0].annotate('kappa_0/2pi = ' + format_exponent(kappa_0_param.value) + ' Hz   kappa_1/2pi = ' +
            format_exponent(kappa_nl_1_param.value) + ' Hz   kappa_2/2pi = ' + format_exponent(kappa_nl_2_param) + ' Hz    Kerr/2pi = ' + format_exponent(Kerr_param.value) + ' Hz', xy=(0.05, 1.1),
                         xycoords='axes fraction', )

        # Plot model
        axes[0].plot(freq, S21mag_model, '-', label='fit')

        # Add labels and legend
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('|S21|')
        axes[0].set_title(f'Power: {power_dBm.iloc[0]:.2f} dBm')
        axes[1].set_aspect('equal')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(np.real(S21_data), np.imag(S21_data), label='data')
        axes[1].plot(np.real(S21_model), np.imag(S21_model), '-', label='fit')
        axes[1].set_title(f'Power: {power_dBm.iloc[0]:.2f} dBm')
        axes[1].set_xlabel('Re()')
        axes[1].set_ylabel('Im()')
        axes[1].set_aspect('equal')
        axes[1].grid(True)

        index = f"image_{i:03d}"

        os.makedirs('../plots', exist_ok=True)
        os.makedirs('../plots/kerr_plots', exist_ok=True)
        os.makedirs('../plots/kerr_plots/' + data_file[-11:-4], exist_ok=True)
        plt.savefig('../plots/kerr_plots/' + data_file[-11:-4] + '/' + index + '.png', dpi=300)
        plt.close()
    print('\n')