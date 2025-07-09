import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import os
from . import data_module

# Constants
pi = np.pi
fig_size = (4, 4)

############ code here ############
Z0 = 0
###################################

# MW functions
############ code here ############
def total_ref_C(var):
    Z_in_C = 0
    return 0

def total_ref_L(var):
    Z_in_L = 0
    return 0

def total_ref_LC(var):
    Z_in = 0
    return 0

def combined_fit_C(var):
    return np.concatenate(np.asarray([
        total_ref_C(var).real,
        total_ref_C(var).imag
    ]))

def combined_fit_L(var):
    return np.concatenate(np.asarray([
        total_ref_L(var).real,
        total_ref_L(var).imag
    ]))

def combined_fit_LC(var):
    return np.concatenate(np.asarray([
        total_ref_LC(var).real,
        total_ref_LC(var).imag
    ]))
###################################

# Load data
def load_data(filepath):
    data = data_module.readdat_pd(filepath)[0]
    ############ code here ############
    # load data as numpy arrays
    omega = 0
    re_s11 = 0
    im_s11 = 0
    ###################################
    fit_array = np.concatenate(np.asarray([re_s11, im_s11]))
    return omega, re_s11, im_s11, fit_array

# General fit and plot function
def fit_and_plot(idx, guess, bounds, fit_func, ref_func, freqs, plot_title, plot_file, param_labels, dat):
    omega, re_s11, im_s11, fit_array = dat[idx]

    print(f"\n{plot_title} Initial guesses:")
    print(f"  {guess}")

    popt, pcov = scipy.optimize.curve_fit(
        fit_func, omega, fit_array,
        p0=guess, bounds=bounds, xtol=1e-20
    )
    stds = np.sqrt(np.diag(pcov))

    # Report results
    print("Fit parameters:")
    for name, val, std in zip(param_labels, popt, stds):
        print(f"  {name}_fit: {val:.4e} ± {std:.4e}")

    # Theoretical fit
    ############ code here ############
    theo_omega = 0
    theo_ref = 0
    ###################################

    # Plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(re_s11, im_s11, 'o', markersize=2, label="Measured", zorder=1)
    ax.plot(theo_ref.real, theo_ref.imag, '-', linewidth=1, label="Fit", zorder=2)

    ax.set_xlabel('Re(S11)')
    ax.set_ylabel('Im(S11)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.5)
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))

    # Annotate
    label = '\n'.join(
        rf"${name} = {val:.2e} \pm {std:.2e}$"
        for name, val, std in zip(param_labels, popt, stds)
    )
    ax.text(0, 0, label, fontsize=12, ha='center')

    os.makedirs('../plots', exist_ok=True)
    os.makedirs('../plots/test_pcb_fit_plots/', exist_ok=True)
    fig.savefig('../plots/test_pcb_fit_plots/' + plot_file, bbox_inches='tight', dpi=1000)
    plt.close(fig)