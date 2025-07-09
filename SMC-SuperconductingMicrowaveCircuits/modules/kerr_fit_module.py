import scipy
import numpy as np

COLNAME_FREQ = "Frequency (Hz)"
COLNAME_SPARAM_DB = "dB (dB)"
COLNAME_SPARAM_RE = "re ()"
COLNAME_SPARAM_IM = "im ()"
COLNAME_SPARAM_PHASE = "Ph (rad)"
sparam_type ='S21'

def Kerr_residual(params, datasets, omega_0, kappa_ext, attenuation, scaling_factor):
    kappa_0 = params['kappa_0']
    kappa_nl_1 = params['kappa_1']
    kappa_nl_2 = params['kappa_2'] * scaling_factor

    Kerr = params['Kerr']

    # Pre-allocate a list to store residuals
    residual_list = []

    # Process each dataset
    for dataset in datasets:
        # Extract data once
        omega = 2 * np.pi * dataset['Frequency (Hz)']
        S21mag_data = np.abs(dataset['S21re ()'] + 1j * dataset['S21im ()'])
        Power_dBm = dataset['Power (dBm)']

        # Get model magnitude (we only need the 4th return value)
        _, _, _, S21mag_model, _, _ = Kerr_fit(omega, Power_dBm, omega_0, kappa_ext, attenuation,
                                              kappa_0, kappa_nl_1, kappa_nl_2, Kerr)

        # Calculate residual and append to list
        residual_list.append(S21mag_model - S21mag_data)

    # Concatenate all residuals into a single array
    return np.concatenate(residual_list)

def Kerr_fit(omega, Power_dBm, omega_0, kappa_ext, attenuation, kappa_0, kappa_nl_1, kappa_nl_2, Kerr):
    # On-chip photon number
    P_onChip = 10 ** ((Power_dBm + attenuation) / 10) / 1000  # in Watts
    n = P_onChip / (scipy.constants.hbar * omega)

    # Intracavity photon number, step 1: Solve third order polynomial
    # Get polynomial coefficients directly as numpy arrays
    Kerr_val = Kerr.value
    kappa_nl_1_val = kappa_nl_1.value
    kappa_nl_2_val = kappa_nl_2

    kappa_0_val = kappa_0.value

    a = 0.25 * kappa_nl_2_val ** 2  # n_c^5
    b = 0.5 * kappa_nl_1_val * kappa_nl_2_val  # n_c^4
    c = Kerr_val ** 2 + 0.5 * kappa_0_val * kappa_nl_2_val + 0.25 * kappa_nl_1_val ** 2  # n_c^3
    d = -2 * Kerr_val * (omega - omega_0) + 0.5 * kappa_0_val * kappa_nl_1_val  # n_c^2
    e = (omega - omega_0) ** 2 + 0.25 * kappa_0_val ** 2  # n_c^1
    f = - kappa_ext / 2 * n

    # Stack coefficients for vectorized root finding
    coeffs = np.column_stack([np.full_like(d, a), np.full_like(d, b), np.full_like(d, c), d, e, f])

    # Compute roots for each polynomial
    roots = np.array([np.roots(poly) for poly in coeffs])

    # Process roots - vectorized operations
    # Get real parts of roots where imaginary part is zero, otherwise -1
    roots_real = np.where(np.isclose(np.imag(roots), 0), np.real(roots), -1.)

    # Find smallest positive root for each polynomial (low-amplitude branch)
    # Replace non-positive values with NaN for min calculation
    roots_pos = np.where(roots_real > 0, roots_real, np.nan)
    n_c = np.nanmin(roots_pos, axis=1)

    # Calculate derived quantities
    alpha_0 = np.sqrt(n_c)
    phi = np.arctan2( -(kappa_0_val + kappa_nl_1_val * n_c + kappa_nl_2_val * n_c**2) / 2, (omega - omega_0 - Kerr_val * n_c))


    # Calculate response
    S21 =  1 + 1j * np.sqrt(kappa_ext/2) * alpha_0 / np.sqrt(n) * np.exp(-1j * phi)
    S21_mag = np.abs(S21)
    S21_db = 10 * np.log10(S21_mag ** 2)
    phase = np.angle(S21)

    return n_c, n, S21, S21_mag, S21_db, phase

def background_correct_data(dataset_list_filtered, bg_fit_result):
    data_list_corrected = []
    # Pre-compute constants for background correction
    x_0 = bg_fit_result['x_0'][0]
    x_1 = bg_fit_result['x_1'][0]
    x_2 = bg_fit_result['x_2'][0]
    phi_0 = bg_fit_result['phi_0'][0]
    phi_1 = bg_fit_result['phi_1'][0]
    theta = bg_fit_result['theta'][0]
    exp_neg_i_theta = np.exp(-1j * theta)

    for idx, data in enumerate(dataset_list_filtered):
        print(f'\rStep {idx + 1}/{len(dataset_list_filtered)}', end='', flush=True)

        # Get frequency and complex S-parameter
        omega = 2 * np.pi * data[COLNAME_FREQ]
        sparam_complex = data[sparam_type + COLNAME_SPARAM_RE] + 1j * data[sparam_type + COLNAME_SPARAM_IM]

        # Vectorized background calculation
        A = x_0 + x_1 * omega + x_2 * omega ** 2
        B = phi_0 + phi_1 * omega
        exp_i_B = np.exp(1j * B)

        # Vectorized correction calculation
        sparam_corrected = (sparam_complex / (A * exp_i_B) - 1) * exp_neg_i_theta + 1

        # Update data with corrected values
        data[sparam_type + COLNAME_SPARAM_RE] = np.real(sparam_corrected)
        data[sparam_type + COLNAME_SPARAM_IM] = np.imag(sparam_corrected)
        data[sparam_type + COLNAME_SPARAM_PHASE] = np.angle(sparam_corrected)
        data[sparam_type + COLNAME_SPARAM_DB] = 20 * np.log10(np.abs(sparam_corrected))

        data_list_corrected.append(data)
    print('\n')
    return data_list_corrected