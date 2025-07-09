import numpy as np
from scipy.signal import savgol_filter as smooth
from lmfit import minimize, Parameters

pi = np.pi

def realimag(array):
    """Makes alternating real and imaginary part array from complex array

    Takes an array-like object of complex number elements and generates a
    1-D array aleternating the real and imaginary part of each element of the
    original array.  If array = (z1,z2,...,zn), then this function returns
    (x1,y1,x2,y2,...,xn,yn) where xi = np.real(zi) and yi=np.imag(zi).

    Parameters
    ----------
    array : array_like of complex
        1-D array of complex numbers

    Returns
    -------
    numpy.ndarray
        New array of alternating real and imag parts of each element of
        original array

    """
    return np.array([(x.real, x.imag) for x in array]).flatten()

def un_realimag(array):
    """Makes complex array from alternating real and imaginary part array

    Performs the reverse operation to realimag

    Parameters
    ----------
    array : array_like of float
        1-D array of real numbers.  Should have an even number of elements.

    Returns
    -------
    numpy.ndarray of numpy.complex128
        1-D array of complex numbers built by taking every two elements of
        original array as the real and imaginary parts

    """
    z = []
    for x,y in zip(array[::2],array[1::2]):
        z.append(x+1j*y)
    return np.array(z)

def backmodel(f, params):
    """Function for background model.

    Returns the background model values for a given set of parameters and
    frequency values.  Uses parameter object from lmfit.

    Parameters
    ----------
    f : float or array_like of float
        Frequency values to evaluate the model at
    params : lmfit.Parameters
        Parameters set with which to generate the background

    Returns
    -------
    numpy.complex128 or numpy.ndarray of numpy.complex128
        Background values at frequencies x with model parameters params

    """
    omega = 2 * pi * f

    x_0 = params['x_0'].value
    x_1 = params['x_1'].value
    x_2 = params['x_2'].value

    phi_0 = params['phi_0'].value
    phi_1 = params['phi_1'].value

    return (x_0 + x_1 * omega + x_2 * np.power(omega, 2.)) * np.exp(1j * (phi_0 + phi_1 * omega))

def S11theo(f, params, ftype='refl_short'): # Equation
    """Theoretical response functions of cavities with no background

    Returns the theory values of cavity response functions for a given set of
    parameters for different cavity models.

    Parameters
    ----------
    f : float or array_like of float
        Frequency values to calculate the response function at
    params : lmfit.Parameters
        Parameter values for calculation
    ftype : {'refl_short','refl_open','trans_side','trans_side_minus'}, optional
        Model for response function selection.  These are described in the
        resonator response function document.  The desired model should be found there and
        selected with this parameter.

        The possible models are (CHECK RESONATOR RESPONSE FUNCTION DOCUMENT):

        - 'refl_short': Reflection cavity with short circuit boundary condition at input port (default selection)
        - 'refl_open': Reflection cavity with open circuit boundary condition at input port (= model A with a minus sign)
        - 'trans_side': Transmission through a side coupled geometry
        - 'trans_side_minus': Same as model trans_side but with a minus sign

    Returns
    -------
    numpy.complex128 or numpy.ndarray of numpy.complex128
        Values of theoretical response function at frequencies given in f

    """
    theta = params['theta'].value
    f_0 = params['f_0'].value
    omega = 2 * pi * f
    omega_0 = 2 * pi * f_0
    Delta_omega = omega - omega_0

    if ftype != "trans_through":
        Q_int = params['Q_int'].value
        Q_ext = params['Q_ext'].value

        kappa_int = omega_0 / Q_int
        kappa_ext = omega_0 / Q_ext

    else:
        Q_load = params['Q_load'].value
        d_k = params['d_k'].value

        kappa_0 = omega_0 / Q_load


    if ftype == "refl_open":
        return  1. - (2. * kappa_ext * np.exp(1j * theta)) / (kappa_ext + kappa_int + 2j * Delta_omega)
    elif ftype== 'refl_short':
        return  1. - (2. * kappa_ext * np.exp(1j * theta)) / (kappa_ext + kappa_int + 2j * Delta_omega)
    elif ftype == 'trans_side':
        return 1. - (kappa_ext * np.exp(1j * theta)) / (kappa_ext + kappa_int + 2j * Delta_omega)
    elif ftype== 'refl_through':
        kappa_in = kappa_ext
        return  1. - (2 * kappa_in * np.exp(1j * theta)) / (kappa_ext + kappa_int + 2j * Delta_omega)
    elif ftype== 'trans_through':
        return  1 + (d_k * np.exp(1j * theta)) / (kappa_0 + 2j * Delta_omega)

    return None

def S11full(f, params, ftype='refl_short'):
    """
    Function for total response from background model and resonator response
    """
    return S11theo(f, params, ftype) * backmodel(f, params)

def S11_residual(params, f, data, ftype='refl_short'):
    """Residual for full fit including background

    Calculates the residual for the full signal and background with respect to
    the given background and theory model.

    Parameters
    ----------
    params : lmfit.Parameters
        Parameter values for calculation
    f : float or array_like of float
        Frequency values to calculate the combined signal and background
        response function at using params
    data : complex or array_like of complex
        Original signal data values at frequencies f
    ftype : {'refl_short','refl_open','trans_side','trans_side_minus'}, optional
        Model for theory response function selection.  See S11theo for
        explanation

    Returns
    -------
        Residual values (model value - data value) at values given in x.
        Alternates real and imaginary values

    """

    model = S11full(f, params, ftype)
    residual = model - data
    return realimag(residual)


def background_residual(params, x, data):
    """Background residual

    Computes the residual vector for the background fitting.  Operates on
    complex vectors but returns a real vector with alternating real and imag
    parts

    Parameters
    ----------
    params : lmfit.Parameters
        Model parameters for background generation
    x : float or array_like of float
        Frequency values for background calculation.  backmodel function is
        used.
    data : complex or array_like of complex
        Background values of signal to be compared to generated background at
        frequency values x.

    Returns
    -------
    numpy.ndarray of numpy.float
        Residual values (model value - data value) at values given in x.
        Alternates real and imaginary values

    """
    model = backmodel(x, params)
    residual = model - data
    return realimag(residual)

def fit(f, sparam, params_guess,ftype='refl_short', cutoff_freqs=(0,0), fit_background=True, bg_cutoff_freqs=(0,0), margin=251, exclude_freqs = (0,0)):
    """Function for fitting the S11 response function"""

    # Convert frequency and S-parameter into arrays
    f = np.array(f)
    sparam = np.array(sparam)

    # Exlude frequency range from data according to lower and upper limit specified in exclude_freqs
    if exclude_freqs != (0, 0):
        f_min, f_max = exclude_freqs
        mask = (f <= f_min) | (f >= f_max)
        f = f[mask]
        sparam = sparam[mask]

    # Truncate data according to lower and upper limit specified in cutoff_freqs
    if cutoff_freqs != (0,0):
        f_min, f_max = cutoff_freqs
        mask = (f >= f_min) & (f <= f_max)
        f = f[mask]
        sparam = sparam[mask]



    # Smooth data for initial guesses
    if margin == None or margin == 0:  # If no smooting desired, pass None as margin
        margin = 0
        Re_sparam_smoothed = sparam.real
        Im_sparam_smoothed = sparam.imag
        sparam_smoothed = np.array([x + 1j * y for x, y in zip(Re_sparam_smoothed, Im_sparam_smoothed)])
    elif type(margin) != int or margin % 2 == 0 or margin <= 3:
        raise ValueError('margin has to be either None, 0, or an odd integer larger than 3')
    else:
        Re_sparam_smoothed = np.array(smooth(sparam.real, margin, 3))
        Im_sparam_smoothed = np.array(smooth(sparam.imag, margin, 3))
        sparam_smoothed = np.array([x + 1j * y for x, y in zip(Re_sparam_smoothed, Im_sparam_smoothed)])

    # seperate signal from background according to lower and upper limit specified in bg_cutoff_freqs
    f_min_bg, f_max_bg = bg_cutoff_freqs
    background_mask = (f < f_min_bg) | (f > f_max_bg)

    f_background = f[background_mask]
    sparam_background = sparam[background_mask]

    # fit background
    omega = 2 * pi * f
    omega_background = 2 * pi * f_background
    if fit_background:
        #Make initial background guesses
        x_1 = (np.abs(sparam_smoothed)[-1] - np.abs(sparam_smoothed)[0])/(omega[-1]-omega[0])
        x_0 = np.average(np.abs(sparam_smoothed)) - x_1 * omega_background[0]
        x_2 = 0.
        xx = []
        for i in range(0,len(omega_background)-1):
            dw = omega_background[i+1]-omega_background[i]
            dtheta = np.angle(sparam_background[i+1])-np.angle(sparam_background[i])
            if (np.abs(dtheta)>pi):
                continue
            xx.append(dtheta/dw)
        #Remove infinite values in xx (from repeated frequency points for example)
        xx = np.array(xx)
        idx = np.isfinite(xx)
        xx = xx[idx]
        phi_1 = np.average(xx)
        phi_0 = np.unwrap(np.angle(sparam_background))[0] - phi_1*omega_background[0]

    else:
        x_0 = 0
        x_1 = 0
        x_2 = 0
        phi_0 = 0
        phi_1 = 0

    params = Parameters()
    myvary = True
    params.add('x_0', value=x_0, vary=myvary)
    params.add('x_1', value=x_1, vary=myvary)
    params.add('x_2', value=x_2, vary=myvary)
    params.add('phi_0', value=phi_0, vary=myvary)
    params.add('phi_1', value=phi_1, vary=myvary)

    bg_fit_result = minimize(background_residual, params, args=(f_background, sparam_background))
    fullbackground = backmodel(f, bg_fit_result.params)

    # remove background from data
    sparam_corr = sparam / fullbackground

    # fit resonance with starting parameters provided in params_guess, do not vary background parameters
    kappa_0_guess = 2 * pi * params_guess['kappa_0/2pi (Hz)']
    f_0_guess = params_guess['f_0 (Hz)']
    f_0_guess_idx = np.argmin(np.abs(f-f_0_guess))

    params = bg_fit_result.params

    if ftype !=  "trans_through":
        Tres = np.abs(sparam_corr[f_0_guess_idx])
        kappa_ext = (1-Tres)*kappa_0_guess

        kappa_int = kappa_0_guess - kappa_ext
        if kappa_int <= 0.:
            kappa_int = kappa_ext

        Q_int = 2 * pi * f_0_guess / kappa_int
        Q_ext =  2 * pi * f_0_guess / kappa_ext

        params.add('Q_int', value=Q_int, vary=True, min=0)
        params.add('Q_ext', value=Q_ext, vary=True, min=0)

    else:
        Q_load = 2 * pi * f_0_guess / kappa_0_guess

        params.add('d_k', value=100e6, vary=True, min=0)
        params.add('Q_load', value=Q_load, vary=True, min=0)


    params.add('f_0', value=f_0_guess, vary=True)
    params.add('theta', value=0, vary=True)
    params['x_0'].set(vary=False)
    params['x_1'].set(vary=False)
    params['x_2'].set(vary=False)
    params['phi_0'].set(vary=False)
    params['phi_1'].set(vary=False)

    prefinal_result = minimize(S11_residual, params, args=(f, sparam, ftype))
    params = prefinal_result.params
    # Redo final fit varying all parameters
    params['x_0'].set(vary=True)
    params['x_1'].set(vary=True)
    params['x_2'].set(vary=True)
    params['phi_0'].set(vary=True)
    params['phi_1'].set(vary=True)
    finalresult = minimize(S11_residual, params, args=(f, sparam, ftype))
    params = finalresult.params

    return params, f, sparam, finalresult

