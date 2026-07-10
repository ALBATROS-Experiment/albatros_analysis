import numpy as np
# Cubic splines
from scipy.interpolate import CubicSpline

import os, sys
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.scripts.ionosonde import params
from albatros_analysis.src.utils import pycufft

def clean_code(code_str):
    """
    Remove underscores from a code string and turn it into a numerical
    array where -1 are where there once were 0s and the 1s remain.
    """
    res = []
    
    for s in code_str.replace("_",""):
        match s:
            case "0":
                res.append(-1)
            case "1":
                res.append(1)
            case _:
                raise ValueError(f"Unknown value {s}. Expected 0 or 1.")

    return np.array(res)

def get_template(which_code = "1first", template_dt = 1 / params.code_baudrate,
                 smoothed = True, repeat_num = params.code_repeat_num) -> np.ndarray:
    '''Generate a repeated code template for later correlation. The two-code
    pattern specified by which_code is repeated repeat_num times.

    Optionally, the discrete template can be interpolated using a cubic spline
    and resampled at template_dt spacing.

    Parameters
    ----------
    which_code : {"0", "1", "0offset", "1offset", "0first", "1first"}, optional
        Determines which code is placed in each of the two code slots:

        - "0"       : code0 in the first slot only.
        - "1"       : code1 in the first slot only.
        - "0offset" : code0 in the second slot only.
        - "1offset" : code1 in the second slot only.
        - "0first"  : code0 in the first slot, code1 in the second slot.
        - "1first"  : code1 in the first slot, code0 in the second slot.

        A ValueError is raised if which_code is not one of these values.
    
    template_dt : float, optional
        Sampling interval of the returned smoothed template in seconds.
        Defaults to one sample per code symbol (1 / params.code_baudrate).

    smoothed : bool, optional
        If True, return a cubic-spline-interpolated version of the template.
        If False, return the original discrete template.

    repeat_num: int, optional
        Number of times the codes are repeated
        Defaults to params.code_repeat_num

    Returns
    -------
    template or smoothed_template : np.ndarray
        The generated template. If smoothed=True, the returned array is
        spline-resampled at template_dt; otherwise, it contains the original
        discrete symbol values.
    '''
    # Each code ends up being the number of symbols it has (usually 16)
    # multiplied by the number of times each symbol is repeated (usually 6)
    # This will then be broadcast at the code baudrate (usually 200 ksps, or
    # kilo-samples per second)
    # Codes are spaced by the interpulse period (abreviated IPP, usually 5.5 ms)
    per_code_len = params.code_len * params.code_snr_boost
    total_len = int(np.ceil(params.ipp * params.code_baudrate))
    
    code0 = clean_code(params.code0_str).repeat(params.code_snr_boost)
    code1 = clean_code(params.code1_str).repeat(params.code_snr_boost)

    # We first create a small template that we will then repeat
    # repeat_num times (usually 10)
    # Since we usually alternate broadcasting code 0 and code 1, this
    # will take 2 IPPs to broadcast
    small_template = np.zeros(2 * total_len)

    if which_code not in ["0", "1", "0offset", "1offset", "0first", "1first"]:
        raise ValueError(f"\"{which_code}\" is not an accepted value for which_code.")

    if which_code == "0" or which_code == "0first":
        small_template[:per_code_len] = code0
    elif which_code == "1" or which_code == "1first":
        small_template[:per_code_len] = code0

    if which_code == "0offset" or which_code == "1first":
        small_template[total_len:total_len + per_code_len] = code0
    elif which_code == "1offset" or which_code == "0first":
        small_template[total_len:total_len + per_code_len] = code1
    
    template = np.tile(small_template, repeat_num)

    if smoothed:
        smoothed_template_points = np.arange(2 * repeat_num * total_len) * template_dt

        cs = CubicSpline(np.arange(2 * repeat_num * total_len) / params.code_baudrate, template)
        smoothed_template = cs(smoothed_template_points)
 
        return smoothed_template
    else:
        return template


def get_code_spectra(len_timestream, which_code = "1first"):
    """
    Returns the spectrum of the desired code template (i.e. the desired code
    template, FFT'd).

    Parameters
    ----------
    len_timestream:
        AKA Nts_dc
    """
    code_templates_gpu = xp.zeros((1, len_timestream), dtype="complex64")
    code_template = get_template(which_code = which_code, smoothed = False)
    code_templates_gpu[0, : len(code_template)] = xp.asarray(code_template, dtype="complex64")
    code_spectra = pycufft.fft(code_templates_gpu, axis=1)
    return code_spectra