import numpy as np
# Cubic splines
from scipy.interpolate import CubicSpline

import os, sys
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp, fft

from albatros_analysis.scripts.ionosonde.params import default_args

import logging
if __name__ == "__main__":
    logger = logging.getLogger("albatros_analysis.scripts.ionosonde")
else:
    logger = logging.getLogger(__name__)

def clean_code(code_str, args = default_args):
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

def get_template(which_code = "1", code_repeat_num = None, args = default_args) -> np.ndarray:
    '''Generate a repeated code template for later correlation. The two-code
    pattern specified by which_code is repeated repeat_num times.

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

    Returns
    -------
    template : np.ndarray
        The generated template.
    '''
    if code_repeat_num is None:
        code_repeat_num = args.code_repeat_num
    # Each code ends up being the number of symbols it has (usually 16)
    # multiplied by the number of times each symbol is repeated (usually 6)
    # This will then be broadcast at the code baudrate (usually 200 ksps, or
    # kilo-samples per second)
    # Codes are spaced by the interpulse period (abreviated IPP, usually 5.5 ms)
    per_code_len = args.code_len * args.code_snr_boost
    total_len = int(np.ceil(args.ipp * args.code_baudrate))
    
    code0 = clean_code(args.code0_str, args=args).repeat(args.code_snr_boost)
    code1 = clean_code(args.code1_str, args=args).repeat(args.code_snr_boost)

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
        small_template[:per_code_len] = code1

    if which_code == "0offset" or which_code == "1first":
        small_template[total_len:total_len + per_code_len] = code0
    elif which_code == "1offset" or which_code == "0first":
        small_template[total_len:total_len + per_code_len] = code1
    
    template = np.tile(small_template, code_repeat_num)

    return template

def get_smoothed_template(samp_rate, which_code = "1", code_repeat_num = None, args = default_args) -> np.ndarray:
    '''Generate a repeated code template for later correlation. The two-code
    pattern specified by which_code is repeated repeat_num times. The discrete
    template is interpolated using a cubic spline and resampled at samp_rate.

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

    Returns
    -------
    smoothed_template : np.ndarray
        The smoothed template. 
    '''
    if code_repeat_num is None:
        code_repeat_num = args.code_repeat_num
    
    template = get_template(which_code = which_code, code_repeat_num = code_repeat_num, args = args)

    template_times = np.arange(len(template)) / args.code_baudrate
    smoothed_template_len = int(template_times[-1] * samp_rate) + 1
    smoothed_template_times = np.arange(smoothed_template_len) / samp_rate

    cs = CubicSpline(template_times, template)
    smoothed_template = cs(smoothed_template_times)

    return smoothed_template


# def get_code_spectra(len_timestream, which_code = "1first", code_repeat_num = None, args=default_args):
#     """
#     Returns the spectrum of the desired code template (i.e. the desired code
#     template, FFT'd).

#     Parameters
#     ----------
#     len_timestream:
#         AKA Nts_dc
#     """
#     if code_repeat_num is None:
#         code_repeat_num = args.code_repeat_num

#     code_templates_gpu = xp.zeros((1, len_timestream), dtype="complex64")
#     code_template = get_template(which_code = which_code, code_repeat_num = code_repeat_num, args=args)
#     code_templates_gpu[0, : len(code_template)] = xp.asarray(code_template, dtype="complex64")
#     code_spectra = fft(code_templates_gpu, axis=1)
#     return code_spectra

def get_code_spectra(len_timestream, samp_rate = None, which_code = "1first", code_repeat_num = None, args=default_args):
    """
    Returns the spectrum of the desired code template (i.e. the desired code
    template, FFT'd).

    Parameters
    ----------
    len_timestream:
        AKA Nts_dc
    """
    if code_repeat_num is None:
        code_repeat_num = args.corr_code_repeat_num

    code_templates_gpu = xp.zeros((1, len_timestream), dtype="complex64")
    if samp_rate is None:
        code_template = get_template(which_code = which_code, code_repeat_num = code_repeat_num, args=args)
    else:
        code_template = get_smoothed_template(samp_rate, which_code = which_code, code_repeat_num = code_repeat_num, args=args)
    logger.info(f"Length of code template is {len(code_template)}.")
    code_templates_gpu[0, : len(code_template)] = xp.asarray(code_template, dtype="complex64")
    code_spectra = fft(code_templates_gpu, axis=1)
    return code_spectra