import logging
logger = logging.getLogger(__name__)

import numpy as np

import os, sys
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.scripts.ionosonde import params
from albatros_analysis.scripts.ionosonde import codes

def get_output_signal(freq_carrier, which_code = "0first", amplitude = 1e-4):
    """
    Generates the output signal produced by the ionosonde.

    Parameters
    ----------
    freq_carrier : float
        the frequency of the carrier wave that is modulated with the template
    which_code : {"0", "1", "0offset", "1offset", "0first", "1first"}, optional
        Determines which code is placed in each of the two code slots:

        - "0"       : code0 in the first slot only.
        - "1"       : code1 in the first slot only.
        - "0offset" : code0 in the second slot only.
        - "1offset" : code1 in the second slot only.
        - "0first"  : code0 in the first slot, code1 in the second slot.
        - "1first"  : code1 in the first slot, code0 in the second slot.

        A ValueError is raised if which_code is not one of these values.
    amplitude : float
        the amplitude of the transmitted signal
    """

    code_template = codes.get_template(which_code = which_code, repeat_num = 1)

    samp_mult = int(params.adc_samp_freq / params.code_baudrate) # FIX: Somethings make break if this isn't an integer

    num_time_samples = len(code_template) * samp_mult
    
    t = np.arange(num_time_samples) / params.adc_samp_freq
    carrier = amplitude * np.cos(2 * np.pi * freq_carrier * t)
    modulated = carrier * code_template.repeat(samp_mult)

    pol0 = np.tile(modulated, params.code_repeat_num)

    if params.num_pol == 1:
        return pol0
    elif params.num_pol == 2:
        pol1 = np.zeros(num_time_samples, dtype='float64') 
        return pol0, pol1
    else:
        raise ValueError(f"Number of polarizations must either be 1 or 2. {params.num_pol} is not an acceptable value.")