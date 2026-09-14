### IMPORTS ###

import logging
logger = logging.getLogger(__name__)

import numpy as np
import cupy as cp
import scipy
import time
# from scipy.interpolate import CubicSpline
from cupyx.scipy.interpolate import CubicSpline
from cupyx.scipy.signal import decimate

import sys, os

# Adds a new string to the path that points to the home directory
# If you have a clone of the albatros_analysis repo there, this allows you to
# import some useful functions
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp, fft, ifft

from albatros_analysis.src.utils import pfb_utils as pu
from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde import codes
    
### IPFB ###

def setup_ipfb(final_channels, ipfb_chunk_size, args=default_args):
    ipfb = pu.StreamingIPFB_IQ(
        args.num_ant,
        args.num_pol,
        final_channels,
        nblock = ipfb_chunk_size,
        ntap = args.num_pfb_tap,
        window = "hamming",
        cut = args.cutsize,
    ) # Reminder: the lblock parameter of this function does absolutely nothing!
    # Do not set it! You will get confused. Use ipfb.lblock to determine it
    logger.debug("IPFB channels: %s", ipfb.channels)
    # ipfb_start_freq = final_channels[0] * params.adc_samp_freq / params.len_pfb_init #center freq of start chan
    logger.debug("IPFB: %s", ipfb)
    
    return ipfb
    
### DOWNCONVERSION ###

# The Elementwise Kernel allows us to parallelize operations on an array
# This one is for digital downconversion
ddc_kernel = cp.ElementwiseKernel(
    in_params='complex64 ts_in, float64 frac_freq, float64 start_cycles',
    out_params='complex64 ts_out',
    operation='''
        const double TWO_PI = 2 * 3.14159265358979323846;
        
        double cycles = frac_freq * i + start_cycles;
        double mod_cycle = cycles - floor(cycles);
        
        double wrapped_phase = mod_cycle * TWO_PI;
        
        //Cast to FP32 and use fast hardware trig
        float im, re;
        sincosf((float)(-wrapped_phase), &im, &re); // exp (-j 2 pi carrier t)
        
        complex<float> phasor(re, im);
        ts_out = ts_in * phasor;
    ''',
    name='ddc_kernel'
)

def process(ant_idx, pol0, pol1, final_channels, ipfb, args=default_args):
    """Processing one chunck at a time no longer supported!

    Parameters
    ----------
    pol0, pol1 :
        Saved data from polarizations 0 and 1
    final_channels :

    ipfb :

    """

    ts_pol0 = ipfb.ipfb(ant_idx, 0, pol0, thresh=args.ipfb_filt_thresh)
    ts_pol1 = ipfb.ipfb(ant_idx, 1, pol1, thresh=args.ipfb_filt_thresh)

    len_timestream = len(ts_pol0)
    ts_pol0_dc = xp.empty(len_timestream, dtype="complex64")
    ts_pol1_dc = xp.empty(len_timestream, dtype="complex64")

    new_samp_rate = args.adc_samp_freq * ipfb.lblock / args.len_pfb_init
    decimate_factor = int(new_samp_rate / args.code_baudrate)
    len_dec_timestream = (len_timestream - 1) // decimate_factor + 1

    filtered_timestreams = xp.zeros((len(args.ionosonde_freqs), args.num_pol, len_dec_timestream), dtype="complex64")

    logger.info(f"Timestream length is {len_timestream }.")
    logger.info(f"Decimation factor is {decimate_factor}.")
    logger.info(f"Decimated timestream length should be {len_dec_timestream// decimate_factor}.")
    logger.info(f"The lblock used by the IPFB was {ipfb.lblock}.")
    logger.info(f"The new sampling rate after the IPFB is {new_samp_rate/1e6:.2f} MHz.")
    ipfb_start_freq = final_channels[0] * args.adc_samp_freq/args.len_pfb_init

    # begin loop over frequencies
    for fi, freq in enumerate(args.ionosonde_freqs):
        
        ddc_freq = (freq - ipfb_start_freq) / new_samp_rate #normalized
        
        ddc_kernel(ts_pol0, ddc_freq, 0, ts_pol0_dc)
        ddc_kernel(ts_pol1, ddc_freq, 0, ts_pol1_dc)

        filtered_timestreams[fi, 0, :] = decimate(ts_pol0_dc, q = decimate_factor, ftype='fir')
        filtered_timestreams[fi, 1, :] = decimate(ts_pol1_dc, q = decimate_factor, ftype='fir')
    
    # Perform correlation
    # Switch this so that the code spectrum is being obtained at new_samp_rate / decimate_factor
    # Instead of the original code_baudrate
    code_spectra = codes.get_code_spectra(filtered_timestreams.shape[2], samp_rate = new_samp_rate / decimate_factor, args=args)
    corr = ifft( fft(filtered_timestreams, axis=2) * cp.conj(code_spectra[None,:, :]), axis=2)
    
    return corr, new_samp_rate / decimate_factor