### IMPORTS ###

import numpy as np
import cupy as cp
import scipy

import sys, os

# Adds a new string to the path that points to the home directory
# If you have a clone of the albatros_analysis repo there, this allows you to
# import some useful functions
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.utils import pycufft
from albatros_analysis.src.utils import pfb_utils as pu
from albatros_analysis.scripts.ionosonde import params
from albatros_analysis.scripts.ionosonde import codes

### SOME UTILITIES ###

def add_zeros(arr, final_len):
    out = np.zeros(final_len)
    out[:len(arr)] = arr
    return out

def quantize(a):
    """One bit quantize a"""
    if a >= 0:
        return 1.
    else:
        return -1.
    
### IPFB ###

def setup_ipfb(final_channels, ipfb_chunk_size):
    ipfb = pu.StreamingIPFB_IQ(
        params.num_ant,
        params.num_pol,
        final_channels,
        nblock = ipfb_chunk_size,
        ntap = params.num_pfb_tap,
        window = "hamming",
        cut = params.cutsize,
    ) # Reminder: the lblock parameter of this function does absolutely nothing!
    # Do not set it! You will get confused. Use ipfb.lblock to determine it
    print("ipfb channels", ipfb.channels)
    # ipfb_start_freq = final_channels[0] * params.adc_samp_freq / params.len_pfb_init #center freq of start chan
    print(ipfb)
    
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

### FILTERING ###

def get_filter():
    cutoff = params.code_baudrate / params.adc_samp_freq
    h = scipy.signal.firwin(params.filter_len, cutoff=cutoff/2, window=('kaiser', 4),fs=1) #cutoff is one sided bandwidth
    h_cupy = cp.zeros((1, params.buf_len), dtype="complex64")
    h_cupy[0, :params.filter_len] = cp.asarray(h, dtype="complex64")
    hf = pycufft.fft(h_cupy, axis=1)

    return hf

def apply_filter(dc_ts, hf, filter_state_1d, buf_len = 4096):
    '''XX.

    Parameters
    ----------
    dc_ts : 
        Downconverted timestream to filter
    hf : 
        Filter coefficients. Roughly 1 in the channels we are interested in
        and 0 in the ones we throw away.
    filter_state_1d :
        Only non-zero if data is being processed in multiple chunks.
    buf_len :
        Length of the FFTs. Must be longer than the filter length.

    Returns
    -------
    out : np.ndarray
        The filtered, downconverted timestream.
    '''
    Nfilt = filter_state_1d.shape[0]
    ncols = buf_len - Nfilt
    nrows = dc_ts.shape[0] // ncols
    dc_ts = dc_ts[: nrows * ncols].reshape(nrows, ncols)

    inp = cp.zeros((nrows, buf_len), dtype="complex64") #buf_len is a fast FFT len, since we'll FFT input
    # print("inp shape is", inp.shape)
    inp[0, :Nfilt] = filter_state_1d
    inp[:, Nfilt:] = dc_ts[:, :]
    inp[1:, :Nfilt] = dc_ts[:-1, -Nfilt:]
    filter_state_1d[:] = dc_ts[-1, -Nfilt:]

    filt_inp = pycufft.ifft(pycufft.fft(inp, axis=1) * hf, axis=1)
    # print("filt_inp shape is", filt_inp.shape)
    out = cp.zeros((nrows, ncols), dtype="complex64")
    out[:, :] = filt_inp[:, Nfilt:]
    out = np.ravel(out)
    return out

def process_one_chunk(pol0, pol1, final_channels, hf, len_timestream, Nts_dc, ts_pol0_dc, ts_pol1_dc, ipfb,
          ant_idx, phase_cycles, filter_state,
          filtered_timestreams):
    """TODO: Write description.

    Parameters
    ----------
    pol0, pol1 :
        Saved data from polarizations 0 and 1
    final_channels :
        pass
    hf :
        Filter coefficients
    len_timestream :
        AKA Nts
    Nts_dc
    ts_pol0_dc, ts_pol1_dc:
        Here to be overwritten, but should be of the right shape (length Nts)
    ipfb : StreamingIPFB_IQ
        Save IPFB state
    ant_idx :
        Antenna index
    phase_cycles :
        Here to keep track of phase so downconversion works between chunks
    filter_state :
        Here to keep track of filter state so that filtering works at chunk boundaries
    filtered_timestream :
        Here to be overwritten, but should be of the right shape
    """

    dsamp = int(params.adc_samp_freq * ipfb.lblock / params.len_pfb_init / params.code_baudrate)

    ts_pol0 = ipfb.ipfb(ant_idx, 0, pol0, thresh=params.filt_thresh)
    ts_pol1 = ipfb.ipfb(ant_idx, 1, pol1, thresh=params.filt_thresh)

    # begin loop over frequencies
    for fi, freq in enumerate(params.ionosonde_freqs):
        ipfb_start_freq = final_channels[0] * params.adc_samp_freq/params.len_pfb_init
        ddc_freq = (freq - ipfb_start_freq)/params.len_pfb_init #normalized
        
        ddc_kernel(ts_pol0, ddc_freq, phase_cycles[fi], ts_pol0_dc)
        ddc_kernel(ts_pol1, ddc_freq, phase_cycles[fi], ts_pol1_dc)

        # update the phase for the next chunk. Remember two_pi_t starts from 0.

        phase_cycles[fi] += ddc_freq * len_timestream
        phase_cycles[fi] -= cp.floor(phase_cycles[fi]) # keep it between 0 and 1
        # print("new phase cycles", phase_cycles[fi])

        # filter the downconverted ts, sampling rate 5 us
        # FIX: Use separate filter states for pol0 and pol1
        ts_pol0_filt = apply_filter(ts_pol0_dc, hf, filter_state[fi, 0], buf_len = 4096)[::dsamp]  
        ts_pol1_filt = apply_filter(ts_pol1_dc, hf, filter_state[fi, 1], buf_len = 4096)[::dsamp]
        # print("ts_pol0_filt shape is", ts_pol0_filt.shape, "and ts_pol1_filt shape is", ts_pol1_filt.shape)
        filtered_timestreams[fi, 0, :] = ts_pol0_filt
        filtered_timestreams[fi, 1, :] = ts_pol1_filt
    
    # Perform correlation
    code_spectra = codes.get_code_spectra(Nts_dc)
    corr = pycufft.ifft( pycufft.fft(filtered_timestreams, axis=2) * cp.conj(code_spectra[None,:, :]), axis=2)
    
    return corr