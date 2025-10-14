import os
import sys
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np
import cupy as cp
import numba as nb
import time
from scipy import linalg
from scipy import stats
from scipy import signal as sn 
from matplotlib import pyplot as plt
from datetime import datetime as dt
from src.correlations import baseband_data_classes as bdc
from src.utils import baseband_utils as butils
from src.utils import orbcomm_utils as outils
from src.utils import orbcomm_utils_gpu as outils_g
import json
from scipy.signal import find_peaks
from scripts.xcorr import helper as hp
from scripts.xcorr import helper_gpu as hpg
import sat_utils as su



def get_cxcorr_many_sats(p0_ref,
                         p0_nref, 
                         tle_path, 
                         times, 
                         sats_present,
                         satmap,
                         coords,
                         N,
                         dN,
                         T_SPECTRA = 4096 / 250e6,
                         c_acclen = 10**6):

    nchans = len(p0_ref[0,:])
    freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)
    cx = []

    pulse_start, pulse_end = times[0], times[1]
    ref_coords, nref_coords = coords[0], coords[1]
    p0_nra_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
    niter = int(pulse_end - pulse_start) + 1  # +1 to avoid edge effects

    #GET GEO DELAY
    delays = np.zeros((c_acclen, len(sats_present)))
    for i, satidx in enumerate(sats_present):
        d = outils.get_sat_delay(
            ref_coords,nref_coords,tle_path,pulse_start,niter,satmap[satidx]
            )
        delays[:, i] = np.interp(
            np.arange(0, c_acclen) * T_SPECTRA, np.arange(0, niter), d
        )
    delays = cp.asarray(delays)
    
    #UNCORRECTED
    cx.append(outils_g.coarse_xcorr(p0_ref, p0_nref, dN))  # no correction

    #CORRECTED
    for i, satidx in enumerate(sats_present):
        print("\nProcessing Satellite with ID:", satmap[satidx])
        outils_g.apply_delay(p0_nref, delays[:,i], freqs, out=p0_nra_delayed)
        cx.append(outils_g.coarse_xcorr(p0_ref, p0_nra_delayed, dN))

    return cx



def get_vis_gpu(pulse_start_t,
                pulse_end_t,
                paths,
                offsets,
                T_SPECTRA = 4096/250e6,
                v_acclen = 30000):
    ''' 
    computes visibilities for one baseline for a set period, given a specnumoffset

    note that this is just a regular CPU visibility computation, mainly useful for sanity checks
    also note that this should be tested with two non-ref antenna (usually run with one ref one non ref)

    '''
    chunk_length = T_SPECTRA * v_acclen
    pulse_len_chunks = int(np.ceil((pulse_end_t - pulse_start_t)/chunk_length))

    idxs, files = hp.get_init_info_all_ant(pulse_start_t, pulse_end_t, offsets, paths)

    channels = bdc.get_header(files[0][0])["channels"].astype('int64')
    chanstart = np.where(channels == 1834)[0][0] 
    chanend = np.where(channels == 1852)[0][0]
    print('starting, ending channels:', chanstart, chanend)
    chanlist = np.arange(1834, 1852)

    vis, channels = hpg.xcorr_avg(idxs, files, v_acclen, pulse_len_chunks, chanlist)
    
    return vis, channels

