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
