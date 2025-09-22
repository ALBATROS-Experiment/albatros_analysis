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


#cxcorr maker for multiple sats

#get_SNR

#peak detection

#reliability finder


def get_cxcorr_many_sats(nchans, 
                         tle_path, 
                         times, 
                         sats_present,
                         satmap,
                         bdc_objects,
                         coords,
                         files,
                         T_SPECTRA = 4096 / 250e6,
                         c_acclen = 10**6):

    pulse_start, pulse_end = times[0], times[1]
    ra, nra = bdc_objects[0], bdc_objects[1]
    ra_coords, nra_coords = coords[0], coords[1]
    ra_files, nra_files = files[0], files[1]

    #--------set up pol arrays---------

    p0_ra = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
    p0_nra = cp.zeros((c_acclen, nchans), dtype="complex64")
    p0_nra_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
    niter = int(pulse_end - pulse_start) + 1  # run it for an extra second to avoid edge effects

    #--------get geo delay (done for each sat in pass)-------
    delays = np.zeros((c_acclen, len(sats_present)))
    for i, satidx in enumerate(sats_present):
        d = outils.get_sat_delay(
            ra_coords,
            nra_coords,
            tle_path,
            pulse_start,
            niter,
            satmap[satidx],
        )
        delays[:, i] = np.interp(
            np.arange(0, c_acclen) * T_SPECTRA, np.arange(0, niter), d
        )
    delays = cp.asarray(delays)

    #----coarse xcorr, WITHOUT corrections----

    cx = []  # store coarse xcorr for each satellite
    N = 2 * c_acclen
    dN = min(100000, int(0.3 * N))
    print("2*N and 2*dN", N, dN)
    cx.append(outils_g.coarse_xcorr(p0_ra, p0_nra, dN))  # no correction


    #----Coarse xcorr, WITH correction----
    # aka beamformed visibilities

    freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)   #hard-coded. need to change for full spectra-variable functionality
    for i, satidx in enumerate(sats_present):
        print("\nProcessing Satellite with ID:", satmap[satidx])
        temp_satmap.append(satmap[satidx])
        # phase_delay = 2 * np.pi * delays[:, i : i + 1] @ freq
        # print("phase delay shape", phase_delay.shape)
        outils_g.apply_delay(p0_nra, delays[:,i], freqs, out=p0_nra_delayed)
        cx.append(
                outils_g.coarse_xcorr(
                    p0_ra, p0_nra_delayed, dN
                )
        )

    return specnum_offset, temp_satmap, cx

        