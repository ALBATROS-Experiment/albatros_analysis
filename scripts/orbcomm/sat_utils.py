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


#cxcorr maker for multiple sats

#get_SNR

#peak detection

#reliability finder


def get_cxcorr_many_sats(p0_ra,
                         p0_nra, 
                         tle_path, 
                         times, 
                         sats_present,
                         satmap,
                         coords,
                         N,
                         dN,
                         T_SPECTRA = 4096 / 250e6,
                         c_acclen = 10**6):

    nchans = len(p0_ra[0,:])
    freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)
    cx = []

    pulse_start, pulse_end = times[0], times[1]
    ra_coords, nra_coords = coords[0], coords[1]

    p0_nra_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
    niter = int(pulse_end - pulse_start) + 1  # run it for an extra second to avoid edge effects

    #GET GEO DELAY
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
    
    #UNCORRECTED
    cx.append(outils_g.coarse_xcorr(p0_ra, p0_nra, dN))  # no correction

    #CORRECTED
    for i, satidx in enumerate(sats_present):
        print("\nProcessing Satellite with ID:", satmap[satidx])
        outils_g.apply_delay(p0_nra, delays[:,i], freqs, out=p0_nra_delayed)
        cx.append(outils_g.coarse_xcorr(p0_ra, p0_nra_delayed, dN))

    return cx



def get_good_chunk(ra_obj,
                   nra_obj,
                   nchans,
                   c_acclen = 10^6):
    
    p0_ra = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
    p0_nra = cp.zeros((c_acclen, nchans), dtype="complex64")

    ra_start = ra_obj.spec_num_start
    nra_start = nra_obj.spec_num_start
    for i, (chunk_ra, chunk_nra) in enumerate(zip(ra_obj, nra_obj)):
        perc_missing_ra = (1 - len(chunk_ra["specnums"]) / c_acclen) * 100
        perc_missing_nra = (1 - len(chunk_nra["specnums"]) / c_acclen) * 100
        print("missing a1", perc_missing_ra, "missing a2", perc_missing_nra)
        if perc_missing_ra > 10 or perc_missing_nra > 10:
            ra_start = ra_obj.spec_num_start
            nra_start = nra_obj.spec_num_start
            continue
        
        bdc.make_continuous_gpu(chunk_ra['pol0'],chunk_ra['specnums']-ra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_ra)
        bdc.make_continuous_gpu(chunk_nra['pol0'],chunk_nra['specnums']-nra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_nra)
        break
            

    #add some checking here!!

    return p0_ra, p0_nra



def get_rel_ratio(data_gpu):
    data_cpu = cp.asnumpy(data_gpu)

    peak_location = cp.argmax(data_gpu)
    peak_data = data_gpu[peak_location - 200:peak_location + 200]
    peaks_total = find_peaks(data_cpu, height=0.001)

    heights = peaks_total[1]['peak_heights']
    height_indices = np.argsort(heights)
    tallest = heights[height_indices[-1]]
    reps, total = 4, 0
    for i in range(reps):
        total += (tallest - heights[height_indices[-(i+2)]])
    reliability_ratio = (total/(tallest * reps)) *100
    
    return reliability_ratio



def get_detections(cx, snr_array, temp_satmap):
    
    nchans = len(snr_array[0,:])
    detected_sats = np.zeros(nchans, dtype="int")
    detected_peaks = np.zeros(nchans, dtype="int")
    rel_ratios = np.zeros(nchans, dtype="int")

    for chan in range(nchans):
        sortidx = np.argsort(snr_array[:, chan])

        #if biggest SNR is for non-beamformed, skip the chan automatically
        if (sortidx[-1] == 0):  
            continue

        #below is the minimum condition of SNR for a detection. Most basic requirement
        diff = snr_array[sortidx[-1], chan] - snr_array[sortidx[-2], chan]
        tol = 5 * np.sqrt(2)
        if diff>tol: 
            cx_idx = sortidx[-1] # which cxcorr has the detection
            print(f"\nDetected Peak in cx index {cx_idx} in channel {chan}")
            satID = temp_satmap[cx_idx]
            print("SatID of detected peak:", satID)

            data_gpu = cp.abs(cx[sortidx[-1]][chan,:])
            rel_ratio = get_rel_ratio(data_gpu)
        
            #detected = graduates from pass to pulse. also picks what channels detection happens
            detected_sats[chan] = temp_satmap[sortidx[-1]]
            detected_peaks[chan] = cp.argmax(cp.abs(cx[sortidx[-1]][chan,:]))
            rel_ratios[chan] = rel_ratio

    return detected_sats, detected_peaks, rel_ratios

                
    
def get_consensus_offset(data):

    print("------GETTING CONSENSUS OFFSETS-----")
    #first we extract all the offset information
    all_SO = []
    rel_SO = []
    for pulse_dict in data:
        print("pulse details", pulse_dict)
        if len(pulse_dict["sats_present"]) > 1:  #for now only worry about one-sat pulses
            continue
        ind_offset = pulse_dict["individual_offset"] #individual offset

        REL = True
        #verify that none of the channels have an unreliable offset. If it passes, add it to reliable list.
        satIDs = list(pulse_dict['sats_present'].keys())
        for satID in satIDs:
            satinfo = pulse_dict['sats_present'][satIDs[0]]
            for detection in satinfo:
                print(detection)
                if detection[1] != 'RELIABLE':
                    REL = False
        
        all_SO.append(ind_offset)
        if REL:
            rel_SO.append(ind_offset)

    print("ALL Specnum Offsets", all_SO)
    print("\nRELIABLE Specnum Offsets", rel_SO)

    #to check for possible anomalies
    tolerance = 50000
    diff_SO = np.diff(all_SO)
    print("\nDiff array", diff_SO)
    for (i, delta)  in enumerate(diff_SO):
        if np.abs(delta) > tolerance:
            print(f"CAUTION: there is an anomaly between sats {i-1} and {i}")

        print("no anomalies between individual offsets")

    # we assume the passes are all within one day
    if len(rel_SO)>=3:
        con_off = int(stats.mode(rel_SO)[0])
        print('enough reliable offsets')
    elif len(all_SO)>=2:               #(this should be at least 5, implemented once possible)
        con_off = int(stats.mode(all_SO)[0])  
        print('not enough reliable offsets, but enough regular offsets')   
    else:
        raise ValueError("not enough offsets for antenna this antenna")

    return int(con_off)



