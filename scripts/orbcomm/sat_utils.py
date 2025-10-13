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

def get_bline_dist(coord1, coord2):
    ''' 
    returns the physical distance between two coordinates in meters
    (i.e. magnitude of baseline vectors)
    '''
    lat1, lon1, alt1 = coord1[0], coord1[1], coord1[2]
    lat2, lon2, alt2 = coord2[0], coord2[1], coord2[2]

    mean_lat = np.radians((lat1 + lat2) / 2)

    meters_per_deg_lat = 111_320 
    meters_per_deg_lon = 111_320 * np.cos(mean_lat)

    delta_lat_deg = lat2 - lat1
    delta_lon_deg = lon2 - lon1
    delta_alt = float(alt2 - alt1)

    delta_lat_m = float(delta_lat_deg * meters_per_deg_lat)
    delta_lon_m = float(delta_lon_deg * meters_per_deg_lon)

    dist_total = np.sqrt(delta_lat_m**2 + delta_lon_m**2 + delta_alt**2)

    return dist_total



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


def get_detections(cx, snr_array, temp_satmap):
    
    nchans = len(snr_array[0,:])
    detected_snrs = np.zeros(nchans, dtype="int")
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
            snr = snr_array[cx_idx, chan]
            print(f"\nDetected Peak in cx index {cx_idx} in channel {chan}")
            satID = temp_satmap[cx_idx]
            print("SatID of detected peak:", satID)

            data_gpu = cp.abs(cx[sortidx[-1]][chan,:])
            rel_ratio = get_rel_ratio(data_gpu)
        
            #detected = graduates from pass to pulse. also picks what channels detection happens
            detected_sats[chan] = temp_satmap[sortidx[-1]]
            detected_peaks[chan] = cp.argmax(cp.abs(cx[sortidx[-1]][chan,:]))
            detected_snrs[chan] = snr
            rel_ratios[chan] = rel_ratio

    return detected_sats, detected_peaks, detected_snrs, rel_ratios




def get_rel_ratio(data_gpu):
    #data may already be a numpy array but this just makes sure
    data_cpu = cp.asnumpy(data_gpu)
    peak_location = np.argmax(data_cpu)
    peak_data = data_cpu[peak_location - 200:peak_location + 200]
    peaks_total = find_peaks(peak_data, height=0.001)

    heights = peaks_total[1]['peak_heights']
    if len(heights)<2:
        return 0
    height_indices = np.argsort(heights)
    tallest = heights[height_indices[-1]]
    reps, total = 4, 0
    for i in range(reps):
        total += (tallest - heights[height_indices[-(i+2)]])
    reliability_ratio = (total/(tallest * reps)) *100
    
    return reliability_ratio


def get_consensus_offset(data):
    all_SO, rel_SO = [], []
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



def get_vis(pulse_start_t,
            pulse_end_t,
            paths,
            offsets,
            T_SPECTRA = 4096/250e6,
            v_acclen = 5000):
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

    vis, rowcount, obj = hp.get_avg_fast(paths[0], 
                                         paths[1], 
                                         pulse_start_t, 
                                         pulse_end_t, 
                                         0, 
                                         v_acclen, 
                                         pulse_len_chunks, 
                                         chanstart=chanstart, 
                                         chanend=chanend)
    
    pol0, pol1 = vis[:,:,0,:], vis[:,:,1,:]
    return pol0, pol1, rowcount, obj, chanlist


def get_fringes_phase(vis, chanlist, chan = None):
    print(vis.shape)
    p_vis = np.angle(vis)
    #auto-select brightest phase
    mean_amp = np.mean(np.abs(vis), axis=0)
    chan_s_idx = np.argmax(mean_amp)
    chan_b_idx = chanlist[chan_s_idx]
    phase = np.unwrap(p_vis[:, chan_s_idx]) - p_vis[0, chan_s_idx] #zero the initial phase
    return p_vis, phase, chan_b_idx
