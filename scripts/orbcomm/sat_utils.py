import os
import sys
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np
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
import json
from scipy.signal import find_peaks
from scripts.xcorr import helper as hp


def get_complex_snr(signal_data, noise_data): #FIX THIS so it can overcome big signal in middle
    signal = np.max(np.abs(signal_data))
    im_std = np.std(noise_data.imag)
    re_std = np.std(noise_data.real)
    std = np.sqrt(im_std**2 + re_std**2)

    return signal/std


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


def get_rel_ratio(data):
    #data may already be a numpy array but this just makes sure
    assert isinstance(data, np.ndarray)
    peak_location = np.argmax(data)
    peak_data = data[peak_location - 200:peak_location + 200]
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


def get_detections(cx, snr_array, temp_satmap):
    '''
    determines detections given a coarse-cross correlation and SNR of a satellite pass

    for each frequency channel, we sort the SNR across all correlations (uncorrected and per-sat beamform)
    we skip the channel if uncorrected (cx[0]) has highest SNR (i.e. if beamforming didn't improve SNR)
    if beamform SNR improves uncorr SNR by factor of (5 * √2) or more, count as detection
    get detections for each chan for each sat
    get reliability ratio from the cxcorr peak shape

    note that we limit each channel to one detection
    (i.e. multiple sats cannot be detected in the same chan)
    

    Parameters
    ----------


    Returns
    -------
    detected_sats
    detected_peaks
    detected_snrs
    rel_ratios
    '''
    for cxcorr in cx:
        assert isinstance(cxcorr, np.ndarray)
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

            data = np.abs(cx[sortidx[-1]][chan,:])
            rel_ratio = get_rel_ratio(data)
        
            #detected = graduates from pass to pulse. also picks what channels detection happens
            detected_sats[chan] = temp_satmap[sortidx[-1]]
            detected_peaks[chan] = np.argmax(np.abs(cx[sortidx[-1]][chan,:]))
            detected_snrs[chan] = snr
            rel_ratios[chan] = rel_ratio

    return detected_sats, detected_peaks, detected_snrs, rel_ratios



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



def get_vis_cpu(pulse_start_t,
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

    vis, rowcount, obj = hp.get_avg_fast2(idxs,files,v_acclen,pulse_len_chunks,chanstart,chanend)
    
    pol0, pol1 = vis[:,:,0,:], vis[:,:,1,:]
    return pol0, pol1, rowcount, obj, chanlist


def get_fringes_phase(vis, chanlist):
    p_vis = np.angle(vis)
    mean_amp = np.mean(np.abs(vis), axis=1)
    chan_s_idx = np.argmax(mean_amp)
    chan_b_idx = chanlist[chan_s_idx]
    phase = np.unwrap(p_vis[chan_s_idx, :]) - p_vis[chan_s_idx, 0] #zero the initial phase
    return p_vis.T, phase, chan_b_idx
