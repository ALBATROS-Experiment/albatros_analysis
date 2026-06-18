import os
import sys
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np
import numba as nb
import time
import psutil
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
import matplotlib.cm as cm
from skyfield.api import load, EarthSatellite, Topos, wgs84
from datetime import datetime, timezone
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time

def median_abs_deviation(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return mad

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e6  # Resident Set Size in MB
    print(f"[{note}] Memory usage (RSS): {mem:.2f} MB")



def get_risen_sats2(tle_file, coords, t_start, satlist, dt=5, niter=560, good=None, altitude_cutoff=1):
    """Get all satellites risen at a particular point on earth at a list of epochs.
    Epochs start at t_start and a list of risen satellites is returned for every t_start + i * dt epoch
    The satellites are read form a TLE file.

    Parameters
    ----------
    coords : tuple of floats
        (latitude, longitude, elevation) of the position on Earth. Elevation is measured in meteres.
    t_start : float
        Start timestamp (ctime). Converted to JD internally.
    dt : float, optional
        Delta between epochs, by default None which sets it to 6.44 seconds internally (accumulation time of direct spectra).
    niter : int, optional
        Number of iterations, by default 560
    elevation_cutoff : float, optional
        Altitude cutoff (in degrees) above which a satellite is considered risen, by default 1 degree.

    Returns
    -------
    risen_sats : list of lists
        One list of risen satellites per epoch. Each epoch's list carries the name of the risen satellite at that epoch.
        E.g. [["FM118","NOAA15"], ["NOAA15"]]
    """
    
    obs1 = sf.wgs84.latlon(*coords)
    sats = sf.load.tle_file(tle_file)

    tt = t_start
    ts = sf.load.timescale()
    risen_sats = []

    print("Starting Time of", tt, "with a dt of", dt)
    for iter in range(niter):
        visible = []
        alt_count = 0
        jd = ctime2mjd(tt, type="JD")
        t = ts.ut1_jd(jd)

        for sat in sats:
            # if sat.model.satnum in junk: continue
            if sat.model.satnum not in good: continue
            # if (
            # "[+]" not in sat.name and "NOAA" not in sat.name
            # ):  # extracting operational ORBCOMM ([+]) and NOAA from TLE file
            # continue
            diff = sat - obs1
            topocentric = diff.at(t)
            alt, az, dist = topocentric.altaz()
            # print(sat.name)
            if alt.degrees > altitude_cutoff:
                # print(alt.degrees, az.degrees)
                # if sat.name is None:
                #     sat.name =
                visible.append([sat.model.satnum, alt.degrees, az.degrees])
        #         if(alt_count in (1,)):
        #             print(iter,'have ',alt_count,' in beam -6 dB range', visible)
        risen_sats.append(visible)
        tt += dt
    return risen_sats



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
        #tol = 5 * np.sqrt(2)
        tol = 20
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
        satIDs = [int(sat_id) for sat_id in pulse_dict['sats_present'].keys()]
        print(satIDs)
        for satID in satIDs:
            #need both sats to be reliable for it to be counted as reliable. 
            satinfo = pulse_dict['sats_present'][satID]
            print(satinfo)
            for detection in satinfo:
                print(detection)
                if detection[3] < 80:
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






def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def unix_to_lst(unix_t, coords):
    loc = EarthLocation(lon=coords[1]*u.deg, lat=coords[0]*u.deg, height=coords[2]*u.m)
    t_obj = Time(unix_t, format="unix", location=loc)
    lst_time = t_obj.sidereal_time("mean").degree % 360
    return lst_time


def unix_to_lst_with_fix(start_unix, end_unix, coords):
    """ 
    Assume that interval can be no longer than 24 hrs, so that we can unwrap safely by 360
    """

    start_lst = unix_to_lst(start_unix, coords)
    end_lst = unix_to_lst(end_unix, coords)

    if start_lst > end_lst:
        return (start_lst, end_lst + 360)
    else:
        return (start_lst, end_lst)


def snr_times_single(json_path, batch_start_unix, batch_end_unix, antname, dt=1):
    data_all = load_json(json_path)
    data = data_all[f'{batch_start_unix}'][f'{antname}']

    # make time grid
    t_start = 0
    t_end = batch_end_unix - batch_start_unix

    time = np.arange(t_start, t_end + 1, dt)
    snr  = np.zeros_like(time, dtype=float)

    # extract pulse times (floats since chunked)
    for block in data:
        t0, t1 = block["times"]
        t0 -= batch_start_unix
        t1 -= batch_start_unix
        snr_vals = [x[0] for x in block["SNR, Chan, Sat"]] #beware name might change here
        n = len(snr_vals)

        # get chunk boundaries (since SNR will change across them)
        boundaries = np.linspace(t0, t1, n + 1)

        for i, val in enumerate(snr_vals):
            # seconds covered by this SNR value
            mask = (time >= boundaries[i]) & (time < boundaries[i + 1])
            snr[mask] = val

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.rcParams.update({
                "font.size": 16,
                "axes.labelsize": 16,
                "axes.titlesize": 20,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "figure.titlesize": 22,
                "figure.dpi": 100,
                "savefig.dpi": 300
            })
    ax.step(time, snr2db(snr), where="post")
    ax.set_xlabel(f"Time after Batch Start ({int(dt)} s)")
    ax.set_ylabel("SNR (dB)")
    fig.suptitle(f"Antenna 1 - {antname}")
    ax.grid(True)
    plt.tight_layout()
    
    return time, snr, fig

def snr2db(snrarr):
    """
    Sends snr as ratio to dB
    By convention of snr_times output, sends snr of zero to 0 dB.  
    """
    snrarr = np.asarray(snrarr)
    snr_db = np.where(snrarr > 0, 10 * np.log10(snrarr), 0.0)
    return snr_db


def snr_times_many(json_paths, 
                    batch_starts, 
                    batch_ends, 
                    antname, 
                    dt=1, 
                    coords = [79.41717895, -90.76721818, 188.095],
                    T_SPECTRA = 4096/250e6,
                    c_acclen = 3e6):
    #using coords of MARS 1 as a reference for LST
    assert len(json_paths) == len(batch_starts)
    assert len(json_paths) == len(batch_ends)
    nbatches = len(json_paths)

    secs, lsts, snrs = [], [], []
    for i in range(nbatches):
        data = load_json(json_paths[i])[f'{antname}']
        secs.append(np.arange(0, batch_ends[i]-batch_starts[i] + 1, dt))
        lsts.append(unix_to_lst_with_fix(batch_starts[i], batch_ends[i], coords))
        snr  = np.zeros_like(secs[i], dtype=float)

        # extract pulse times (floats since chunked)
        for pulse in data:
            t0, t1 = pulse["times"]
            t0 -= batch_starts[i]
            t1 -= batch_starts[i]
            snr_vals = [x[0] for x in pulse["SNR, Chan, Sat"]]
            n = len(snr_vals)

            # get chunk boundaries (since SNR will change across them)
            boundaries = np.linspace(t0, t1, n + 1)

            for j, val in enumerate(snr_vals):
                # seconds covered by this SNR value
                mask = (secs[i] >= boundaries[j]) & (secs[i] < boundaries[j + 1])
                snr[mask] = val
        snrs.append(snr)

    start_lsts = [float(lst[0]) for lst in lsts]
    print('start lsts', start_lsts)
    lst_min = min(start_lsts)
    print('min lst', lst_min)

    secs_raw = secs.copy()
    snrs_raw = snrs.copy()

    lst_to_sec = 240
    aligned_secs = []
    aligned_snrs = []

    #padding at the start
    for sec, snr, start_lst in zip(secs, snrs, start_lsts):
        shift_sec = int((start_lst - lst_min) * lst_to_sec)
        if shift_sec > 0:
            print(f'Front padding triggered at {shift_sec}')
            snr = np.pad(snr, (shift_sec, 0), constant_values=0)
            sec = np.concatenate([np.arange(-shift_sec, 0), sec])
        aligned_snrs.append(snr)
        aligned_secs.append(sec)

    max_len = max(len(s) for s in aligned_snrs)
    print('max array len', max_len)

    for i in range(len(aligned_snrs)):
        pad = max_len - len(aligned_snrs[i])
        print('pad', pad)
        if pad > 0:
            aligned_snrs[i] = np.pad(
                aligned_snrs[i], (0, pad), constant_values=0
            )
            aligned_secs[i] = np.concatenate(
                [aligned_secs[i], np.arange(max(aligned_secs[i]), max(aligned_secs[i]) + pad)]
            )

    
    fig, ax = plt.subplots(nbatches, 1, figsize=(12, 4 * nbatches))
    plt.rcParams.update({
                "font.size": 16,
                "axes.labelsize": 16,
                "axes.titlesize": 20,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "figure.titlesize": 22,
                "figure.dpi": 100,
                "savefig.dpi": 300
            })
    if nbatches == 1:
        ax = [ax]
    fig.suptitle(f"Antenna 1 - {antname} (aligned by LST, integration time ~{int(c_acclen * T_SPECTRA)} s)")
    for i in range(nbatches):
        ax[i].step(aligned_secs[i], snr2db(aligned_snrs[i]), where="post")
        ax[i].set_ylabel("SNR (dB)")
        ax[i].grid(True)
    
    ax[nbatches-1].set_xlabel(f"Time after Batch Start ({int(dt)} s)")
    plt.tight_layout()
    
    return secs_raw, snrs_raw, fig 
