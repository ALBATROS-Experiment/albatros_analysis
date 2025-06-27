import os
import sys
from sys import path
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np 
import numba as nb
import time
from scipy import linalg
from scipy import stats
from matplotlib import pyplot as plt
from datetime import datetime as dt
from src.correlations import baseband_data_classes as bdc
from src.utils import baseband_utils as butils
from src.utils import orbcomm_utils as outils
from scipy.optimize import least_squares, minimize_scalar
import json
import random
from datetime import datetime, timezone
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from skyfield.api import load, EarthSatellite, Topos, wgs84
import skyfield.api as sf
from scipy.linalg import block_diag
import math
from scipy.linalg import cho_factor, cho_solve, inv
from scipy.interpolate import interp1d
import numbers




def pred(fit_coords, ds, pulse_idx, data_list, context_list, zeroing = True):

    '''
    gives a predicted phase, with time offsets thrown in for each pulse.
    recall that dc is in fact a fit parameter
    '''
    assert hasattr(fit_coords, "__len__") and len(fit_coords) == 3, "fit_coords should really be a 1x3 numpy array"
    assert isinstance(zeroing, bool), "zeroing gotta be true/false"

    extension = 6
    
    start_time = time.time()

    #unpack from info list 
    relative_start_time, relative_end_time, global_start_time = data_list[pulse_idx][1][0], data_list[pulse_idx][1][1], data_list[pulse_idx][1][2]
    sat_ID, pulse_channel_idx = data_list[pulse_idx][2][0], data_list[pulse_idx][2][1]
    tle_path = data_list[pulse_idx][3]

    #unpack from context list
    visibility_window = context_list[0]
    T_SPECTRA, v_acclen, v_nchunks = context_list[1], context_list[2], context_list[3]
    ref_coords = context_list[4]

    #---------------------------------------------------------------------------------


    pulse_duration_sec = relative_end_time - relative_start_time
    time_start = global_start_time + relative_start_time

    #get delay with extension at the front, and large buffer at the back
    print('ref_coords', ref_coords)
    print('fit coords', fit_coords)
    d = outils.get_sat_delay_new(ref_coords, fit_coords, tle_path, time_start - extension, np.ceil(pulse_duration_sec + 10), sat_ID)

    #add a couple chunks to ensure this is not shorter than the observed data length
    pulse_duration_chunks = int(pulse_duration_sec / (T_SPECTRA * v_acclen)) + 5
    pulse_freq = outils.chan2freq(pulse_channel_idx, alias=True)

    #number of chunks we need to shift the interpolation array forward to match with the actual time_start zero (cancel out extension)
    buffer_chunks = extension / ((v_acclen * T_SPECTRA))

    #each entry represents a chunk index, but their value is in seconds that corresponds to that chunk in the d (delay) array
    interp_chunk_times = (np.arange(buffer_chunks, pulse_duration_chunks + buffer_chunks) * v_acclen * T_SPECTRA) + ds

    #get the delay values for each of these chunks
    delay = np.interp(interp_chunk_times, np.arange(len(d)), d)

    #get the predicted phase at each chunk

    if zeroing == True:
        pred = (-delay + delay[0]) * 2 * np.pi * pulse_freq

    if zeroing == False:
        pred = -delay * 2 * np.pi * pulse_freq

    print("time taken for one prediction", time.time() - start_time)

    return pred



def pred_mohan(fit_coords, dt, pulse_idx, data_list, context_list, zeroing = True):

    '''
    gives a predicted phase, with time offsets thrown in for each pulse.
    recall that dc is in fact a fit parameter
    '''
    
    start_time = time.time()

    #unpack from info list 
    relative_start_time, relative_end_time, global_start_time = data_list[pulse_idx][1][0], data_list[pulse_idx][1][1], data_list[pulse_idx][1][2]
    sat_ID, pulse_channel_idx = data_list[pulse_idx][2][0], data_list[pulse_idx][2][1]
    tle_path = data_list[pulse_idx][3]

    #unpack from context list
    visibility_window = context_list[0]
    T_SPECTRA, v_acclen, v_nchunks = context_list[1], context_list[2], context_list[3]
    ref_coords = context_list[4]

    #---------------------------------------------------------------------------------

    pulse_duration_sec = relative_end_time - relative_start_time
    time_start = global_start_time + relative_start_time
    print("time_start =", time_start, "type:", type(time_start))
    print("dt =", dt, "type:", type(dt))
    d = outils.get_sat_delay_new(ref_coords, fit_coords, tle_path, time_start + dt, np.ceil(pulse_duration_sec + 5), sat_ID)
    pulse_duration_chunks = int(pulse_duration_sec / (T_SPECTRA * v_acclen)) + 2
    pulse_freq = outils.chan2freq(pulse_channel_idx, alias=True)
    interp_chunk_times = (np.arange(0, pulse_duration_chunks) * v_acclen * T_SPECTRA)
    delay = np.interp(interp_chunk_times, np.arange(len(d)), d)

    #could return only delays? or maybe throw an option into the function?

    if zeroing == True:
        pred = (-delay + delay[0]) * 2 * np.pi * pulse_freq
    if zeroing == False:
        pred = -delay  * 2 * np.pi * pulse_freq

    print("time taken for one prediction", time.time() - start_time)

    return pred


#----------------------------------------------------------------------------------------------------------


def std_res_all(guess_coords, dt_list, pred, data_list, context_list, zeroing = True):
    '''weighted residuals for all pulses, using the standard deviation of the residuals'''
    guess_coords = np.array(guess_coords)

    res_all = []
    for pulse_idx, data in enumerate(data_list):

        
        if isinstance(dt_list, (list, np.ndarray)):
            predicted = pred(guess_coords, dt_list[pulse_idx], pulse_idx, data_list, context_list, zeroing = zeroing)[:len(data[4])]
        elif np.isscalar(dt_list) and dt_list == 0:
            predicted = pred(guess_coords, 0, pulse_idx, data_list, context_list, zeroing = zeroing)[:len(data[4])]

        res = data[4] - predicted
        std = np.std(res, ddof=1) 
        res_weighted = res / std
        res_all.append(res_weighted.flatten())
    
    return np.concatenate(res_all)



def std_res_ind(coords, dt, pulse_idx, pred, data_list, context_list, zeroing = True):
    '''weighted residuals for all pulses, using the standard deviation of the residuals'''

    coords = np.array(coords)
    predicted = pred(coords, dt, pulse_idx, data_list, context_list, zeroing = zeroing)[:len(data_list[pulse_idx][4])]
    res = data_list[pulse_idx][4] - predicted
    std = np.std(res, ddof=1)  # standard deviation for whitening
    res_weighted = res / std
    
    return res_weighted



def get_res_and_cov_blockwise(coords, dt_list, pred, data_list, context_list):
    residuals_all = []
    var_all = []

    for pulse_idx, data in enumerate(data_list):

        if isinstance(dt_list, (list, np.ndarray)):
            predicted = pred(coords, dt_list[pulse_idx], pulse_idx, data_list, context_list)[:len(data[4])]
        elif np.isscalar(dt_list) and dt_list == 0:  
            predicted = pred(coords, 0, pulse_idx, data_list, context_list)[:len(data[4])]  
  
        res = data[4] - predicted
        residuals_all.append(res.flatten())

        var = np.var(res, ddof=1)
        var_ind = np.eye(len(res)) * var
        var_all.append(var_ind)
        
    S = block_diag(*var_all)
    
    return np.concatenate(residuals_all), S



#----------------------------------------------------------------------------------------------------------------------------------

#I want the weighting to be done in the residual functions, as opposed to the fitting function. 
#that way there's a natural separation of methods that is done. moreover, I need to feed normal residuals into the least_squares function
#if I want it to be streamlined. so if I feed in 


def solid_fit(initial_coordinates, dt_list, pred, data_list, context_list, weight_type = 'std'):
    '''fits for coordinates only, optionally given a certain per-pulse time offset
    if you want no dt time offsets, just set it to zero'''

    initial_coordinates = np.array(initial_coordinates)

    if weight_type == 'std':
        fit = least_squares(
            fun = std_res_all,
            x0 = initial_coordinates,
            args=(dt_list, pred, data_list, context_list)
            )

        #this part is a little sketchy so double check
        R, S = get_res_and_cov_blockwise(fit.x, dt_list, pred, data_list, context_list)
        

    J = fit.jac

    print(J.shape)
    print(S.shape)
    param_S = inv(J.T @ S @ J)
    param_err = np.sqrt(np.diag(param_S))

    return fit.x, param_err






def offset_fit(coords, pred, data_list, context_list, weight_type = 'std', method = 'trf'):
    ''' fits for time offsets only, given a certain non-ref coordinate and (optionally) with certain initial time offset guesses
    note that this is done individually for each pulse. they are treated as having independent offsets'''
    
    coords = np.array(coords)
    dt_all = []

    for pulse_idx in range(len(data_list)):
        if weight_type == 'std':
            result = least_squares(
            lambda dt: std_res_ind(coords, dt, pulse_idx, pred, data_list, context_list, zeroing = False), 
            0,
            #bounds=(-4.0, 4.0),
            method = 'lm'
            )

            if np.abs(result.x) > 4.0:
                result = least_squares(
                lambda dt: std_res_ind(coords, dt, pulse_idx, pred, data_list, context_list), 
                0,
                bounds=(-4.0, 4.0),
                method = 'trf'
                )

        dt_all.append(result.x)

    return dt_all





def joint_fit(initial_coords, pred, data_list, context_list, weight_type = 'std'):
    '''fits for coordinates and time offsets all together.'''
    
    initial_coords = np.array(initial_coords)
    n_pulses = len(data_list)
    dts = np.zeros(n_pulses)
    params = np.concatenate([initial_coords, dts])


    #clever bounding thing that I cannot claim responsibility for
    coord_bounds = ([-np.inf]*3, [np.inf]*3)
    ds_bounds = ([-4.0]*n_pulses, [4.0]*n_pulses)

    lower = coord_bounds[0] + ds_bounds[0]
    upper = coord_bounds[1] + ds_bounds[1]


    if weight_type == 'std': 
        fit = least_squares(
            lambda x: std_res_all(x[:3], x[3:], pred, data_list, context_list),
            x0 = params,
            bounds = (lower, upper),
            method = 'trf'
            )

        fitted_coords = fit.x[:3]
        dt_list = fit.x[3:]

        #this part is a little sketchy so double check
        R, S = get_res_and_cov_blockwise(fitted_coords, dt_list, pred, data_list, context_list)
        

    J = fit.jac

    print(J.shape)
    print(S.shape)
    param_S = inv(J.T @ S @ J)
    param_err = np.sqrt(np.diag(param_S))

    return fit.x, param_err










def interp_rect(vis):

    #interpolates rectangular form
    t = np.arange(len(vis))
    valid = ~vis.mask

    real_interp = interp1d(t[valid], vis[valid].real, kind='linear', fill_value="extrapolate")
    imag_interp = interp1d(t[valid], vis[valid].imag, kind='linear', fill_value="extrapolate")
    
    vis_interp = real_interp(t) + 1j * imag_interp(t) 
    return vis_interp  



def interp_polar(vis):

    #interpolates phase and magnitude, so in polar form
    t = np.arange(len(vis))
    valid = ~(np.isnan(vis.real) | np.isnan(vis.imag))
    amps = np.abs(vis)
    phases = np.angle(vis)

    amp_interp = interp1d(t[valid], amps[valid], kind='linear', fill_value="extrapolate")

    phase_unwrapped = np.unwrap(phases[valid])
    phase_interp = interp1d(t[valid], phase_unwrapped, kind='linear', fill_value="extrapolate")

    interp_amp = amp_interp(t)
    interp_phase = phase_interp(t)
    vis_interp = interp_amp * np.exp(1j * interp_phase)

    return vis_interp





def dist_components(coord1, coord2):
    ''' 
    Returns the actual physical distance between two coordinates. 
    Seperates the superficial component (latitude and longitude) and the altitude component into two seperate measurements
    '''

    lat1, lon1, alt1 = coord1[0], coord1[1], coord1[2]
    lat2, lon2, alt2 = coord2[0], coord2[1], coord2[2]

    mean_lat = math.radians((lat1 + lat2) / 2.0)

    # Approximate meters per degree
    meters_per_deg_lat = 111_320  # constant
    meters_per_deg_lon = 111_320 * math.cos(mean_lat)  # varies with latitude

    # Differences in degrees
    delta_lat_deg = lat2 - lat1
    delta_lon_deg = lon2 - lon1
    delta_alt = float(alt2 - alt1)  # already in meters

    # Convert angular differences to meters
    delta_lat_m = float(delta_lat_deg * meters_per_deg_lat)
    delta_lon_m = float(delta_lon_deg * meters_per_deg_lon)

    return delta_lat_m, delta_lon_m, delta_alt


def split_array(array, tolerance):
    for i in range(1, len(array)):
        if abs(array[i] - array[i - 1]) > tolerance:
            return array[:i], array[i:]
    
    return array, []

def get_overflow_index(array, tolerance):
    for i in range(1, len(array)):
        if abs(array[i] - array[i - 1]) > tolerance:
            return i




def satpass_plotter(info_list, obscoords, step_seconds=5, hard_list = [28654, 25338, 33591, 57166, 59051, 44387]):

    obslat, obslon, obselev = obscoords
    observer = wgs84.latlon(obslat, obslon, obselev)

    pulse_data = []
    
    for pulse in info_list:
        t_start, t_end, global_start_time = pulse[1]
        satID, chanbig = pulse[2]
        tle_path = pulse[3]

        #times
        global_pulse_start = t_start + global_start_time
        pulse_duration_secs = t_end - t_start
        ts = load.timescale()
        dt = datetime.fromtimestamp(global_pulse_start, tz=timezone.utc) #careful with time zone!!
        t0 = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    
        seconds = np.arange(0, pulse_duration_secs, step_seconds)
        times = t0 + seconds / (24 * 60 * 60)

        #get sat info
        sats = load.tle_file(tle_path)
        for sat in sats:
            if sat.model.satnum == satID:
                diff = sat - observer
                topocentric = diff.at(times)
                alt, az, _ = topocentric.altaz()

        #conversion
        az_rad = np.radians(az.degrees)
        el = alt.degrees

        pulse_data.append((satID, az_rad, el))

    
    #set up colors for label
    satIDs = sorted(set(pulse[0] for pulse in pulse_data))
    cmap = cm.get_cmap('tab10', len(satIDs)) 
    id2colour = {sat_id: cmap(i) for i, sat_id in enumerate(satIDs)}


    #plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    #loop over pulses
    for pulse in pulse_data:
        color = id2colour[pulse[0]]
        ax.plot(pulse[1], 90 - pulse[2], label=f"{pulse[0]}", color=color)

    # Zenith at center, horizon at outer edge
    ax.set_rlim(0, 90)
    ax.set_rlabel_position(225)  # Move radial labels away from overlap

    # Azimuth 0° = North at top, increase clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Add labels for cardinal directions
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])

    # Add grid and title
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    good_labels = []
    good_handles = []
    for i in range(len(labels)):
        if labels[i] not in good_labels:
            good_labels.append(labels[i])
            good_handles.append(handles[i])
    ax.legend(good_handles, good_labels)
    
    return fig



def make_coord_plot(initial_coordinate, tuple_list, title):

    labels = ['guess']
    lats = [0]
    lons = [0]

    for pair in tuple_list:
        labels.append(pair[0])
        lat_delta, lon_delta = dist_components(initial_coordinate, pair[1])[:2]
        lats.append(lat_delta)
        lons.append(lon_delta)

    fig, ax = plt.subplots()

    ax.scatter(lons, lats)

    for lon, lat, label in zip(lons, lats, labels):
        ax.text(lon, lat, label, fontsize=9, ha='right')

    ax.set_title(title)
    ax.set_xlabel("Delta meters lon direction")
    ax.set_ylabel("Delta meters lat direction")
    ax.grid(True)

    return fig




def add_to_json(day, bline, fits, path_to_json):
    #this loop is smth I looked up to make sure the json exists and is in the form we want
    if os.path.exists(path_to_json):
        with open(path_to_json, 'r') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, dict):
                    raise ValueError("not a dict")
            except json.JSONDecodeError:
                all_data = {}
    else:
        all_data = {}

    
    if day not in all_data:
        all_data[day] = {}

    all_data[day][f'bline{bline}'] = fits

    with open(path_to_json, 'w') as f:
        json.dump(all_data, f, indent=4)



