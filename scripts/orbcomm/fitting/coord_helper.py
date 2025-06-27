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


#--------------------------------PHASE PREDICTORS------------------------


def phase_pred(fit_coords, pulse_idx, data_list, context_list, satdict = None):

    '''
    gives a predicted phase, with no time offsets
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

    #--------------------------------------------------------------------------


    #basic setup
    pulse_duration_sec = relative_end_time - relative_start_time
    time_start = global_start_time + relative_start_time

    #give this a buffer to ensure no problems with observed_length
    pulse_duration_chunks = np.ceil(pulse_duration_sec / (T_SPECTRA * v_acclen)) + 5
    pulse_freq = outils.chan2freq(pulse_channel_idx, alias=True)

    # 'd' has one entry per second
    
    


    d = outils.get_sat_delay_new(ref_coords, fit_coords, tle_path, time_start, visibility_window+1, sat_ID)

    interpolation_chunk_times = np.arange(0, pulse_duration_chunks) * v_acclen * T_SPECTRA
    
    # 'delay' has one entry per chunk (~0.5s) 
    delay = np.interp(interpolation_chunk_times, np.arange(0, visibility_window+1), d)
    #thus 'pred' has one entry for each chunk
    pred = (-delay + delay[0]) * 2 * np.pi * pulse_freq

    print("time taken for one prediction", time.time() - start_time)

    return pred




def pred(fit_coords, ds, pulse_idx, data_list, context_list, satdict = None):

    '''
    gives a predicted phase, with time offsets thrown in for each pulse.
    recall that dc is in fact a fit parameter
    '''

    extension = 2
    
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
    pred = (-delay + delay[0]) * 2 * np.pi * pulse_freq

    print("time taken for one prediction", time.time() - start_time)

    return pred












#--------------------------------------------------RESIDUALS------------------------------

def res_ind(coords, phase_pred, pulse_idx, data_list, context_list):
    ''' 
    Get residuals of only one specific pulse
    '''
    predicted = phase_pred(coords, 0, pulse_idx, data_list, context_list)
    res = data_list[pulse_idx][4] - predicted
    return res



def res_ds(fit_coords, pred, ds, pulse_idx, data_list, context_list):

    predicted = pred(fit_coords, ds, pulse_idx, data_list, context_list)[:len(data_list[pulse_idx][4])]
    res = data_list[pulse_idx][4] - predicted

    return res



def res_all(coords, phase_pred, data_list, context_list, satdict = None):

    residuals_all = []
    for pulse_idx, data in enumerate(data_list):

        if satdict == None:
            predicted = phase_pred(coords, 0, pulse_idx, data_list, context_list)[:len(data[4])]  
        else:
            predicted = phase_pred(coords, pulse_idx, data_list, context_list, satdict = satdict)[:len(data[4])]
            
        res = data[4] - predicted
        residuals_all.append(res.flatten())
    return np.concatenate(residuals_all)


def res_ds_all(fit_coords, ds, phase_pred_ds, data_list, context_list):

    residuals_all = []
    for p_idx, data in enumerate(data_list):

        if isinstance(ds, numbers.Number):
            predicted = pred(fit_coords, ds, p_idx, data_list, context_list)[:len(data[4])]
        else:
            predicted = pred(fit_coords, ds[p_idx], p_idx, data_list, context_list)[:len(data[4])]
        res = data[4] - predicted
        
        residuals_all.append(res.flatten())

    return np.concatenate(residuals_all)




def res_cov(coords, pred, data_list, context_list):

    residuals_all = []
    R_all = []
    for pulse_idx, data in enumerate(data_list):
        predicted = pred(coords, 0, pulse_idx, data_list, context_list)[:len(data[4])]  
  
        res = data[4] - predicted
        residuals_all.append(res.flatten())

        var = np.var(res, ddof=1)
        R_ind = np.eye(len(res)) * var
        R_all.append(R_ind)
        
    R = block_diag(*R_all)
    
    return np.concatenate(residuals_all), R



def weighted_res(coords, pred, data_list, context_list):
    coords = np.array(coords)
    residuals = []

    for pulse_idx, data in enumerate(data_list):
        predicted = pred(coords, 0, pulse_idx, data_list, context_list)[:len(data[4])]
        res = data[4] - predicted
        std = np.std(res, ddof=1)  # standard deviation for whitening
        res_weighted = res / std
        residuals.append(res_weighted.flatten())

    return np.concatenate(residuals)









#---------------------------------------FITTING--------------------------------------


def fitting_individual(initial_coordinates, pred, pulse_idx, data_list, context_list, method = 'trf'):
    ''' 
    Calls least squares to optimize coordinates for one pulse only
    '''
    result = least_squares(
        lambda coords: residuals_individual(coords, observed_data, pred, pulse_idx, data_list, context_list), 
        initial_coordinates,
        method = method
    )
    optimized_coordinates = result.x
    return optimized_coordinates, result



def fit_ds(coords, pred, pulse_idx, data_list, context_list, method='trf'):


    result = least_squares(
        lambda ds: res_ds(coords, pred, ds, pulse_idx, data_list, context_list), 
        0,
        bounds=(-2.0, 2.0),
        method = method
    )
    ds_fit = result.x
    return ds_fit, result



def fit_all(initial_coordinates, pred, data_list, context_list, method='trf', tle_path_list = None):

    if tle_path_list == None:
        
        result = least_squares(
            lambda coords: res_all(coords, pred, data_list, context_list),
            initial_coordinates,
            method = method
        )

    else:
        satdict = {}
        for tle_path in tle_path_list:
            satdict[tle_path] = sf.load.tle_file(tle_path)

        result = least_squares(
            lambda coords: res_all(coords, pred, data_list, context_list, satdict = satdict),
            initial_coordinates,
            method = method
        )


    optimized_coordinates = result.x
    return optimized_coordinates, result



def fit_ds_all(initial_coords, pred, data_list, context_list, method='trf'):

    ds_list = []
    for pulse_idx, data in enumerate(data_list):
        result = least_squares(
            lambda ds: res_ds(initial_coords, pred, ds, pulse_idx, data_list, context_list), 
            0,
            bounds=(-2.0, 2.0),
            method = method
        )
        ds_list.append(result.x)
    return ds_list


def fit_4p(initial_coords, pred, data_list, context_list, method='trf'):

    n_pulses = len(data_list)
    x0 = np.concatenate([initial_coords, np.zeros(n_pulses)])  # coords + N time offsets


    #clever bounding thing that I cannot claim responsibility for
    coord_bounds = ([-np.inf]*3, [np.inf]*3)
    ds_bounds = ([-1.0]*n_pulses, [1.0]*n_pulses)

    lower = coord_bounds[0] + ds_bounds[0]
    upper = coord_bounds[1] + ds_bounds[1]

    
    result = least_squares(
        lambda x: res_ds_all(x[:3], x[3:], pred, data_list, context_list),
        x0,
        bounds = (lower, upper),
        method=method
    )

    return result.x[:3], result


def fit_4p_fixed(initial_coords, pred, data_list, context_list, method='trf'):

    x0 = np.zeros(4)
    x0[:3] = initial_coords  # coords + single clock offset

    bounds = (
        [-np.inf, -np.inf, -np.inf, -5.0],  # lower
        [ np.inf,  np.inf,  np.inf,  5.0],  # upper
    )

    result = least_squares(
        lambda x: res_ds_all(x[:3], x[3], pred, data_list, context_list),
        x0,
        bounds = bounds,
        method=method
    )

    return result.x, result


def fit_given_offsets(initial_coords, pred, ds_list, data_list, context_list, method='trf'):

    result = least_squares(
            lambda coords: res_ds_all(coords, ds_list, pred, data_list, context_list),
            initial_coords,
            method = method
        )

    
    optimized_coordinates = result.x
    return optimized_coordinates, result





def fit_cov(initial_coordinates, pred, data_list, context_list, method='trf'):

    initial_coordinates = np.array(initial_coordinates)



    result = least_squares(
        fun = whitened_res,
        x0 = initial_coordinates,
        args=(pred, data_list, context_list)
        )

    optimized_coordinates = result.x

    final_residuals, final_cov_matrix = res_cov(optimized_coordinates, pred, data_list, context_list)

    return optimized_coordinates, final_cov_matrix













#-------------------------------INTERPOLATION FOR 1BIT--------------------------


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




#---------auxilary stuff---------


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




def make_fuzzed_coords(initial_guess, meters=10, reps=5):

    ''' 
    Generates a custom number of random coordinates, where each component is within a certain number of meters of the initial guess.
    This is used for the coordinate fuzz test.
    '''
    
    lat, lon, alt = initial_guess[0], initial_guess[1], initial_guess[2]
    random_coords = []

    for _ in range(reps):
        # Convert meters to degrees
        d_lat = (random.uniform(-meters, meters)) / 111320
        d_lon = (random.uniform(-meters, meters)) / (111320 * np.cos(np.radians(lat)))
        d_alt = random.uniform(-meters, meters)

        new_lat = float(lat + d_lat)
        new_lon = float(lon + d_lon)
        new_alt = float(alt + d_alt)

        random_coords.append([new_lat, new_lon, new_alt])

    return random_coords



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





#--------visibility stuff-------------

@nb.njit()
def get_common_rows(specnum0,specnum1,idxstart0,idxstart1):
    nrows0,nrows1=specnum0.shape[0],specnum1.shape[0]
    maxrows=min(nrows0,nrows1)
    rownums0=np.empty(maxrows,dtype='int64')
    rownums0[:]=-1
    rownums1=rownums0.copy()
    rowidx=rownums0.copy()
    i=0;j=0;row_count=0;
    while i<nrows0 and j<nrows1:
        if (specnum0[i]-idxstart0)==(specnum1[j]-idxstart1):
            rownums0[row_count]=i
            rownums1[row_count]=j
            rowidx[row_count]=specnum0[i]-idxstart0
            i+=1
            j+=1
            row_count+=1
        elif (specnum0[i]-idxstart0)>(specnum1[j]-idxstart1):
            j+=1
        else:
            i+=1
    return row_count,rownums0,rownums1,rowidx

@nb.njit(parallel=True)
def avg_xcorr_4bit_2ant_float(pol0,pol1,specnum0,specnum1,idxstart0,idxstart1,delay=None,freqs=None):
    row_count,rownums0,rownums1,rowidx = get_common_rows(specnum0,specnum1,idxstart0,idxstart1)
    ncols=pol0.shape[1]
#     print("ncols",ncols)
    assert pol0.shape[1]==pol1.shape[1]
    xcorr=np.zeros((row_count,ncols),dtype='complex64') # in the dev_gen_phases branch
    if delay is not None:
        for i in nb.prange(row_count):
            for j in range(ncols):
                xcorr[i,j] = pol0[rownums0[i],j]*np.conj(pol1[rownums1[i],j]*np.exp(2j*np.pi*delay[rowidx[i]]*freqs[j]))
    else:
        for i in nb.prange(row_count):
            xcorr[i,:] = pol0[rownums0[i],:]*np.conj(pol1[rownums1[i],:])
    return xcorr












#---------------------------OLD AND MISC---------------------

def fitting_latlon_only(observed_data, initial_coordinates, phase_pred, info_list, context_list, method='trf'):
    """
    Fit only latitude and longitude, keeping altitude fixed.
    """
    fixed_alt = initial_coordinates[2]
    print(initial_coordinates[:2])

    def latlon_residuals(latlon):
        coords = [latlon[0], latlon[1], fixed_alt]
        return residuals_all(coords, observed_data, phase_pred, info_list, context_list)

    result = least_squares(
        latlon_residuals,
        x0=initial_coordinates[:2],  # Only lat and lon
        method=method
    )

    # Reconstruct full coordinate with fixed altitude
    optimized_coordinates = [result.x[0], result.x[1], fixed_alt]
    return optimized_coordinates, result




def get_std_meters(fit, predictor, data_list, context_list, satdict = None):

    meta = fit[1]

    jacobian = meta.jac  

    res, R = res_cov(fit[0], phase_pred, data_list, context_list, satdict = satdict)
    
    cov_matrix = inv(jacobian.T @ R @ jacobian)

    param_errors = np.sqrt(np.diag(cov_matrix))

    print("Parameter uncertainties:", param_errors) 



def res_offsets(coords, offsets, phase_pred, data_list, context_list, satdict = None):
    residuals = []
    for i, obs in enumerate(data_list):
        if satdict == None:
            pred = phase_pred(coords, i, data_list, context_list, satdict=satdict)
        else:
            pred = phase_pred(coords, i, data_list, context_list, satdict=satdict)

        phase_res = obs[4] - (pred + offsets[i])
        residuals.append(phase_res)
    return np.concatenate(residuals)

def fit_offsets(initial_coords, phase_pred, data_list, context_list, method='trf', tle_path_list = None):

    n_pulses = len(data_list)
    x0 = np.concatenate([initial_coords, np.zeros(n_pulses)])  # coords + N offsets

    if tle_path_list == None:
        result = least_squares(
            lambda x: residuals_with_offsets(x[:3], x[3:], phase_pred, data_list, context_list),
            x0,
            method=method
        )

    else:
        satdict = {}
        for tle_path in tle_path_list:
            satdict[tle_path] = sf.load.tle_file(tle_path)

        result = least_squares(
            lambda x: residuals_with_offsets(x[:3], x[3:], phase_pred, data_list, context_list, satdict=satdict),
            x0,
            method=method
        )

    return result.x[:3], result