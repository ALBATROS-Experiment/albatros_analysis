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
from scipy.optimize import least_squares
import json
import random




def phase_pred(fit_coords, pulse_idx, info_list, context_list):
    
    ''' 
    Function that returns the predicted phase with time of a satellite pass.

    Inputs:

    fit_coords: coordinates of the non-reference antenna. this is the fitting parameter
    pulse_idx: the index (with respect to the list info_list) of which pulse we want to predcit the phase of
    info_list: list with all pulse data, for each pulse we consider. this list is different for each baseline
    context_list: list of relevant data we get from the config file. examples include the spectrum period and the accumulation length (aka integration time)
    
    Outputs:

    pred:   list of total phase of the pulse, with one entry for each chunk that goes by. length of array depends on the duration of the pulse, in units of chunks.
            these are in units of radians. 
    '''
    #unpack from info list 
    relative_start_time = info_list[pulse_idx][0]
    relative_end_time = info_list[pulse_idx][1]
    sat_ID = info_list[pulse_idx][2]
    pulse_channel_idx = info_list[pulse_idx][3]
    global_start_time = info_list[pulse_idx][4]
    tle_path = info_list[pulse_idx][5]

    #unpack from context list
    visibility_window = context_list[0]
    T_SPECTRA = context_list[1]
    v_acclen = context_list[2]
    v_nchunks = context_list[3]
    ref_coords = context_list[4]

    pulse_duration_sec = relative_end_time - relative_start_time
    time_start = global_start_time + relative_start_time

    pulse_duration_chunks = int( pulse_duration_sec / (T_SPECTRA * v_acclen) )
    pulse_freq = outils.chan2freq(pulse_channel_idx, alias=True)

    # 'd' has one entry per second
    
    d = outils.get_sat_delay(ref_coords, fit_coords, tle_path, time_start, (2*visibility_window)+1, sat_ID)
    # 'delay' has one entry per chunk (~0.5s) 
    delay = np.interp(np.arange(0, v_nchunks) * v_acclen * T_SPECTRA, np.arange(0, int(2*visibility_window)+1), d)
    #thus 'pred' has one entry for each chunk
    pred = (-delay[:pulse_duration_chunks]+ delay[0]) * 2 * np.pi * pulse_freq

    return pred





def residuals_all(coords, observed_data, phase_pred, info_list, context_list):
    ''' 
    Get all residuals (for all pulses in the info_list) in one long array.

    Inputs:
    
    coords: physical coordinates of the non-reference antenna
    observed_data: list of (list of unwrapped phase data for a pulse) for each pulse in info_list
    phase_pred: function that predicts the unwrapped phase depending on the non-ref antenna position
    info_list : list with all pulse data, for each pulse we consider. this list is different for each baseline
    context_list: list of relevant data we get from the config file. examples include the spectrum period and the accumulation length (aka integration time)


    Outputs:

    massive array of all residuals for each pulse, all concatenated, in order
    '''
    residuals_all = []
    for pulse_idx, observed in enumerate(observed_data):
        
        predicted = phase_pred(coords, pulse_idx, info_list, context_list)  
        res = observed - predicted
        residuals_all.append(res.flatten())

    return np.concatenate(residuals_all)


def fitting_all(observed_data, initial_coordinates, phase_pred, info_list, context_list, method='trf'):
    ''' 
    Calls least squares to optimize antenna coordinates for every pulse

    Inputs:

    observed_data: list of (list of unwrapped phase data for a pulse) for each pulse in info_list
    initial_coordinates: the initial guess of where the non-reference antenna is located
    phase_pred: function that predicts the unwrapped phase depending on the non-ref antenna position
    info_list : list with all pulse data, for each pulse we consider. this list is different for each baseline
    context_list: list of relevant data we get from the config file. examples include the spectrum period and the accumulation length (aka integration time)


    Outputs:

    optimized_coordinates:  the fitted coordinates of the non-ref antenna
    '''

    result = least_squares(
        lambda coords: residuals_all(coords, observed_data, phase_pred, info_list, context_list),  # Pass a lambda that calls residuals
        initial_coordinates,
        method = method
    )
    optimized_coordinates = result.x
    return optimized_coordinates, result


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




def residuals_individual(coords, observed_data, phase_pred, pulse_idx, info_list, context_list):
    ''' 
    Get residuals of only one specific pulse
    '''
    predicted = phase_pred(coords, pulse_idx, info_list, context_list)
    res = observed_data[pulse_idx] - predicted
    return res


def fitting_individual(observed_data, initial_coordinates, phase_pred, pulse_idx, info_list, context_list, method = 'trf'):
    ''' 
    Calls least squares to optimize coordinates for one pulse only
    '''
    result = least_squares(
        lambda coords: residuals_individual(coords, observed_data, phase_pred, pulse_idx, info_list, context_list), 
        initial_coordinates,
        method = method
    )
    optimized_coordinates = result.x
    return optimized_coordinates, result


def distance_calculator(coord1, coord2):
    ''' 
    Returns the actual physical distance between two coordinates. 
    Seperates the superficial component (latitude and longitude) and the altitude component into two seperate measurements
    '''

    lat1, lon1, alt1 = coord1[0], coord1[1], coord1[2]
    lat2, lon2, alt2 = coord2[0], coord2[1], coord2[2]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    meters_flat = 6367 * c *1000
    meters_alt = np.abs(alt1-alt2)

    return float(meters_flat), float(meters_alt)


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
    row_count,rownums0,rownums1,rowidx=get_common_rows(specnum0,specnum1,idxstart0,idxstart1)
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





#some experimental shit

def fitting_all_with_offsets(observed_data, initial_coords, phase_pred, info_list, context_list, method='trf'):
    n_pulses = len(info_list)
    x0 = np.concatenate([initial_coords, np.zeros(n_pulses)])  # coords + N offsets

    result = least_squares(
        lambda x: residuals_with_offsets(x[:3], x[3:], observed_data, phase_pred, info_list, context_list),
        x0,
        method=method
    )
    return result.x[:3], result


def residuals_with_offsets(coords, offsets, observed_data, phase_pred, info_list, context_list):
    residuals = []
    for i, obs in enumerate(observed_data):
        pred = phase_pred(coords, i, info_list, context_list)
        phase_res = obs - (pred + offsets[i])
        residuals.append(phase_res)
    return np.concatenate(residuals)




