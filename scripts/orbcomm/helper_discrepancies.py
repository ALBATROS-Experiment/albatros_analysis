import os
import sys
sys.path.append(os.path.expanduser('~'))
import numpy as np 
from matplotlib import pyplot as plt
from datetime import datetime as dt
import figures as fgs
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
import numba as nb
import time
import importlib
import json
from scipy.optimize import minimize
from scipy.ndimage import binary_closing
from skyfield.api import load, wgs84
import cupy
#from pyuvdata import UVData

def get_start_specnum(t_start, dir_parent):
    files, idx = butils.get_init_info(t_start, t_start+100, dir_parent)
    pulse = bdc.BasebandFileIterator(
            files,
            0,
            idx,
            1024,
            None,
            chanstart=1834,
            chanend=1852,
            type="float",
        )
    start_specnum = pulse.spec_num_start
    print('start specnum', start_specnum)
    return start_specnum

def get_prediction_error(res, spectra):
    '''Get the error on the predicted UTC discrepancy given residual matrix'''
    A = np.column_stack((spectra, np.ones(len(spectra))))

    N = np.cov(res)
    print(N)
    #N2 = np.var(res)
    #print(N2)

    if res.ndim >1:
        v1 = np.linalg.inv(A.T@np.linalg.inv(N)@A)

    else:
        v1 = N*np.linalg.inv(A.T@A)

    return A@v1@A.T


def correct_overflow(spectra_uncorr):
    '''Corrects a spectrum number overflow to allow for UTC map fitting over whole batch'''
    overflow, spectra_corr = False, []
    for i in range(len(spectra_uncorr)):
        if i > 0:
            if abs(spectra_uncorr[i] - spectra_uncorr[i-1]) > 2**31:
                overflow = True
        if overflow:
            spectra_corr.append(spectra_uncorr[i] + 2**32)
        else:
            spectra_corr.append(spectra_uncorr[i])
    spectra_corr = np.array(spectra_corr)
    return spectra_corr

def get_MAD(data, axis=None):
    '''getting the real median absolute deviation'''
    data_median = np.median(data, axis=axis, keepdims=True)
    abs_deviations = np.abs(data - data_median)
    mad = np.median(abs_deviations, axis=axis)
    return mad



@nb.njit(parallel=True)
def apply_delay(arr, out, delay, freqs):
    # apply delay to an array of complex electric field or their correlation
    # does exp( j 2 pi nu tau) sign of tau is user dependent
    # freqs should correspond to the columns of the nspec x nchan array
    nspec = arr.shape[0]
    nchan = arr.shape[1]
    for i in nb.prange(nspec):
        for j in range(nchan):
            out[i, j] = arr[i, j] * np.exp(2j * np.pi * freqs[j] * delay[i])
    return out

@nb.njit(parallel=True)
def apply_delay_1d(arr, out, delay, freq):
    # apply delay to an array of complex electric field or their correlation
    # does exp( j 2 pi nu tau) sign of tau is user dependent
    # freqs should correspond to the columns of the nspec x nchan array
    nspec = arr.shape[0]
    for i in nb.prange(nspec):
            out[i] = arr[i] * np.exp(2j * np.pi * freq * delay[i])
    return out

@nb.njit(parallel=True)
def xcorr_avg(arr1,arr2,acclen):
    #helper function
    nblocks = arr1.shape[0]//acclen
    nchan = arr1.shape[1]
    out = np.zeros((nblocks,nchan),dtype=arr1.dtype)
    for i in nb.prange(nblocks):
        for j in range(acclen):
            for k in range(nchan):
                out[i,k] += arr1[i*acclen + j,k]*np.conj(arr2[i*acclen + j,k])
        out[i,:]/=acclen
    return out

@nb.njit(parallel=True)
def xcorr_avg_1d(arr1,arr2,acclen):
    #helper function
    nblocks = arr1.shape[0]//acclen
    out = np.zeros((nblocks,),dtype=arr1.dtype)
    for i in nb.prange(nblocks):
        for j in range(acclen):
                out[i] += arr1[i*acclen + j]*np.conj(arr2[i*acclen + j])
        out[i]/=acclen
    return out
    

def haversine(p1, p2, radius=6371000):
    """
    Vectorized Haversine distance using NumPy.

    Parameters
    ----------
    p1 : array-like of shape (..., 2)
        lat, lon in degrees
    p2 : array-like of shape (..., 2)
        lat, lon in degrees
    radius : float
        Earth radius (km by default)

    Returns
    -------
    distances : ndarray
        Distance(s) in the same unit as `radius`.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    lat1 = np.radians(p1[..., 0])
    lon1 = np.radians(p1[..., 1])
    lat2 = np.radians(p2[..., 0])
    lon2 = np.radians(p2[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius * c


def beamformed_xcorr(data, ant1_idx, ant2_idx, pol1_idx, pol2_idx, delay, freqs, acclen):
    nants, npols, nspec, nchans = data.shape
    assert len(freqs) == nchans
    assert len(delay) == nspec
    spec1=data[ant1_idx,pol1_idx,:,:].copy()
    spec2=data[ant2_idx,pol2_idx,:,:].copy()
    spec2_phased = np.empty_like(spec2)
    spec2_phased = apply_delay(spec2, spec2_phased, -delay, freqs)
    V = xcorr_avg(spec1,spec2_phased,acclen)
    return V


def efield_to_vis(
    data_all,
    t_start,
    t_end,
    ant_idxs,
    ant_coords,
    satID,
    sat_freqs,
    acclen = 1024,
    osamp = 64,
    bb_spectrum_T = 4096/250e6
):

    """ 
    Notes: require t_start to coincide with the first spectrum of data_all.
    As long as t_end is after the last spectrum, the interpolation is fine.
    """
    T_SPECTRA = bb_spectrum_T * osamp
    nspec = int((t_end - t_start)/T_SPECTRA)
    nants = len(ant_idxs)
    print(nspec)
    data_all = data_all[:, :, :nspec, :]
    print(data_all.shape)
    tle_path = outils.get_tle_file(t_start, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    sats_objects = load.tle_file(tle_path)
    V = []
    arr_spectra = np.arange(0, nspec) * T_SPECTRA
    arr_times = np.arange(0, int(t_end-t_start)+1)
    for i in range(nants):
        ant1_idx = ant_idxs[i]
        ant1_coords = ant_coords[ant1_idx]
        for j in range(i+1, nants):
            ant2_idx = ant_idxs[j]
            ant2_coords = ant_coords[ant2_idx]

            dly = outils.get_sat_delay2(
                                ant1_coords,
                                ant2_coords,
                                sats_objects,
                                t_start,
                                int(t_end-t_start)+1,
                                satID,
                                altaz=False
                        )
            #print(dly.shape)

            delay = np.interp(arr_spectra, arr_times, dly)
            #print(delay.shape)

            Vxx = beamformed_xcorr(data_all, ant1_idx, ant2_idx, 0, 0, delay, sat_freqs, acclen)
            Vyy = beamformed_xcorr(data_all, ant1_idx, ant2_idx, 1, 1, delay, sat_freqs, acclen)
            vis=(Vxx+Vyy)/2
            V.append(np.abs(vis))
    return np.array(V)


def find_signal_channels(data, sat_type):
    '''
    Cuts channels by taking median power in time. Isolates signal channels for later.
    '''
    assert sat_type in {"METEOR","NOAA"}
    if sat_type == "METEOR":
        nchan_signal = 100
    if sat_type == "NOAA":
        nchan_signal = 20
    nchan = data.shape[-1]

    power = np.abs(data)
    chan_power = np.mean(np.median(power, axis=1), axis=(0))

    cumsum = np.concatenate(([0], np.cumsum(chan_power)))
    window_sums = cumsum[nchan_signal:] - cumsum[:nchan - nchan_signal + 1]
    best_start = int(np.argmax(window_sums))
 
    return slice(best_start, best_start + nchan_signal)


def amplitude_cut(data, chan_slice, win_size=120):
    '''
    Selects the contiguous window (length = win_size samples) with the highest median SNR.
    '''
    # --- signal power ---
    data_signal = data[:, :, chan_slice]
    power = np.median(np.abs(data_signal)**2, axis=2)

    # --- noise power ---
    off_mask = np.ones(data.shape[-1], dtype=bool)
    off_mask[chan_slice] = False
    off   = data[:, :, off_mask]
    noise = np.median(np.median(np.abs(off)**2, axis=2), axis=1, keepdims=True)

    # --- SNR ---
    snr = power / (noise + 1e-30)

    # collapse pols → robust time series
    snr_all = np.median(snr, axis=0)

    # --- sliding window median ---
    if win_size > len(snr_all):
        return (0, len(snr_all))

    windows = np.lib.stride_tricks.sliding_window_view(snr_all, win_size)
    medians = np.median(windows, axis=1)

    # --- best window ---
    start = int(np.argmax(medians))
    end = start + win_size

    return start, end


def discrep_cutting(V, satID, acclen=1024):
    '''
    Full function that goes from beamformed visibilities to final amplitude cut
    '''
    assert satID in {28654,25338,33591,57166,59051}
    if satID in {59051, 57166}:
        sat_type = "METEOR"
    if satID in {28654,25338,33591}:
        sat_type = "NOAA"

    print('Satellite Type we see is:', sat_type)
    chan_slice = find_signal_channels(V, sat_type)
    start_chunk, end_chunk = amplitude_cut(V, chan_slice)

    new_chans = [chan_slice.start,chan_slice.stop]
    spectra_start = start_chunk*acclen
    spectra_end = end_chunk*acclen

    return spectra_start,spectra_end,new_chans