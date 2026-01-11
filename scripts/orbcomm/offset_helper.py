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
    print('perfoming beamformed cxcorr')
    if len(data.shape) == 4:
        print('assuming shape (nants, npols, nspec, nchans)')
        nants, npols, nspec, nchans = data.shape
        assert len(freqs) == nchans
        assert len(delay) == nspec
        spec1=data[ant1_idx,pol1_idx,:,:].copy()
        spec2=data[ant2_idx,pol2_idx,:,:].copy()
        spec2_phased = np.empty_like(spec2)
        spec2_phased = apply_delay(spec2, spec2_phased, -delay, freqs)
        V = xcorr_avg(spec1,spec2_phased,acclen)
    else:
        print('assmuming shape (nants, npols, nspec)')
        nants, npols, nspec = data.shape
        assert len(delay) == nspec
        spec1=data[ant1_idx,pol1_idx,:].copy()
        spec2=data[ant2_idx,pol2_idx,:].copy()
        spec2_phased = np.empty_like(spec2)
        spec2_phased = apply_delay_1d(spec2, spec2_phased, -delay, freqs)
        V = xcorr_avg_1d(spec1,spec2_phased,acclen)
    return V


def efield_to_vis(
    data_all,
    t_start,
    t_end,
    ant1_idx,
    ant2_idx,
    pol1_idx,
    pol2_idx,
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
    ant1_coords = ant_coords[ant1_idx]
    ant2_coords = ant_coords[ant2_idx]
    T_SPECTRA = bb_spectrum_T * osamp
    tle_path = outils.get_tle_file(t_start, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    sats_objects = load.tle_file(tle_path)

    dly = outils.get_sat_delay2(
                        ant1_coords,
                        ant2_coords,
                        sats_objects,
                        t_start,
                        int(t_end-t_start)+1,
                        satID,
                        altaz=False
                    )

    delay = np.interp(np.arange(0, data_all.shape[2]) * T_SPECTRA, np.arange(0, int(t_end-t_start)+1), dly)


    Vxx = beamformed_xcorr(data_all, ant1_idx, ant2_idx, 0, 0, delay, sat_freqs, acclen)
    Vyy = beamformed_xcorr(data_all, ant1_idx, ant2_idx, 1, 1, delay, sat_freqs, acclen)
    V=(Vxx+Vyy)/2
    return V.T



def objective_times(time_offset,
                    t_start,
                    t_end,
                    data_slice,
                    ant_idxs,
                    ant_coords,
                    freq,
                    satID,
                    plot=False,
                    bb_spectrum_T=4096/250e6,
                    osamp=64,
                    acclen = 1024):

    """ 
    t_start needs to coincide with the first spectrum of the data.
    however, t_end will determine how much the data is cut by, so as long as it's
    BEFORE the last spectrum in the data, we're all good.

    this is done to make cutting garbage time at the end easier, and minimize interference with the data array
    """

    process_start_ts = time.time()
    #turn time_offset into float when passed in scipy.optimize.minimize
    if type(time_offset) == np.ndarray:
        time_offset = time_offset[0]

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.axhline(6.28, ls='--',c='black')
        ax.axhline(-6.28,ls='--', c='black')
        plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
            "legend.fontsize": 14
        })

    T_SPECTRA = bb_spectrum_T * osamp
    tle_path = outils.get_tle_file(t_start, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    sats_objects = load.tle_file(tle_path)

    nspec_tot = data_slice.shape[2]
    nspec = int((t_end-t_start)/T_SPECTRA)
    print('nspec total', nspec_tot)
    print('nspec', nspec)
    data_slice = data_slice[:, :, :nspec].copy()

    assert nspec_tot > nspec
    assert (nspec*bb_spectrum_T) < (t_end-t_start+1)

    chisq = 0
    sum_wt = 0
    for i in range(len(ant_idxs)):
        for j in range(i+1, len(ant_idxs)):
            ai = ant_idxs[i]
            aj = ant_idxs[j]
            a1_coords=ant_coords[ai]
            a2_coords=ant_coords[aj]
            dist = haversine(a1_coords,a2_coords)
            sum_wt += dist**2


            dly = outils.get_sat_delay2(
                                a1_coords,
                                a2_coords,
                                sats_objects,
                                t_start+time_offset,
                                int(t_end-t_start)+1,
                                satID,
                                altaz=False
                            )
            delay = np.interp(
                np.arange(0, nspec) * T_SPECTRA, np.arange(0, int(t_end-t_start)+1), dly
            )
            spec2_phased = np.empty_like(data_slice[aj,0,:])
            spec2_phased = apply_delay_1d(data_slice[aj,0,:], spec2_phased, -delay, freq)
            Vxx = xcorr_avg_1d(data_slice[ai,0,:],spec2_phased,acclen)
            spec2_phased = apply_delay_1d(data_slice[aj,1,:], spec2_phased, -delay, freq)
            Vyy = xcorr_avg_1d(data_slice[ai,1,:],spec2_phased,acclen)
            V=(Vxx+Vyy)/2


            if plot:
                ax.plot(np.unwrap(np.angle(V))-np.angle(V)[0],label=f'{ai}-{aj}')
                plt.legend()

            
            Vnew = np.exp(1j*np.angle(V))
            chisq-= dist**2*np.abs(np.mean(Vnew))**2 
    chisq/=sum_wt
    print('time for objective compute:', time.time()-process_start_ts)
    print("offset", time_offset, "chisq", chisq)

    if plot:
        ax.set_xlabel(f"Visibility Chunk(~{np.round(T_SPECTRA*acclen, decimals=2)} s)")
        ax.set_ylabel("Unwrapped Phase (radians)")
        ax.set_title(f"Phase at Baslines (offset={time_offset}, sat={satID})")
        ax.grid(True, alpha=0.3)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )
        plt.tight_layout()
        return chisq, fig
    
    return chisq


def cost_curve(
    offset_start,
    offset_end,
    ntimes,
    batch_start_ts,
    batch_end_ts,
    data_slice,
    ant_idxs,
    ant_coords,
    freq,
    satID,
    bb_spectrum_T=4096/250e6,
    osamp=64,
    include_fig=False,
    vline = None,
):
    """ 
    Generates cost curve of single pulse given time offset intervals and steps.
    """

    offsets = np.linspace(offset_start, offset_end, ntimes)
    chisqs = np.zeros(len(offsets),dtype='float64')
    for i,offset in enumerate(offsets):
        print(f'\n----ITERATION {i}-----')
        chisqs[i]=objective_times(offset,
                            batch_start_ts,
                            batch_end_ts,
                            data_slice,
                            ant_idxs,
                            ant_coords,
                            freq,
                            satID,
                            plot=False,
                            bb_spectrum_T=bb_spectrum_T,
                            osamp=osamp)
    


    if include_fig:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
            "legend.fontsize": 14
        })
        ax.plot(offsets,chisqs)
        ax.set_xlabel('Time Discrepancy (s)')
        ax.set_ylabel(r'$\chi ^2$')
        if vline:
            plt.axvline(x=vline, color='r', linestyle='--')
        return offsets, chisqs, fig

    else:
        return offsets, chisqs


