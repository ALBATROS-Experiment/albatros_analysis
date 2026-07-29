#system stuff
import os
import sys
sys.path.append(os.path.expanduser('~/albatros_analysis'))
#general
import numpy as np
import cupy as cp
import numba as nb
import time
from scipy import linalg
from scipy import stats
from scipy import signal as sn 
from matplotlib import pyplot as plt
from datetime import datetime as dt
#utils
from src.correlations import baseband_data_classes as bdc
from src.utils import baseband_utils as butils
from src.utils import orbcomm_utils as outils
from src.utils import orbcomm_utils_gpu as outils_g
from src.utils import sat_utils as sutils
#helpers and functions
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scripts.xcorr import helper as hp
from scripts.xcorr import helper_gpu as hpg


def get_complex_snr(data): 
    return cp.asnumpy(cp.max(cp.abs(data), axis=1) / outils_g.median_abs_deviation(cp.abs(data),axis=1))


def median_abs_deviation(x,axis=1):
    med = cp.median(x,axis=axis)
    return cp.median(cp.abs(x-med[:, cp.newaxis]),axis=axis)


def get_cxcorr_many_sats(p0_ref,
                        p0_nref, 
                        tle_path, 
                        times, 
                        sats_present,
                        satmap,
                        coords,
                        dN,
                        T_SPECTRA = 4096 / 250e6,
                        c_acclen = 10**6):
    
    ''' 
    get the coarse cross-correlation for a given chunk of data, for a given set of sats

    Parameters
    ----------
    p0_ref/p0_nref: cupy array of complex
        Data array of reference/non-reference antenna, shape (c_acclen, nchans).
        'Reference' means it is the second in spectrum number, and lower in index.
        By convention, we always write specnumoffset (and therefore delays) as (nref-ref).

    tle_path: string
        Directory path of TLE file containing satellite positions for pulse time we care about
        This allows us to beamform accurately.

    times: list
        List of [start_time, end_time] for the pulse, in unix time. Lets us know where to beamform

    sats_present: list
        List of length nsats which tells us which satellites to try and beamform on. 
        Sats are in index form, of legacy hard-coded satlist [28654,25338,33591,57166,59051,44387].

    satmap: dictionary
        Dictionary that maps each sat_idx to it satID, and vice versa. Lets us move between these 
        forms for utility and ease, depending on which is needed. To perhaps phase out later down the line

    coords: list
        List of the two sets of coordinates [ref_coords, nref_coords]. 
        Each coord is in form [lat, lon, alt]

    dN: int
        Gives how many spectra of shift (per direction) we actually include in our cxcorr data.
        So for example, if dN is 10e5, we will have 2*10e5 different attempted offsets 

    T_SPECTRA: float
        period of spectra

    c_acclen: int
        Number of spectra in each chunk. Integration time of the cxcorr.



    Returns
    -------
    cx: list of arrays
        A list of the coarse cross-correlations for each sat in sats_present,
        and also for no beamforming. Uncorrected always given first.
        ['Uncorrected', sat1, sat2]
        Each coarse xcorr has shape (nchans, 2*dN)
    
    '''

    nchans = len(p0_ref[0,:])
    freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)
    cx = []
    coords_ref, coords_nref = coords[0], coords[1]
    p0_nref_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
    
    pulse_start, pulse_end = times[0], times[1]
    niter = int(pulse_end - pulse_start) + 1  #+1 to avoid edge effects

    #GET GEO DELAY
    delays = np.zeros((c_acclen, len(sats_present)))
    for i, satidx in enumerate(sats_present):
        d = outils.get_sat_delay(  #get delay for whole pulse even though we only interpolate over one chunk.
            coords_ref, coords_nref, tle_path, pulse_start, niter, satmap[satidx]
            )
        delays[:, i] = np.interp(
            np.arange(0, c_acclen) * T_SPECTRA, np.arange(0, niter), d
        )
    delays = cp.asarray(delays)

    #print('delays shape', delays.shape)
    #print('DELAYS', delays)
    
    #UNCORRECTED
    cx.append(outils_g.coarse_xcorr(p0_ref, p0_nref, dN))  # no correction

    #CORRECTED
    for i, satidx in enumerate(sats_present):
        print("getting cxcorr of:", satmap[satidx])
        outils_g.apply_delay(p0_nref, delays[:,i], freqs, out=p0_nref_delayed)
        cx.append(outils_g.coarse_xcorr(p0_ref, p0_nref_delayed, dN))

    return cx



def get_vis_gpu(pulse_start_t,
                pulse_end_t,
                paths,
                offsets,
                T_SPECTRA = 4096/250e6,
                v_acclen = 30000):
    ''' 
    computes visibilities for one baseline for a set period, given a specnumoffset

    note that this is just a regular CPU visibility computation, mainly useful for sanity checks
    also note that this should be tested with two non-ref antenna (usually run with one ref one non ref)
    '''

    chunk_length = T_SPECTRA * v_acclen
    pulse_len_chunks = int(np.ceil((pulse_end_t - pulse_start_t)/chunk_length))
    spec_offset = offsets[1]-offsets[0]

    print('pulse_start', pulse_start_t)
    print('pulse_end', pulse_end_t)
    print('pulse duration', pulse_end_t-pulse_start_t)
    print('offsets', offsets) 
    print('paths', paths)

    files0, idx0, files1, idx1 = hp.get_init_info_2ant(pulse_start_t, pulse_end_t, spec_offset, paths[0], paths[1])
    files = [files0, files1]
    idxs = [idx0, idx1]

    channels = bdc.get_header(files[0][0])["channels"].astype('int64')
    chanstart = np.where(channels == 1834)[0][0] 
    chanend = np.where(channels == 1852)[0][0]
    print('starting, ending channels:', chanstart, chanend)
    chanlist = np.arange(1834, 1852)

    vis, channels = hpg.xcorr_avg(idxs, files, v_acclen, pulse_len_chunks, chanlist)
    
    return vis, channels




def get_chunk_data(files, idxs, chanstart, chanend, nchunks = None, c_acclen = 10**6):
    ''' 
    Get chunk data for two antenna given times and paths, using bfi.

    Purpose of wrapping in a function is to minimize cached memory usage 
    (which is very heavy for bfi files) for modules using large loops.

    Parameters
    ----------

    files: list of strings
        Paths for the two antenna, in form [ref_path, nref_path]

    idxs: list of integers
        List of starting indices within the first file for the timestream.
        In form [ref_idx, nref_idx]

    chanstart/chanend: int
        starting/ending channel index in baseband file index

    c_acclen: int
        length of chunk in spectra

    ref_bool: bool
        tells you if you have to switch the order of files and idx lists. 
        used when fitting for coordinates. If true, ref ant is fit ant, so all good.
        If False, then must switch, since convention is that we write 
        ref ant first, fitting ant first.
    
    Returns
    -------

    p0_ref_copy/p0_nref_copy: cupy array
        array shape (c_acclen, nchans), with one chunk of data for ref/nref antenna

    specnum_offset: int
        initial offset of absolute spectrum numbers between the data streams.
        i.e. for this set of data (this load), what the clock tells you the offset is.
        The calculated relative offset for THIS chunk load will then update this specnum offset,
        and give the actual absolute offset between timestreams.

    '''

    nchans = chanend - chanstart
    ref = bdc.BasebandFileIterator(
        files[0],
        0,
        idxs[0],
        c_acclen,
        nchunks,
        chanstart=chanstart,
        chanend=chanend,
        type="float",
    )
    nref = bdc.BasebandFileIterator(
        files[1],
        0,
        idxs[1],
        c_acclen,
        nchunks,
        chanstart=chanstart,
        chanend=chanend,
        type="float",
    )

    print('Reference antenna acclen:', ref.acclen)
    print('Non-reference antenna acclen', nref.acclen)

    #PICK THE CHUNK, PUT IN DATA
    p0_ref = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
    p0_nref = cp.zeros((c_acclen, nchans), dtype="complex64")
    ra_start = ref.spec_num_start
    nra_start = nref.spec_num_start
    

    for i, (chunk_ra, chunk_nra) in enumerate(zip(ref, nref)):
        perc_missing_ra = (1 - len(chunk_ra["specnums"]) / c_acclen) * 100
        perc_missing_nra = (1 - len(chunk_nra["specnums"]) / c_acclen) * 100
        print("percentage data missing ref ant", perc_missing_ra)
        print("percentage data missing nonref ant", perc_missing_nra)
        if perc_missing_ra > 10 or perc_missing_nra > 10:
            ra_start = ref.spec_num_start
            nra_start = nref.spec_num_start
            continue

        bdc.make_continuous_gpu(chunk_ra['pol0'],
                                chunk_ra['specnums']-ra_start,
                                np.arange(nchans),
                                c_acclen,
                                nchans=nchans, 
                                out=p0_ref)
        
        bdc.make_continuous_gpu(chunk_nra['pol0'],
                                chunk_nra['specnums']-nra_start,
                                np.arange(nchans),
                                c_acclen,
                                nchans=nchans, 
                                out=p0_nref)
        break

    specnum_offset = ref.spec_num_start - nref.spec_num_start #this is the initial delay between specnums when the antennas booted up
    p0_ref_copy = cp.copy(p0_ref)
    p0_nref_copy = cp.copy(p0_nref)

    return p0_ref_copy, p0_nref_copy, specnum_offset



def get_snr_from_coords(fit_coords,
                        nfit_coords,
                        fit_p0,
                        nfit_p0, 
                        times,
                        satmap,
                        satID,
                        chan_s_idx,
                        dN = 100000,
                        T_SPECTRA = 4096 / 250e6,
                        c_acclen = 10**6):
    
    ''' 
    Get the SNR of a pulse for ONE pulse on ONE baseline using ONE chunk.

    Parameters
    ----------

    Returns
    -------
    '''
    tle_path = outils.get_tle_file(times[0], "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    satidx = satmap[satID]
    cxcorr = get_cxcorr_many_sats(fit_p0,
                                nfit_p0, 
                                tle_path, 
                                times, 
                                [satidx],
                                satmap,
                                [fit_coords, nfit_coords],
                                dN,
                                T_SPECTRA = T_SPECTRA,
                                c_acclen=c_acclen)
    assert len(cxcorr) == 2
    snr = get_complex_snr(cxcorr[1])
    #print(f'snr {satID}', snr)
    #print('current SNR outputted', -snr[chan_s_idx])
    return -snr[chan_s_idx]


def get_snr_from_coords_many(fit_coords,
                            fit_path,
                            other_coords,
                            other_paths,
                            pulse_times,
                            pulse_chans,
                            pulse_sats,
                            satmap,
                            dN = 10e5,
                            T_SPECTRA = 4096/250e6,
                            c_acclen = 10e6):
    
    ''' 
    Get the total summed SNR for all desired pulses for all desired blines, 
    using a certain fixed fitting antenna.

    Purpose is to fit for that desired fitting antenna's coordinates.

    
    Parameters
    ----------

    
    Returns
    ------- 
    '''
    
    total_snr = 0

    for p_idx, p_times in enumerate(pulse_times):
        fit_files, fit_idx = butils.get_init_info(p_times[0], p_times[1], fit_path)
        channels = np.asarray(bdc.get_header(fit_files[0])["channels"],dtype='int64')
        chanstart = np.where(channels == 1834)[0][0]
        chanend = np.where(channels == 1852)[0][0]

        for ant_idx, nfit_coords in enumerate(other_coords):
            nfit_path = other_paths[ant_idx]
            nfit_files, nfit_idx = butils.get_init_info(p_times[0], p_times[1], nfit_path)
            p0_fit, p0_nfit, _ = get_chunk_data([fit_files, nfit_files], 
                                                [fit_idx, nfit_idx], 
                                                chanstart, 
                                                chanend, 
                                                c_acclen = c_acclen)
            total_snr += get_snr_from_coords(fit_coords,
                                            nfit_coords,
                                            p0_fit,
                                            p0_nfit,
                                            pulse_times,
                                            satmap,
                                            pulse_sats[p_idx],
                                            pulse_chans[p_idx],
                                            dN = dN,
                                            T_SPECTRA = T_SPECTRA,
                                            c_acclen = c_acclen)
    return total_snr



def get_snr_vs_coords(fit_coords, nfit_coords,
                      fit_p0, nfit_p0, 
                      step_len, nstep,
                      times,
                      satmap,
                      satID,
                      chan_s_idx,
                      dN = 100000,
                      T_SPECTRA = 4096 / 250e6,
                      c_acclen = 10**6):
    
    axis_len = 2*nstep + 1
    max_del = nstep*step_len
    fit_lat, fit_lon, fit_alt = fit_coords

    lats = np.linspace(fit_lat-max_del, fit_lat+max_del, axis_len)
    print('lats', lats)
    lons = np.linspace(fit_lon-max_del, fit_lon+max_del, axis_len)
    print('lons', lons)
    
    snr_arr = np.empty((axis_len, axis_len))

    for lat_idx, lat in enumerate(lats):
        for lon_idx, lon in enumerate(lons):
            snr_arr[lat_idx, lon_idx] = get_snr_from_coords([lat, lon, fit_alt],
                                                            nfit_coords,
                                                            fit_p0,
                                                            nfit_p0, 
                                                            times,
                                                            satmap,
                                                            satID,
                                                            chan_s_idx,
                                                            dN = dN,
                                                            T_SPECTRA = T_SPECTRA,
                                                            c_acclen = c_acclen)
    return snr_arr, lats, lons


def fit_coords_on_snr(fit_coords,
                      nfit_coords,
                      fit_p0,
                      nfit_p0, 
                      times,
                      satmap,
                      satID,
                      chan_s_idx,
                      fit_mask = (True, True, True), 
                      bounds = (0.001, 0.001, 10),
                      dN = 100000,
                      T_SPECTRA = 4096 / 250e6,
                      c_acclen = 10**6):
    
    fit_coords = np.array(fit_coords, dtype=np.float64)
    fit_mask = np.array(fit_mask, dtype = bool)
    guess_lat, guess_lon, guess_alt = fit_coords

    all_bounds = [(guess_lat - bounds[0], guess_lat + bounds[0]), 
                  (guess_lon - bounds[1], guess_lon + bounds[1]), 
                  (guess_alt - bounds[2], guess_alt + bounds[2])]
    
    bounds_cut = [b for b, m in zip(all_bounds, fit_mask) if m]

    x0 = fit_coords[fit_mask]
    print('x0', x0)

    def wrapper(x_free):
       full_coords = fit_coords.copy()
       print(full_coords)
       full_coords[fit_mask] = x_free
       print(full_coords[fit_mask])
       print(full_coords)
       return get_snr_from_coords(  full_coords,
                                    nfit_coords,
                                    fit_p0,
                                    nfit_p0,
                                    times,
                                    satmap,
                                    satID,
                                    chan_s_idx,
                                    T_SPECTRA=T_SPECTRA,
                                    dN = dN,
                                    c_acclen=c_acclen)
    
    
    fit = minimize(wrapper,
                   x0,
                   method='Powell',
                   bounds=bounds_cut,
                   options={'maxiter': 300, 'disp': True})
    
    print('FITTED X', fit.x)

    snr_fitted = fit.fun
    coords_fitted_full = fit_coords.copy()
    coords_fitted_full[np.array(fit_mask)] = fit.x


    return snr_fitted, coords_fitted_full
    
