#system stuff
import os
from os import path
import sys
sys.path.insert(0, "/home/thomasb/")
#general
import numpy as np
import numba as nb
from matplotlib import pyplot as plt
#utils
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.utils import sat_utils as sutils

#=========================================================================
#BASIC STUFF
#=========================================================================

def get_MAD(data, axis=None):
    '''Find real median absolute deviation'''
    data_median = np.median(data, axis=axis, keepdims=True)
    abs_deviations = np.abs(data - data_median)
    mad = np.median(abs_deviations, axis=axis)
    return mad

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

#=========================================================================
#VIS STUFF
#=========================================================================

@nb.njit(parallel=True)
def apply_delay(arr, out, delay, freqs):
    """Apply phase delay to visibilities.

    Parameters
    ----------
    arr : np.ndarry shape (nspec, nchan)
        Input array
    out : np.ndarray shape (nspec, nchan)
        Output array
    delay : float
        Time delay between the timestreams, in seconds.
    freqs : np.ndarray shape (nchan)
        Frequencies corresponding to each channel

    Returns
    -------
    out: np.ndarray shape ()
        Same object but now with delayed data from 'arr' in it.
    """
    nspec, nchan = arr.shape
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
def xcorr_avg(arr1, arr2, acclen):
    """
    Computes cross-correlation between two timestreams.

    Takes chunks of size 'acclen' and computes averaged conjugate product for each chunk.

    Parameters
    ----------
    arr1 : np.ndarray shape (nspec, nchan)
        First array for correlation
    arr2 : np.ndarray shape (nspec, nchan)
        Second array for correlation
    acclen : int
        number of spectra to include in single chunk of correlated data

    Returns
    -------
    np.ndarray shape (nspec//acclen, nfreq)
        Cross-correlated data.
    """
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
def xcorr_avg_1d(arr1, arr2, acclen):
    """
    Computes cross-correlation between two timestreams, but for only one channel.

    Takes chunks of size 'acclen' and computes averaged conjugate product for each chunk.

    Parameters
    ----------
    arr1 : np.ndarray shape (nspec)
        First array for correlation
    arr2 : np.ndarray shape (nspec)
        Second array for correlation
    acclen : int
        number of spectra to include in single chunk of correlated data

    Returns
    -------
    out : np.ndarray shape (nspec//acclen)
        Cross-correlated data.
    """
    nblocks = arr1.shape[0]//acclen
    out = np.zeros((nblocks,),dtype=arr1.dtype)
    for i in nb.prange(nblocks):
        for j in range(acclen):
                out[i] += arr1[i*acclen + j]*np.conj(arr2[i*acclen + j])
        out[i]/=acclen
    return out


def beamformed_xcorr(data, a1, a2, p1, p2, delay, freqs, acclen):
    """
    Compute beamformed visibilities for single baseline for specific polarizations.

    Parameters
    ----------
    data : np.ndarray shape (nblines, npol, nspec, nchans)
        Raw baseband data
    a1 : int
        Antenna 1 index
    a2 : int
        Antenna 2 index
    p1 : int
        Polarization 1 index
    p2 : int
        Polarization 2 index
    delay : np.ndarray shape (nspec)
        Time delays between antenna 1 and 2
    freqs : np.ndarray shape (nchans)
        Frequencies for each channel
    acclen : int
        Accumulation length for single visibility

    Returns
    -------
    V : np.npdarray shape (nspec//acclen, nchans)
        Beamformed visibility data for single baseline and single polarization pair.
    """
    nants, npols, nspec, nchans = data.shape
    assert len(freqs) == nchans
    assert len(delay) == nspec
    spec1=data[a1, p1, :, :].copy()
    spec2=data[a2, p2, :, :].copy()
    spec2_phased = np.empty_like(spec2)
    spec2_phased = apply_delay(spec2, spec2_phased, -delay, freqs)
    V = xcorr_avg(spec1,spec2_phased,acclen)
    return V

def get_vis(data,satID,freqs, pstart,pend,antpos,ant_idxs, tle_path, T_SPECTRA, acclen): 
    """
    Computes beamformed visibilities for N antenna over time.

    Different to version in sat_utils because this uses data already present on hard drive after upsampling.

    Parameters
    ----------
    data : np.ndarray shape (nblines, npols, nspectra, nchans)
        Contains channelized data for each baseline, for the same corresponding spectra

    satID : int
        NORAD satellite ID of the satellite we want to beamform onto
    freqs : np.ndarray shape (nchans)
        Frequencies in Hz corresponding to each channel
    pstart : int
        Pulse start time in 10 digit UNIX timestamp
    pend : int
        Pulse end time in 10 digit UNIX timestamp
    antpos : list
        List of coordinates for each antenna, each entry in form [lat, lon, alt]
    ant_idxs : list
        List of indicies of antenna that we wish to use. Some antenna may have corrupted data.
    tle_path : string
        File path leading to TLE files, which contain satellite position information.
    T_SPECTRA : float
        Period of spectrum, e.g. 1.04 ms for x64 upsampled baseband data
    acclen : int
        Number of spectra to include in each visibility chunk
    
    Returns
    -------
    np.ndarray shape (nblines, ntimes, nchans)
        Visibility Data.
    """
    n_blines = len(ant_idxs)*(len(ant_idxs)-1)//2
    vis = np.zeros((n_blines, data.shape[2]//acclen, data.shape[3]), dtype='complex64', order='c')
    print("vis shape", vis.shape)
    bl_proc=0
    for i in range(len(ant_idxs)):
        for j in range(i+1, len(ant_idxs)):
            ai, aj = ant_idxs[i], ant_idxs[j]
            a1_coords, a2_coords = antpos[ai], antpos[aj]

            dly=outils.get_sat_delay(a1_coords,a2_coords,tle_path,pstart,int(pend-pstart)+2,satID,altaz=False)
            delay = np.interp(np.arange(0, data.shape[2])*T_SPECTRA,np.arange(0,int(pend-pstart)+2),dly)

            Vxx = beamformed_xcorr(data,ai,aj,0,0,delay,freqs,acclen)
            Vyy = beamformed_xcorr(data,ai,aj,1,1,delay,freqs,acclen)
            vis[bl_proc,:,:] = (Vxx+Vyy)/2
            bl_proc+=1
            print("done", ai,aj,".Processed",bl_proc, "baselines")

    return vis


#=========================================================================
#OBJECTIVE FUNCTION AND JACOBIAN
#=========================================================================

@nb.njit()
def func(tau_t, data, freq, weights):
    ''' 
    Get residuals for given tau at single time sample, for all blines

    Parameters
    ----------
    tau_t : np.ndarray shape (nant-1)
        The time offsets for each non-reference antenna
    data : np.ndarray shape (nblines, nchan)
        Visibility data for single timestamp
    freq : np.npdarray shape (nchan)
        Frequencies corresponding to each channel
    weights : np.npdarray shape (nblines)
        Weights (from phase noise) for each baseline. 

    Returns
    -------
    np.ndarray shape (2*nblines*nchan)
        Residuals with real and imaginary parts separated for fitting.

    Notes:
    Reference antenna has zero delay, all other ants are with respect to it.
    That's why tau_t has length nant-1.
    Convention is reference - nonreference.
    Weights can also be per-channel, so shape (nblines, nchan).
    '''
    # for one time sample
    nant=len(tau_t) + 1
    nbl = nant*(nant-1)//2
    nfreq = len(freq)
    n_eval = nfreq * nbl
    residuals = np.zeros((2 * n_eval,), dtype="float64")
    for j in range(nfreq):
        blnum = 0
        two_pi_nu = 2 * np.pi * freq[j]
        for ai in range(nant):
            for aj in range(ai + 1, nant):
                rowidx = j * nbl + blnum
                if weights.ndim==2:
                    w = weights[blnum,j]
                else:
                    w = weights[blnum]
                if ai == 0:
                    pred=(-two_pi_nu * tau_t[aj - 1])  # ant 0 is refant
                else:
                    pred=(two_pi_nu * (tau_t[ai - 1] - tau_t[aj - 1]))
                y = np.exp(1j * pred) - np.exp(1j*data[rowidx])
                residuals[rowidx] = w * np.real(y)
                residuals[rowidx + n_eval] = w * np.imag(y)
                blnum += 1
    return residuals

@nb.njit()
def jac(tau_t, data, freq, weights):
    '''
    Get the Jacobian for given tau at single time sample, for all blines

    Parameters
    ----------
    tau_t : np.ndarray shape (nant-1)
        Delays for each non-reference antenna
    data : np.npdarray shape (nblines, nchan)
        Visibility data for single timestamp
    freq : np.npdarray shape (nchans)
        Frequency values for each channel
    weights: np.npdarray shape (nblines)
        Weights for each baseline determined by phase noise.
    
    Returns
    -------
    np.npdarray shape (2*nblines*nchan, nant-1)
        Jacobian of objective function at given tau values. 

    See also
    --------
    helper_finetiming.func()
        The parent objective function.
    '''
    nant=len(tau_t) + 1
    nbl = nant*(nant-1)//2
    nfreq = len(freq)
    n_eval = nfreq * nbl
    J = np.zeros((2 * n_eval, nant-1), dtype="float64")
    for j in range(nfreq):
        blnum = 0
        two_pi_nu = 2 * np.pi * freq[j]
        for ai in range(nant):
            for aj in range(ai + 1, nant):
                rowidx = j * nbl + blnum
                if weights.ndim==2:
                    w = weights[blnum,j]
                else:
                    w = weights[blnum]
                if ai == 0:
                    pred = (
                        -two_pi_nu * tau_t[aj - 1]
                    )  # ant 0 is refant
                    ym = np.exp(1j * pred)
                    dtheta = 1j * ym * w
                    re = two_pi_nu * np.real(dtheta)
                    im = two_pi_nu * np.imag(dtheta)
                    J[rowidx, aj - 1] = -re
                    J[rowidx + n_eval, aj - 1] = -im
                else:
                    pred = (
                        two_pi_nu * (tau_t[ai - 1] - tau_t[aj - 1])
                    )
                    ym = np.exp(1j * pred)
                    dtheta = 1j * ym * w
                    re = two_pi_nu * np.real(dtheta)
                    im = two_pi_nu * np.imag(dtheta)
                    J[rowidx, ai - 1] = re
                    J[rowidx, aj - 1] = -re
                    J[rowidx + n_eval, ai - 1] = im
                    J[rowidx + n_eval, aj - 1] = -im
                blnum += 1
    return J

#=========================================================================
#DATA MANIPULATION
#=========================================================================

def find_signal_channels(data, sat_type):
    '''
    Cuts channels by taking median power in time. Isolates signal channels for later.

    Parameters
    ----------
    data : np.ndarray shape (nblines, ntimes, nchan)
        Visibility data
    sat_type : str
        Type of satellite (determines bandwidth)

    Returns
    -------
    slice
        Slice of data array in channel axis, of corresponding bandwidth to sat_type
    '''
    assert sat_type in {"METEOR","NOAA"}
    if sat_type == "METEOR":
        nchan_signal = 80
    if sat_type == "NOAA":
        nchan_signal = 20
    nchan = data.shape[-1]

    power = np.abs(data)
    chan_power = np.mean(np.median(power, axis=1), axis=(0))

    cumsum = np.concatenate(([0], np.cumsum(chan_power)))
    window_sums = cumsum[nchan_signal:] - cumsum[:nchan - nchan_signal + 1]
    best_start = int(np.argmax(window_sums))

    return slice(best_start, best_start + nchan_signal)


def find_highest_amp(data, chan_slice, window_size=120):
    '''
    Determine contiguous window of highest median SNR (in terms of amplitude).

    Parameters
    ----------
    data : np.ndarray shape (nblines, ntimes, nchans)
        Visibility data
    chan_slice : slice
        Slice along frequency axis where signal is found. Determined by find_signal_channels()
    window_size : int
        Desired number of visibility times inside of which to maximize median SNR (amplitude)

    Returns
    -------
    start, end
        Start and end indices of visibility times corresponding to optimal window
    '''
    #signal power
    data_signal = data[:, :, chan_slice]
    power = np.median(np.abs(data_signal)**2, axis=2)

    #noise power
    off_mask = np.ones(data.shape[-1], dtype=bool)
    off_mask[chan_slice] = False
    off   = data[:, :, off_mask]
    noise = np.median(np.median(np.abs(off)**2, axis=2), axis=1, keepdims=True)

    #SNR
    snr = power / (noise + 1e-30)

    # collapse pols → robust time series
    snr_all = np.median(snr, axis=0)

    #windows
    if window_size > len(snr_all):
        return (0, len(snr_all))

    windows = np.lib.stride_tricks.sliding_window_view(snr_all, window_size)
    medians = np.median(windows, axis=1)
    start = int(np.argmax(medians))
    end = start + window_size

    return start, end


def find_lowest_noise(noise, window_size=120):
    """
    Determine continuous window of lowest median phase noise for multi-baseline visibility data.

    Parameters
    ----------
    noise : np.npdarray shape (nblines, ntimes)
        Array of phase noise for all baselines at all times.
    window_size : int
        Desired number of visibility times inside of which to minimize median phase noise.

    Returns
    -------
    int, int
        The best starting and ending time indices (from array noise)
        Give window of size specified in window_size
    """
    nblines, ntimes = noise.shape
    min_noise = float('inf')
    best_start_idx = -1

    for t in range(ntimes - window_size + 1):
        window = noise[:, t:t + window_size]
        window_noise = np.median(window)

        if window_noise < min_noise:
            min_noise = window_noise
            best_start_idx = t

    return best_start_idx, best_start_idx+window_size


def discrep_cutting(V, satID, acclen=1024):
    '''
    Full function that goes from beamformed visibilities to final amplitude cut
    
    Eventually phase out.
    '''
    assert satID in {28654,25338,33591,57166,59051}
    if satID in {59051, 57166}:
        sat_type = "METEOR"
    if satID in {28654,25338,33591}:
        sat_type = "NOAA"

    print('Satellite Type we see is:', sat_type)
    chan_slice = find_signal_channels(V, sat_type)
    start_chunk, end_chunk = find_highest_amp(V, chan_slice)

    new_chans = [chan_slice.start,chan_slice.stop]
    spectra_start = start_chunk*acclen
    spectra_end = end_chunk*acclen

    return spectra_start,spectra_end,new_chans


def cost_surface(data, guesses, freqs_normalized, wts, antidx = 0, timeidx=0, N1=10001, N2=40, err = None):
    """
    Compute and plot the cost surface for func() given initial offsets, and a single antenna direction to vary.

    Also includes: weightings, vertical line showing guess value, shaded region showing error on guess fit.
    Will also determine the offset value of the nearest cost minimum (trough).
    
    Parameters
    ----------
    data : np.ndarray shape (nblines, ntimes, nchans)
        Visibility data
    guesses : np.ndarray shape (nant-1)
        Offset values for each non-reference antenna (units of ns)
    freqs_normalized : np.ndarray shape (nchans)
        Frequencies for each channel (units of GHz)
    wts : np.npdarray shape (nblines)
        Weights corresponding to phase noise
    antidx : int
        Antenna index for which we want to vary tau (the x-axis variable)
    timeidx : int
        The timestamp for which we want to look at the data
    N1 : int
        Number of data points to have in the figures
    N2 : int
        Units of tau (in ns) to zoom into, either side of the guess
    err : float
        Error in the guess tau we want to plot (units of ns)

    Returns
    -------
    cost1 : np.ndarray shape (N1)
        Zoomed out cost surface
    cost2 : np.ndarray shape (N1)
        Zoomed in cost surface.
    trough : float
        Location of nearest cost minimum to guess value
    fig : matplotlib figure
        Figure with zoomed out and zoomed in cost surfaces.
    """
    antmap = ['MARS 2', 'MARS 4', 'MARS 5', 'MARS 6', 'MARS 7', 'MARS 8']
    print(data.shape)
    print(guesses[antidx])
    taus_trial1, taus_trial2 = np.tile(guesses, (N1, 1)), np.tile(guesses, (N1, 1))
    taus_trial1[:, antidx] = np.linspace(-30000, 30000, N1) + guesses[antidx]
    taus_trial2[:,antidx] = np.linspace(-N2, N2, N1) + guesses[antidx]

    cost1, cost2 = np.zeros(N1), np.zeros(N1)
    for i in range(N1):
        cost1[i] = np.sum(func(taus_trial1[i], data[timeidx,:,:].ravel(), freqs_normalized, wts)**2)
        cost2[i] = np.sum(func(taus_trial2[i], data[timeidx,:,:].ravel(), freqs_normalized, wts)**2)

    center = N1//2
    trough = taus_trial2[np.argmin(cost2[center-100:center+100])+center-100]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5),sharey=True,constrained_layout=True)

    #left
    ax[0].plot(taus_trial1[:, antidx],cost1,lw=2,color='k')
    ax[0].axvline(guesses[antidx],color='tab:red',ls='--',lw=2,label=r'Initial guess')
    ax[0].set_title(r"Full Cost Surface")
    ax[0].set_xlabel(r"Delay $\tau$ [ns]")
    ax[0].set_ylabel(r"Cost Function")
    ax[0].legend(frameon=False)

    #right
    ax[1].plot(taus_trial2[:, antidx],cost2,lw=2,color='k')
    ax[1].axvline(guesses[antidx],color='tab:red',ls='--',lw=2)
    ax[1].set_title(r"Zoomed Cost Surface")
    ax[1].set_xlabel(r"Delay $\tau$ [ns]")
    print(guesses[antidx])
    if err is not None:
        ax[1].axvspan(guesses[antidx]-err,guesses[antidx]+err,color='gray',alpha=0.3)

    #styling
    for a in ax:
        a.tick_params(direction='in',top=True,right=True)
    fig.suptitle(rf"Antenna {antmap[antidx]}: Initial Guess "rf"$\tau = {guesses[antidx]:.1f}$ ns")

    return cost1, cost2, trough, fig


def get_mask(vis, tol=1.5): 
    """
    
    """
    angle = np.angle(vis[:,:,:])
    nblines, ntimes, nchans = vis.shape

    fig, ax = plt.subplots(2,2,sharex=True)
    fig.set_size_inches(10, 10)
    img=ax[0,0].imshow(angle[0,:,:],aspect='auto',interpolation='none',cmap='RdBu')
    ax[0,0].set_title(f'Angles for Bline Index 0')

    x = np.arange(ntimes)
    residuals = np.zeros((nblines, ntimes, nchans),dtype='float64')
    blnum=0
    for i in range(len(ant_idxs)):
        for j in range(i+1, len(ant_idxs)):
            ai = ant_idxs[i]
            aj = ant_idxs[j]
            for k in range(angle.shape[2]):
                m, c = np.polyfit(x, angle[blnum,:,k], 1)
                residuals[blnum, :, k] = angle[blnum,:,k] - (m*x + c)
            blnum+=1

    mad = np.median(np.abs(residuals[:,:,:]), axis=0)
    img = ax[0,1].imshow(mad, aspect='auto',interpolation='none')
    ax[0,1].set_title('MAD across all blines')
    fig.colorbar(img, ax=ax[0,1])

    mask = mad > tol
    img = ax[1,0].imshow(mask, aspect='auto',interpolation='none')
    ax[1,0].set_title('Raw Mask all Blines')

    mask = binary_closing(mask, structure=np.ones((7,1)))
    mask_pulse = np.zeros_like(mask, dtype=bool)
    n_time, n_chan = mask.shape

    for ch in range(n_chan):
        labeled, n_features = label(mask[:, ch])
        for i in range(1, n_features + 1):
            # indices of this block
            block = (labeled == i)
            if block.sum() >= 5:
                mask_pulse[:, ch][block] = True

    img = ax[1,1].imshow(mask_pulse, aspect='auto',interpolation='none')
    ax[1,1].set_title('Filtered Mask all blines')
    return mask_pulse, fig


def get_thermal_noise(vis, ant_idxs, mask=None, T_SPECTRA=4096/250e6 * 64, acclen=1024):
    """
    Computes thermal noise for visibility data, and also plots all visibility phase angles.

    Parameters
    ----------
    vis : np.npdarray shape (nblines, ntimes, nchans)
        Visibiilty Data
    ant_idxs : list
        Indices of antenna that we want to use
    mask : np.npdarray shape (nblines, ntimes, nchans)
        Optional way to mask high noise regions out of the main data (RFI masking)
    T_SPECTRA : float
        Single spectrum sampling period
    acclen : int
        Number of spectra in single visibility time sample

    Returns
    -------
    thermal_noise : np.npdarray shape (nblines, ntimes)
        Phase noise for all blines and all times
    fig : matplotlib figure
        Figure of all baseline visibility plots
    """

    antmap = ['MARS1', 'MARS 2', 'MARS 4', 'MARS 5', 'MARS 6', 'MARS 7', 'MARS 8']
    fig,ax = plt.subplots(7,3, constrained_layout=True)
    fig.set_size_inches(10,15)
    ax=np.ravel(ax)
    plt.suptitle(f"stokes I (phase), int. time {T_SPECTRA*acclen:4.2f}s")
    nblines, ntimes, nchans = vis.shape

    thermal_noise = np.zeros((nblines, ntimes),dtype='float64')
    vis_noise = np.zeros((nblines, ntimes),dtype='float64')
    x_all  = np.arange(nchans)
    angle = np.angle(vis)
    blnum = 0
    for i in range(len(ant_idxs)):
        for j in range(i+1, len(ant_idxs)):
            ai = ant_idxs[i]
            aj = ant_idxs[j]
            ax[blnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {blnum})")
            a = angle[blnum, :, :].copy()  # (ntimes, nchan)

            if mask is None:
                a2=np.unwrap(a,axis=1)
                for k in range(a2.shape[0]):
                    m, c = np.polyfit(x_all, a2[k,:], 1)
                    residual = a2[k,:] - (m*x_all + c)
                    # sigma_phi = np.sqrt(np.mean(residual**2))
                    sigma_phi = np.std(np.angle(np.exp(1j*residual)))
                    thermal_noise[blnum,k] = sigma_phi
                    vis_noise[blnum,k] = np.std(np.cos(residual))
                img=ax[blnum].imshow(np.angle(vis[blnum,:,:]),aspect='auto',interpolation='none',cmap='RdBu')
                cbar=plt.colorbar(img,ax=ax[blnum])
            else:
                #iterate over all time chunks
                for k in range(ntimes):
                    good_idx = ~mask[k, :]
                    if good_idx.sum() < 2:
                        #print(f"Too few points to fit for time {k}")
                        thermal_noise[blnum, k] = 1
                        continue

                    elif good_idx.sum() < 10:
                        #print(f'Very few channels time {k}')
                        thermal_noise[blnum, k] = 1
                        continue

                    # unwrap and fit linear phase ramp (only for 'good' data)
                    phase_good = np.unwrap(a[k, good_idx])
                    x = np.arange(a.shape[1])[good_idx]
                    y = phase_good
                    m, c = np.polyfit(x, y, 1)

                    # get noise on ramp (only for 'good' data)
                    residual = y - (m * x + c)
                    sigma_phi = np.std(np.angle(np.exp(1j * residual)))  # circular std
                    vis_noise_val = np.std(np.cos(residual))

                    thermal_noise[blnum, k] = sigma_phi
                    vis_noise[blnum, k] = vis_noise_val

                # plot masked arrays
                a_masked = np.ma.masked_where(mask, np.angle(vis[blnum, :, :]))
                img = ax[blnum].imshow(a_masked, aspect='auto', interpolation='none', cmap='RdBu')
                plt.colorbar(img, ax=ax[blnum])
            blnum+=1
    return thermal_noise, fig

#=========================================================================
#LINEAR ALGEBRA
#=========================================================================

def get_prediction_error(res, spectra):
    '''Get the error on the predicted UTC discrepancy given residual matrix'''
    A = np.column_stack((spectra, np.ones(len(spectra))))
    N = np.cov(res)
    print(N)

    if res.ndim >1:
        v1 = np.linalg.inv(A.T@np.linalg.inv(N)@A)
    else:
        v1 = N*np.linalg.inv(A.T@A)

    return A@v1@A.T


@nb.njit()
def symmetrize(A):
    """
    Symmetrize a matrix
    """
    nr, nc = A.shape
    for i in range(nr):
        for j in range(i, nc):
            A[j, i] = A[i, j]

def get_grammian(nant):
    """
    Determine grammian for nant anntena.
    """
    nbl = (nant - 1) * nant // 2
    Ag = np.zeros((nbl, nant - 1), dtype="float64")
    b = 0
    for i in range(nant):
        for j in range(i + 1, nant):
            # print(i,j)
            if i == 0:
                Ag[b, j - 1] = -1
            else:
                Ag[b, i - 1] = 1
                Ag[b, j - 1] = -1
            b += 1
    return Ag

def get_AtA_Atd(data, Ag, noise_var, nu, nant, nfreq, ntime, fit_constant=True):
    """
    Determine both A^T A and A^T d for OLS fitting.

    Parameters
    ----------
    data : np.ndarray shape (ntimes, nchans, nblines)
        Data input. Different form than usual, as in column-major ordering
    Ag : 
        XXXX
    noise_var : 
        XXXX
    nu : 
        XXXX
    nant : int
        Number of antenna in entire system (including reference antenna)
    nfreqs : int
        Number of channels
    ntime : int
        Number of time samples we are fitting over
    fit_constant : bool
        XXXX
    """
    #ntimes, nchans, nblines = data.shape
    nparam = (nant - 1) * ntime
    if fit_constant:
        nparam += nant - 1
    nbl = (nant - 1) * nant // 2
    npertime = nfreq * nbl
    myAtA = np.empty((nparam, nparam), dtype="float64")
    myAtd = np.empty((nparam,), dtype="float64")
    myAtA[:] = 0.0
    myAtd[:] = 0.0
    two_pi = 2 * np.pi
    Snu2 = np.sum(nu**2) * two_pi**2
    Snu = np.sum(nu) * two_pi
    print("Snu", Snu, "Snu2", Snu2)
    bs = nant - 1  # block size
    # print("nant", nant, "nbl", nbl, "ntime", ntime, "nfreq", nfreq, "bs", bs)
    # print("d shape", d.shape)
    # print("n pertime", npertime)
    for i in range(ntime):
        Agw = Ag / noise_var[i, :][:, None]  # whitened A
        myAtA[i * bs : (i + 1) * bs, i * bs : (i + 1) * bs] = Ag.T @ Agw * Snu2
        if fit_constant:
            myAtA[i * bs : (i + 1) * bs, ntime * bs :] = Ag.T @ Agw * Snu
            myAtA[ntime * bs :, ntime * bs :] += Ag.T @ Agw * nfreq
        for j in range(nfreq):
            idx = i * npertime + j * nbl
            vec = Agw.T @ data[i, j, :]
            myAtd[i * bs : (i + 1) * bs] += two_pi * nu[j] * vec
            if fit_constant:
                myAtd[-bs:] += vec
    symmetrize(myAtA)
    return myAtA, myAtd