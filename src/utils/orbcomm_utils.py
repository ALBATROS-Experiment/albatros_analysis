import skyfield.api as sf
import numpy as np
import operator
import time
from matplotlib import pyplot as plt
import numba as nb
from scipy.interpolate import CubicSpline
import skyfield.api as sf
from scipy import fft
import datetime
import os
# from . import math_utils as mutils
# from . import mkfftw as mk

def ctime2mjd(tt=None, type="Dublin"):
    """Return Various Julian Dates given ctime.  Options include Dublin, MJD, JD"""
    if tt is None:
        tt = time.time()
    jd = tt / 86400 + 2440587.5
    if type == "JD":
        return jd
    elif type == "Dublin":
        return jd - 2415020
    elif type == "MJD":
        return jd - 2400000.5
    else:
        raise ValueError(
            "Unsupported Julian date type requested. Options are JD, MJD, Dublin"
        )

def get_tle_file(tstamp, dir_parent):
    date=datetime.datetime.fromtimestamp(tstamp).strftime("%Y-%m-%d")
    yyyy,mm,dd = date.split("-")
    fname = os.path.join(dir_parent, yyyy, yyyy+mm, yyyy+mm+dd+".txt")
    if os.path.isfile(fname):
        return fname
    else:
        raise FileNotFoundError(f"No TLE file found for requested date {date}.")

@nb.njit(parallel=True)
def make_continuous(newarr, arr, spec_idx):
    n = len(spec_idx)
    for i in nb.prange(n):
        newarr[spec_idx[i], :] = arr[i, :]

@nb.njit(parallel=True)
def make_continuous_rand(newarr, arr, spec_idx):
    n = len(spec_idx)
    nchan = arr.shape[1]
    for i in nb.prange(n):
        newarr[spec_idx[i], :] = arr[i, :]


@nb.njit(parallel=True)
def generate_window(N:int, type='hann'):
    win=np.empty(N,dtype='float64')
    if type=='hann':
        for i in nb.prange(N):
            win[i] = np.sin(np.pi*i/N)**2
    elif type=='hamming':
        for i in nb.prange(N):
            win[i] = 0.5434782 - (1- 0.5434782)*np.cos(2*np.pi*i/N)
    return win

@nb.njit()
def apply_window(arr,win):
    # assumes a thin horizontal rectangle for arr
    newarr = np.empty(arr.shape,dtype=arr.dtype)
    nr,nc=arr.shape
    for i in range(nr):
        for j in range(nc):
            newarr[i,j] = arr[i,j]*win[j]
    return newarr

@nb.njit(parallel=True)
def add_1d_scalar(x,y):
    n=len(x)
    z = np.empty(n, dtype=x.dtype)
    for i in nb.prange(n):
        z[i] = x[i] + y
    return z

@nb.njit(parallel=True)
def make_complex(cmpl, mag, phase):
    N = cmpl.shape[0]
    for i in nb.prange(0, N):
        cmpl[i] = mag[i] * np.exp(1j * phase[i])


@nb.njit(parallel=True)
def complex_mult_conj(arr1, arr2):
    newarr = np.empty(arr1.shape,dtype=arr1.dtype)
    Nrows = arr1.shape[0]
    Ncols = arr1.shape[1]
    for i in nb.prange(Nrows):
        for j in range(Ncols):
            newarr[i, j] = arr1[i, j] * np.conj(arr2[i, j])
    return newarr

@nb.njit(parallel=True)
def vstack_zeros_transpose(arr, bigarr):
    Nrows = arr.shape[0]
    Ncols = arr.shape[1]
    for j in nb.prange(0, Ncols):
        for i in range(0, Nrows):
            bigarr[j, i] = arr[i, j]
        for i in range(Nrows, 2 * Nrows):
            bigarr[j, i] = 0

# @nb.njit(parallel=True)
# def vstack_zeros_transpose2(arr, bigarr):
#     Nrows = arr.shape[0]
#     Ncols = arr.shape[1]
#     for j in range(0, Ncols):
#         for i in nb.prange(0, Nrows):
#             bigarr[j, i] = arr[i, j]
#         for i in nb.prange(Nrows, 2 * Nrows):
#             bigarr[j, i] = 0
            
@nb.njit(parallel=True)
def vstack_zeros_transpose2(arr, bigarr, columns):
    Nrows = arr.shape[0]
    Ncols = len(columns)
    for j in nb.prange(0, Ncols):
        for i in range(0, Nrows):
            bigarr[j, i] = arr[i, columns[j]]
        for i in range(Nrows, 2 * Nrows):
            bigarr[j, i] = 0


@nb.njit(parallel=True)
def get_weights(Nsmall):
    # get weights to normalize a zero-padded FFT
    # shape of weights is N -> EVEN = 2 * Nsmall.
    N = 2*Nsmall
    weights = np.empty(N,dtype="float64")
    weights[0] = N // 2
    weights[N // 2] = np.nan
    for i in nb.prange(1, N // 2):
        weights[i] = N // 2 - i
        weights[N - i] = N // 2 - i
    return weights


@nb.njit(parallel=True)
def apply_delay(arr, newarr, delay, freqs):
    # apply delay to an array of complex electric field or their correlation
    # does exp( j 2 pi nu tau) sign of tau is user dependent
    # freqs should correspond to the columns of the nspec x nchan array
    nspec = arr.shape[0]
    nchan = arr.shape[1]
    for i in nb.prange(nspec):
        for j in range(nchan):
            newarr[i, j] = arr[i, j] * np.exp(2j * np.pi * freqs[j] * delay[i])

@nb.njit(parallel=True)
def apply_sat_delay(arr, newarr, col2sat, delays, freqs):
    nspec = arr.shape[0]
    nchan = arr.shape[1]
    for i in nb.prange(nspec):
        for j in range(nchan):
            delay = delays[i,col2sat[j]]
            # print(f"delay {i},{j} is", delay )
            newarr[i, j] = arr[i, j] * np.exp(2j * np.pi * freqs[j] * delay)

@nb.njit(parallel=True)
def get_normalized_stamp(cxcorr, weights, dN, Npfb):
    nchans = cxcorr.shape[0]
    stamp = np.empty((nchans, 2*dN), dtype=cxcorr.dtype)
    N = cxcorr.shape[1]  # = len(weights)
    M = stamp.shape[1]  # = dN*2
    M2 = M // 2
    for i in nb.prange(nchans):
        for j in range(M2):
            stamp[i, j] = cxcorr[i, -M2 + j] / weights[-M2 + j] / Npfb
            stamp[i, M2 + j] = cxcorr[i, j] / weights[j] / Npfb
    return stamp

def get_coarse_xcorr(f1, f2, Npfb=4096):
    """Get coarse xcorr of each channel of two channelized timestreams.
    The xcorr is 0-padded, so length of output is twice the original length (shape[0]).

    Parameters
    ----------
    f1, f2 : ndarray of complex64
        First and second timestreams. Both n_spectrum x n_channel complex array.
    chans: tuple of int
        Channels (columns) of f1 and f2 that should be correlated.

    Returns
    -------
    ndarray of complex128
        xcorr of each channel's timestream. 2*n_spectrum x n_channel complex array.
    """
    if len(f1.shape) == 1:
        f1 = f1.reshape(-1, 1)
    if len(f2.shape) == 1:
        f2 = f2.reshape(-1, 1)
    chans = f1.shape[1]
    Nsmall = f1.shape[0]
    wt = np.zeros(2 * Nsmall)
    wt[:Nsmall] = 1
    n_avg = np.fft.irfft(np.fft.rfft(wt) * np.conj(np.fft.rfft(wt)))
    n_avg[Nsmall] = np.nan
    n_avg = np.tile(n_avg, chans).reshape(chans, 2*Nsmall)
    print(n_avg)
    print(f1, "\n", f2)
    bigf1 = np.vstack([f1, np.zeros(f1.shape, dtype="complex128")])
    bigf2 = np.vstack([f2, np.zeros(f2.shape, dtype="complex128")])
    bigf1 = bigf1.T.copy()
    bigf2 = bigf2.T.copy()
    bigf1f = np.fft.fft(bigf1,axis=1)
    bigf2f = np.fft.fft(bigf2,axis=1)
    # print("bigf1 fx old\n", bigf1)
    # print("bigf2 fx old\n", bigf2)
    xx = bigf1f * np.conj(bigf2f)
    xcorr = np.fft.ifft(xx,axis=1)
    # # print("n_avg is", n_avg)
    # for i, chan in enumerate(chans):
    #     # print("processing chan", chan)
    #     bigf1 = np.hstack([f1[:, chan].flatten(), np.zeros(Nsmall, dtype="complex128")])
    #     bigf2 = np.hstack([f2[:, chan].flatten(), np.zeros(Nsmall, dtype="complex128")])
    #     bigf1f = np.fft.fft(bigf1)
    #     bigf2f = np.fft.fft(bigf2)
    #     xx = bigf1f*np.conj(bigf2f)
    #     xcorr[i, :] = np.fft.ifft(xx)
    xcorr[:] = xcorr / n_avg / Npfb
    return xcorr


def get_coarse_xcorr_fast(f1, f2, dN, chans=None, Npfb=4096):
    """Get coarse xcorr of each channel of two channelized timestreams.
    The xcorr is 0-padded, so length of output is twice the original length (shape[0]).

    Parameters
    ----------
    f1, f2 : ndarray of complex64
        First and second timestreams. Both n_spectrum x n_channel complex array.
    chans: tuple of int
        Channels (columns) of f1 and f2 that should be correlated.

    Returns
    -------
    ndarray of complex128
        xcorr of each channel's timestream. 2*n_spectrum x n_channel complex array.
    """
    if len(f1.shape) == 1:
        f1 = f1.reshape(-1, 1)
    if len(f2.shape) == 1:
        f2 = f2.reshape(-1, 1)
    if chans == None:
        chans = np.arange(f1.shape[1])
    Nsmall = f1.shape[0]
    # print("Shape of passed channelized timestream =", f1.shape)
    bigf1 = np.empty((len(chans), 2 * Nsmall), dtype="complex128")
    bigf2 = np.empty((len(chans), 2 * Nsmall), dtype="complex128")
    xcorr = np.empty((len(chans), 2 * Nsmall), dtype="complex128")
    xcorr_stamp = np.empty((len(chans), 2 * dN), dtype="complex128")
    n_avg = np.empty(2 * Nsmall, dtype="float64")
    vstack_zeros_transpose(f1, bigf1)
    vstack_zeros_transpose(f2, bigf2)
    # print("bigf1",bigf1)
    get_weights(n_avg)

    n_workers = 40
    # print("n_avg is", n_avg)
    with fft.set_workers(n_workers):
        bigf1 = fft.fft(bigf1, axis=1, workers=n_workers)
        bigf2 = fft.fft(bigf2, axis=1, workers=n_workers)
        complex_mult_conj(bigf1, bigf2, xcorr)
        # print("bigf1 fx\n", bigf1)
        # print("bigf2 fx\n", bigf1)
        # print("conj mult\n", xcorr)
        xcorr = fft.ifft(xcorr, axis=1, workers=n_workers)
    # print(xcorr[0,0:5])
    get_normalized_stamp(xcorr, xcorr_stamp, n_avg, Npfb)
    return xcorr_stamp


def get_coarse_xcorr_fast2(f1, f2, dN, chans=None, Npfb=4096,window=None):
    if len(f1.shape) == 1:
        f1 = f1.reshape(-1, 1)
    if len(f2.shape) == 1:
        f2 = f2.reshape(-1, 1)
    if chans == None:
        chans = np.arange(f1.shape[1])
    Nsmall = f1.shape[0]
    # print("Shape of passed channelized timestream =", f1.shape)
    # xcorr = np.empty((len(chans), 2 * Nsmall), dtype="complex128")
    bigf1=mutils.transpose_zero_pad(f1)
    bigf2=mutils.transpose_zero_pad(f2)
    if window is not None:
        win=generate_window(2*Nsmall,window)
        bigf1=apply_window(bigf1,win)
        bigf2=apply_window(bigf2,win)
    # print("bigf1",bigf1)
    n_avg = get_weights(Nsmall)
    # print("n_avg is", n_avg)
    # with mk.parallelize_fft():
    bigf1f = mk.many_fft_c2c_1d(bigf1,axis=1)
    bigf2f = mk.many_fft_c2c_1d(bigf2,axis=1)
    # print("bigf1 fx new\n", bigf1)
    # print("bigf2 fx new\n", bigf2)
    xcorr = complex_mult_conj(bigf1f, bigf2f)

    # print("conj mult\n", xcorr)
    xcorr1 = mk.many_fft_c2c_1d(xcorr,axis=1,backward=True)
    # print("FT of xcorr is", xcorr1[0,0:5])
    xcorr_stamp = get_normalized_stamp(xcorr1, n_avg, dN, Npfb)
    return xcorr_stamp


def get_interp_xcorr(coarse_xcorr, chan, sample_no, coarse_sample_no):
    """Get a upsampled xcorr from coarse_xcorr by adding back the carrier frequency.

    Parameters
    ----------
    coarse_xcorr: ndarray
        1-D array of coarse xcorr of one channel.
    chan : int
        Channel for the passed coarse xcorr.
    osamp: int
        Number of times to over sample over the default 4 ns time-resolution. E.g. osamp=4 means 1 ns time-resolution.

    Returns
    -------
    final_xcorr_cwave: ndarray
        Complex upsampled xcorr.
    """
    # print("coarse shape", coarse_xcorr.shape)
    final_xcorr_cwave = np.empty(sample_no.shape[0], dtype="complex128")
    # print("Total upsampled timestream samples in this coarse chunk =", sample_no.shape)
    uph = np.unwrap(np.angle(coarse_xcorr))  # uph = unwrapped phase
    newphase = 2 * np.pi * chan * np.arange(0, coarse_xcorr.shape[0]) + uph
    newphase = np.interp(sample_no, coarse_sample_no, newphase)
    cs = CubicSpline(coarse_sample_no, np.abs(coarse_xcorr))
    newmag = cs(sample_no)
    make_complex(final_xcorr_cwave, newmag, newphase)
    return final_xcorr_cwave

def get_interp_xcorr_fast(coarse_xcorr, chan, sample_no, coarse_sample_no, shift, out=None, shift_phase=False):
    """Get a upsampled xcorr from coarse_xcorr by adding back the carrier frequency.

    Parameters
    ----------
    coarse_xcorr: ndarray
        1-D array of coarse xcorr of one channel.
    chan : int
        Channel for the passed coarse xcorr.
    osamp: int
        Number of times to over sample over the default 4 ns time-resolution. E.g. osamp=4 means 1 ns time-resolution.

    Returns
    -------
    final_xcorr_cwave: ndarray
        Complex upsampled xcorr.
    """
    # print("coarse shape", coarse_xcorr.shape)
    # print("sample no", sample_no)
    # print("coarse sample", coarse_sample_no)
    if isinstance(out, np.ndarray):
        final_xcorr_cwave = out
    else:
        final_xcorr_cwave = np.empty(sample_no.shape[0], dtype="complex128")
    # print("Total upsampled timestream samples in this coarse chunk =", sample_no.shape)
    shifted_sample_no = add_1d_scalar(sample_no, shift)
    uph = np.unwrap(np.angle(coarse_xcorr))  # uph = unwrapped phase
    newphase = 2 * np.pi * chan * np.arange(0, coarse_xcorr.shape[0]) + uph
    if shift_phase:
        newphase = mutils.linear_interp(shifted_sample_no, coarse_sample_no, newphase)
    else:
        newphase = mutils.linear_interp(sample_no, coarse_sample_no, newphase)
    newmag = mutils.cubic_spline(shifted_sample_no, coarse_sample_no, np.abs(coarse_xcorr))
    make_complex(final_xcorr_cwave, newmag, newphase)
    return final_xcorr_cwave


def gauss_smooth(data, sigma=5):
    """Gaussian smooth an N-dim signal. The user should take care about 0-padding.
    This function will simply smooth whatever is provided with a gaussian kernel of the same size/dimensions with an N-dim FFT.

    Parameters
    ----------
    data : ndarray
        Data to smooth
    sigma : int, optional
        width of gaussian in no. of samples, by default 5

    Returns
    -------
    ndarray
        Smoothened data
    """
    dataft = np.fft.rfft(data)
    x = np.fft.fftfreq(len(data)) * len(data)
    gauss = np.exp(-0.5 * (x**2) / sigma**2)
    gauss = gauss / gauss.sum()
    kernelft = np.fft.rfft(gauss)
    return np.fft.irfft(dataft * kernelft)


def get_risen_sats(tle_file, coords, t_start, dt=None, niter=560, good=None,altitude_cutoff=1):
    """Get all satellites risen at a particular point on earth at a list of epochs.
    Epochs start at t_start and a list of risen satellites is returned for every t_start + i * dt epoch
    The satellites are read form a TLE file (currently hardcoded).

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
    junk = [11060, 
            35865, #Meteor M-1
            14154, #HILAT
            28650,  #HAMSAT, old Indian sat
            28371, #Saudi sat 2
            5580 #PROSPERO
            ]
    if not good:
        good=[28654,25338,33591,57166,59051,44387]
    obs1 = sf.wgs84.latlon(*coords)
    sats = sf.load.tle_file(tle_file)
    tt = t_start
    ts = sf.load.timescale()
    if not dt:
        dt = 393216 * 4096 / 250e6
    print("starting at ", tt, "dt is", dt)
    risen_sats = []
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


def find_pulses(x, cond="==", thresh=None, pulses=True):
    """Given a signal "x", find locations where the signal is ON.
    Whether or not the signal is ON is determined by the comparison condition passed.
    Comparison done is `x cond 0`.

    Parameters
    ----------
    x : ndarray (float or int)
        Signal timestream
    cond : str
        One of "==", ">", "<", ">=", "<=", "!="
        Consider the effect of round-off error if you are using "eq" with a float array.
    thresh : float, optional
        If passed, (x - thresh) is compared against the condition, by default None
    """
    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    if thresh:
        x = x - thresh
    arr = ops[cond](x, 0)
    crossings = np.where(np.abs(np.diff(arr)) == 1)[0] + 1
    # print("crossings", crossings)
    nseg = len(crossings) + 1
    # print("num of segments", nseg)
    boundaries = np.hstack([0, crossings, len(arr)])
    # print("boundaries", boundaries)
    pulse_boundaries = []
    if pulses:
        if arr[0] == 0:
            # print("OFF")
            # first segment is OFF. return every other segment
            # most often there will be even number of crossings, odd no. of segments
            # since data starts with OFF and ends with OFF
            for i in range(1, nseg, 2):
                pulse_boundaries.append([boundaries[i], boundaries[i + 1]])
        #     for i in range(0,len(crossings),2): #if len crossings is even, will never reach the last element
        #         if (crossings[i+1] - crossings[i]) > 5: # only if signal stays ON for at least 30 seconds, consider it a sat pass
        #             pulse_boundaries.append([crossings[i]+1,crossings[i+1]+1])
        else:
            # print("ON")
            for i in range(0, nseg, 2):
                #         if (crossings[i+1]-crossings[i]) > 1:
                pulse_boundaries.append(
                    [boundaries[i], boundaries[i + 1]]
                )  # len of boundaries is nseg+1. boundaries i+1 will always work. since i can be max nseg
        return pulse_boundaries
    else:
        return boundaries


# this function takes in satellite transit data returned by a different function
# def get_simul_pulses(transits, nrows, mask=None, thresh=5):
#     nchan = len(transits)
#     passes = np.zeros((nrows, nchan), dtype=bool)
#     for c in range(0, 20):
#         for t in transits[c]:
#             if t[1] - t[0] > 10:
#                 passes[t[0] : t[1], c] = 1
#     if mask is not None:
#         assert len(mask) == nrows
#         passes[:] = passes * mask
#     plt.imshow(passes, aspect="auto", interpolation="none")
#     x = np.arange(nchan - 1, -1, -1, dtype=int).reshape(nchan, -1)
#     pwr = 2 ** (np.ones(nrows, dtype=int).reshape(nrows, 1) @ x.T)
#     rep = np.sum(passes * pwr, axis=1)
#     cur = 0
#     curidx = 0
#     curlen = 0
#     pulses = []
#     for i, x in enumerate(rep):
#         #         print(i, x)
#         if x != cur:
#             if curlen > thresh and cur > 0:
#                 pulses.append([[curidx, i], get_set_bits(cur, reverse=True)])
#             cur = x
#             curlen = 0
#             curidx = i
#         else:
#             curlen += 1
#     return pulses


def get_simul_pulses(passes, mask=None, thresh=9):
    if mask is not None:
        assert len(mask) == nrows  # mask is buggy currently. fix
        passes[:] = passes * mask
    plt.imshow(passes, aspect="auto", interpolation="none")
    x = 2 ** np.arange(0, passes.shape[1], dtype=int).reshape(passes.shape[1], -1)
    rep = passes @ x  # treat each row of 0s and 1s as a binary number.
    rep = np.vstack(
        [rep, 0]
    )  # make sure that if the data ends with risen sats, we catch them. force a transition. nothing happens if 0s.
    # get a timestream of numbers. if the number changes then at least one of the bits flipped somewhere.
    cur = 0
    curidx = 0
    curlen = 0
    pulses = []
    for i, x in enumerate(rep):
        #         print(i, x)
        if x != cur:
            if curlen > thresh and cur > 0:
                pulses.append([[curidx, i], get_set_bits(cur, nbits=passes.shape[1])])
            cur = x
            curlen = 0
            curidx = i
        else:
            curlen += 1

    return pulses


def get_set_bits(x, nbits=20, reverse=False):
    if reverse:
        return [nbits - i - 1 for i in range(0, nbits) if (x >> i) & 1]
    return [i for i in range(0, nbits) if (x >> i) & 1]


def find_sat_transits(spectra, acctime=None, snr_thresh=5):
    nspec, nchan = spectra.shape
    # convert to SNR in dB. median inside log same as median outside log since log is strictly monotonously increasing.
    # gauss smooth along time after appending zeros
    spectra = np.apply_along_axis(
        gauss_smooth,
        0,
        np.vstack(
            [
                10 * np.log10(spectra / np.median(spectra)),
                np.zeros((nspec, nchan), dtype=spectra.dtype),
            ]
        ),
    )[:nspec, :]
    transits = {}
    for chan in range(0, nchan):
        transits[chan] = find_pulses(spectra[:, chan], cond=">", thresh=snr_thresh)

    return transits


def get_sat_delay(pos1, pos2, tle_path, time_start, niter, satnorad, altaz=False):
    obs1 = sf.wgs84.latlon(pos1[0], pos1[1], pos1[2])
    # obs1=sf.wgs84.latlon(51.4641932, -68.2348603,336.499)
    obs2 = sf.wgs84.latlon(pos2[0], pos2[1], pos2[2])

    num_iter = niter
    sim_delay = np.zeros(num_iter)
    c = 299792458
    tt = time_start  # seek to timestamp where signal begins
    ind = None

    sats = sf.load.tle_file(tle_path)

    for i, sat in enumerate(sats):
        if sat.model.satnum == satnorad:
            ind = i
            break
    print(sats[ind])
    altaz1=np.zeros((num_iter,2),dtype="float64")
    altaz2=altaz1.copy()
    diff1 = sats[ind] - obs1
    diff2 = sats[ind] - obs2
    ts = sf.load.timescale()
    for iter in range(num_iter):
        jd = ctime2mjd(tt, type="JD")
        t = ts.ut1_jd(jd)

        topo1 = diff1.at(t)
        alt, az, dist = topo1.altaz()
        range1 = dist.m
        altaz1[iter]=alt.degrees,az.degrees

        topo2 = diff2.at(t)
        alt, az, dist = topo2.altaz()
        range2 = dist.m
        altaz2[iter]=alt.degrees,az.degrees

        sim_delay[iter] = (range2 - range1) / c
        tt += 1
    if altaz:
        return sim_delay, altaz1, altaz2
    return sim_delay

def delay_corrector(idx1, idx2, delay, dN):
    # convetion is xcorr = <a(t)b(t-delay)>
    # where a = antenna1 and b = antenna2
    delay = delay - dN  # this is dN from the coarse xcorr
    print("original", idx1, idx2)
    if delay > 0:
        idx1 += delay
    else:
        idx2 += np.abs(delay)
    print("corrected", idx1, idx2)
    return idx1, idx2

def chan2freq(chan,alias=False,samp=250e6,fftlen=4096):
    if alias:
        return samp*(1-chan/fftlen)
    else:
        return samp*chan/fftlen
