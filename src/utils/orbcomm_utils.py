import skyfield.api as sf
import numpy as np
import operator
import time
from matplotlib import pyplot as plt
import numba as nb
from scipy.interpolate import CubicSpline
import skyfield.api as sf

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


@nb.njit(parallel=True)
def make_continuous(newpol, pol, spec_idx):
    n = len(spec_idx)
    for i in nb.prange(n):
        newpol[spec_idx[i], :] = pol[i, :]

@nb.njit(parallel=True)
def make_complex(cmpl, mag, phase):
    N = cmpl.shape[0]
    for i in nb.prange(0, N):
        cmpl[i] = mag[i] * np.exp(1j * phase[i])

def get_coarse_xcorr(f1, f2, chans=None, Npfb=4096):
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
        f1 = f1.reshape(-1,1)
    if len(f2.shape) == 1:
        f2 = f2.reshape(-1,1)
    if(chans==None):
        chans = np.arange(f1.shape[1])
    Nsmall = f1.shape[0]
    #print("Shape of passed channelized timestream =", f1.shape)
    xcorr = np.zeros((len(chans),2 * Nsmall), dtype="complex128")
    wt = np.zeros(2 * Nsmall)
    wt[:Nsmall] = 1
    n_avg = np.fft.irfft(np.fft.rfft(wt) * np.conj(np.fft.rfft(wt)))
    #print("n_avg is", n_avg)
    for i, chan in enumerate(chans):
        #print("processing chan", chan)
        xcorr[i, :] = np.fft.ifft(
            np.fft.fft(
                np.hstack([f1[:, chan].flatten(), np.zeros(Nsmall, dtype="complex128")])
            )
            * np.conj(
                np.fft.fft(
                    np.hstack(
                        [f2[:, chan].flatten(), np.zeros(Nsmall, dtype="complex128")]
                    )
                )
            )
        )
        xcorr[i, :] = xcorr[i, :] / n_avg / Npfb
    return xcorr

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
    final_xcorr_cwave = np.empty(
        sample_no.shape[0], dtype="complex128"
    )
    # print("Total upsampled timestream samples in this coarse chunk =", sample_no.shape)
    uph = np.unwrap(np.angle(coarse_xcorr))  # uph = unwrapped phase
    newphase = 2 * np.pi * chan * np.arange(0, coarse_xcorr.shape[0]) + uph
    newphase = np.interp(sample_no, coarse_sample_no, newphase)
    cs = CubicSpline(coarse_sample_no, np.abs(coarse_xcorr))
    newmag = cs(sample_no)
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


def get_risen_sats(tle_file, coords, t_start, dt=None, niter=560, altitude_cutoff=1):
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
                visible.append([sat.model.satnum, alt.degrees])
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


def get_simul_pulses(transits, nrows, mask=None, thresh=5):
    nchan = len(transits)
    passes = np.zeros((nrows, nchan), dtype=bool)
    for c in range(0, 20):
        for t in transits[c]:
            if t[1] - t[0] > 10:
                passes[t[0] : t[1], c] = 1
    if mask is not None:
        assert len(mask) == nrows
        passes[:] = passes * mask
    plt.imshow(passes, aspect="auto", interpolation="none")
    x = np.arange(nchan - 1, -1, -1, dtype=int).reshape(nchan, -1)
    pwr = 2 ** (np.ones(nrows, dtype=int).reshape(nrows, 1) @ x.T)
    rep = np.sum(passes * pwr, axis=1)
    cur = 0
    curidx = 0
    curlen = 0
    pulses = []
    for i, x in enumerate(rep):
        #         print(i, x)
        if x != cur:
            if curlen > thresh and cur > 0:
                pulses.append([[curidx, i], get_set_bits(cur, reverse=True)])
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

def get_sat_delay(pos1, pos2, tle_path, time_start, niter, satnorad):
    obs1=sf.wgs84.latlon(pos1[0], pos1[1], pos1[2])
    # obs1=sf.wgs84.latlon(51.4641932, -68.2348603,336.499)
    obs2=sf.wgs84.latlon(pos2[0], pos2[1], pos2[2])

    num_iter=niter
    sim_delay=np.zeros(num_iter)
    c=299792458
    tt=time_start # seek to timestamp where signal begins
    ind=None

    sats=sf.load.tle_file(tle_path)

    for i,sat in enumerate(sats):
        if(sat.model.satnum==satnorad):
            ind=i
            break
    print(sats[ind])
    diff1=sats[ind]-obs1
    diff2=sats[ind]-obs2
    for iter in range(num_iter):
        ts=sf.load.timescale()
        jd=ctime2mjd(tt,type='JD')
        t=ts.ut1_jd(jd)

        topo1=diff1.at(t)
        alt,az,dist=topo1.altaz()
        range1=dist.m

        topo2=diff2.at(t)
        alt,az,dist=topo2.altaz()
        range2=dist.m

        sim_delay[iter]= (range2-range1)/c
        tt+=1
    return sim_delay
