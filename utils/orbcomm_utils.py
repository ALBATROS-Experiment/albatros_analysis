import skyfield.api as sf
import numpy as np
import operator


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


def get_risen_sats(coords, t_start, dt=None, niter=560, altitude_cutoff=1):
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
    sats = sf.load.tle_file("./data/orbcomm_28July21.txt")
    tt = t_start
    ts = sf.load.timescale()
    if not dt:
        dt = 393216 * 4096 / 250e6
    print("starting at ", tt, "dt is", dt)
    risen_sats = []
    for iter in range(560):
        visible = []
        alt_count = 0
        jd = ctime2mjd(tt, type="JD")
        t = ts.ut1_jd(jd)
        for sat in sats:
            if (
                "[+]" not in sat.name and "NOAA" not in sat.name
            ):  # extracting operational ORBCOMM ([+]) and NOAA from TLE file
                continue
            diff = sat - obs1
            topocentric = diff.at(t)
            alt, az, dist = topocentric.altaz()
            # print(sat.name)
            if alt.degrees > altitude_cutoff:
                # print(alt.degrees, az.degrees)
                visible.append(sat.name)
        #         if(alt_count in (1,)):
        #             print(iter,'have ',alt_count,' in beam -6 dB range', visible)
        risen_sats.append(visible)
        tt += dt
    return risen_sats


def find_pulses(x, cond="==", thresh=None):
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
    print("crossings", crossings)
    nseg = len(crossings) + 1
    print("num of segments", nseg)
    boundaries = np.hstack([0, crossings, len(arr)])
    print("boundaries", boundaries)
    retval = []
    if arr[0] == 0:
        print("OFF")
        # first segment is OFF. return every other segment
        # most often there will be even number of crossings, odd no. of segments
        # since data starts with OFF and ends with OFF
        for i in range(1, nseg, 2):
            retval.append([boundaries[i], boundaries[i + 1]])
    #     for i in range(0,len(crossings),2): #if len crossings is even, will never reach the last element
    #         if (crossings[i+1] - crossings[i]) > 5: # only if signal stays ON for at least 30 seconds, consider it a sat pass
    #             retval.append([crossings[i]+1,crossings[i+1]+1])
    else:
        print("ON")
        for i in range(0, nseg, 2):
            #         if (crossings[i+1]-crossings[i]) > 1:
            retval.append(
                [boundaries[i], boundaries[i + 1]]
            )  # len of boundaries is nseg+1. boundaries i+1 will always work. since i can be max nseg
    print(retval)
