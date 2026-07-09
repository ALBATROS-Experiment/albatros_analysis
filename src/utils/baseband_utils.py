import numpy as np
import os, warnings
from matplotlib import pyplot as plt
import subprocess
import pytz
from datetime import datetime, timezone
import glob
import re
import json

def _find(dir_parent, search_type, search_tag, min_depth):
    return subprocess.run(
        [
            "find",
            dir_parent,
            "-type",
            search_type,
            "-regextype",
            "posix-extended",
            "-regex",
            search_tag,
            "-mindepth",
            min_depth,
        ],
        capture_output=True,
    ).stdout.decode("utf-8")


def get_file_from_timestamp(ts, dir_parent, search_type, force_ts=False, acclen=393216):
    """Given a timestamp, return the file inside of which that timestamp lies.
    The function works with both baseband and direct spectra files.

    Parameters
    ----------
    ts : int or str
        The timestamp you're interested in. (ctime)
    dir_parent : str
        The directory which contains 5 digit folders.
    search_type : 'd' or 'f'
        'd' (directory) if you are using this function for direct spectra, 'f' (file) for baseband.
    force_ts : bool, default False
        Return the next available file if a file containing ts is not found.

    Returns
    -------
    str
        Absolute path to the file which contains your timestamp.

    Raises
    ------
    NotFoundError
        If there is no matching file, the user is required to start their integration
        from the next available timestamp that's helpfully suggested.
    """
    assert(search_type in ["f", "d"])
    if isinstance(ts, int) or isinstance(ts, float):
        ts = str(ts)
    stamps = [int(ts[:5]), int(ts[:5])-1]
    search_tags = [f".*\/{stamp}[0-9]{{5}}" for stamp in stamps]
    if search_type == "f":
        search_tags = [tag + "\.raw" for tag in search_tags]
    search_tag = "|".join(search_tags)
    ts = float(ts)
    op = _find(dir_parent, search_type, search_tag, "2")
    files = op.split()
    files.sort()  # files need to be in the same order as the timestamps, so we can simply pick the correct file later
    # print("FROM UTILS_______:",files)
    tstamps = np.asarray(
        [int(s.split("/")[-1].split(".")[0]) for s in files]
    )  # will work with both tstamp.raw (bband) and 5digit/tstamp/ (direct)
    tstamps.sort()  # could remove this.
    # print(tstamps)
    if search_type == "d":
        delta = 3600  # direct spectra files are time-limited files. no need to run a median.
        dt = acclen * 4096 / 250e6
    else:
        delta = np.median(
            np.diff(tstamps)
        )  + 3 # find the time period. assumption: usually there will be several files in an hour.
        # + 3 because tstamp accuracy is only 1 s, and duration of file tstamp may fluctuate a bit
        # if something happens to system, file gap >> 1 s.
        mean_diff = np.mean(np.diff(tstamps))
        std_diff = np.std(np.diff(tstamps))
        # plt.hist(np.diff(tstamps), bins=20)
        # plt.show()
        # print("mean diff", mean_diff, "+/-", std_diff)
        dt = 4096 / 250e6
    # print(tstamps>ts)
    # if len(tstamps) == 1:
    #     flip = tstamps >= ts
    # print("delta is", delta, "ts is", ts)
    tstamps = np.hstack([tstamps, tstamps[-1] + delta])
    # print(tstamps)
    # print(tstamps  < ts)
    # xx=(tstamps < ts).astype(int)
    # print(xx)
    if len(tstamps) == 1:
        flip = 0
    else:
        flip = np.where(np.diff(tstamps > ts) != 0)[0][0]
    # plt.plot(tstamps>ts)
    # print(flip,delta,tstamps[flip],files[flip])
    # print("consecutive file diff", tstamps[flip+1]-tstamps[flip], "distance of requested tstamp from start of current file",ts - tstamps[flip])
    if (tstamps[flip+1]-tstamps[flip]) < (ts - tstamps[flip]): #if length of current file is shorter than distance of requested time from start of current file, 
                                                                #tstamp prolly in next file
        print("INCREASING FLIP BY ONE...")
        assert 1 == 0
        # should never happen because flip means next file tstamp is greater than requested.
        flip+=1
    if ts - tstamps[flip] <= delta:
        return files[flip], np.round((ts - tstamps[flip]) / dt).astype(
            int
        )  # return the file and where inside the file you expect to find this timestamp
    else:
        if force_ts:
            if len(tstamps) == 1:
                raise FileNotFoundError(
                    f"You're using force_ts but ran out of files close to requested timestamp. Should've collected more data."
                )
            warnings.warn(
                f"Returning a file whose start is {tstamps[flip + 1]}, {tstamps[flip + 1] -  ts} seconds away from your requested timestamp"
            )
            return files[flip + 1], 0
        else:
            raise FileNotFoundError(
                f"No file match for requested timestamp. Perhaps there was a data acquisition gap. Use force_ts = True."
            )
    # should be less than equal to delta, then our tstamp lies in a file.
    # otherwise there's no such file with that timestamp. tell user to start from the next future timestamp
    # force_ts = True  may be?

def time2fnames(time_start, time_stop, dir_parent, search_type, fraglen=5,mind_gap=False):
    """Gets a list of filenames within specified time-rage.

    Given a start and stop ctime, retrieve list of corresponding files.
    This function assumes that the parent directory has the directory
    structure <dir_parent>/<5-digit coarse time fragment>/<10-digit
    fine time stamp>.

    Paramaters
    -----------
    time_start: int
        start time in ctime
    time_stop: int
        stop time in ctime
    dir_parent: str
        parent directory, e.g. /path/to/data_100MHz
    fraglen: int
        number of digits in coarse time fragments

    Returns
    -------
    list of str
        List of files in specified time range.
    """
    # print(time_start, time_stop)
    assert(search_type in ["f", "d"])
    assert(time_stop >= time_start)
    time_start, time_stop = [str(t) for t in [time_start, time_stop]]
    print(time_start, time_stop)
    stamps = np.arange(int(time_start[:5]), int(time_stop[:5])+1)
    search_tags = [f".*\/{stamp}[0-9]{{5}}" for stamp in stamps]
    if search_type == "f":
        search_tags = [tag + "\.raw" for tag in search_tags]
    search_tag = "|".join(search_tags)
    op = _find(dir_parent, search_type, search_tag, "2")
    # print(op)
    files = op.split() #get all files for all 5-digit tstamps spanning the range.
    files.sort()
    tstamps = np.asarray(
        [int(s.split("/")[-1].split(".")[0]) for s in files]
    )
    # print(tstamps)
    idx = np.where(np.bitwise_and(tstamps>=int(time_start),tstamps<=int(time_stop)))[0]
    if mind_gap:
        tdiff = np.diff(tstamps[idx])
        if search_type == "d":
            delta = 3600 + 5*60 # direct spectra files are time-limited files. no need to run a median.
        else:
            delta = np.median(tdiff)+10 #same logic as get_file_from_timestamp
            print("Using delta of", delta, "seconds to check for gaps between files")
        gaps=np.where(tdiff>delta)[0]
        if len(gaps) > 0: #there's a big gap in the middle. Either return only the first part or raise error. raising error for now
            for ii in gaps:
                print("gap after file", files[idx[ii]], "of", tdiff[ii], "seconds.")
            raise Exception("Whoops! big gap in the requested timestop - timestart range. May be system rebooted?")
    if len(idx) == 0:
        raise FileNotFoundError("No files found between the requested timestamps")
    return [files[i] for i in idx]

def get_tstamp_from_filename(f):
    s = re.compile(r"(\d{10})")
    if s.search(f):
        return int(s.search(f).groups()[0])
    return None


def get_ctime_from_locatime(lt, tz="US/Eastern"):
    tz = pytz.timezone(tz)
    tstamp = tz.localize(datetime.datetime.strptime(lt, "%Y%m%d_%H%M%S")).timestamp()
    return tstamp


def get_localtime_from_ctime(tstamp, tz="US/Eastern"):
    tz = pytz.timezone(tz)
    return datetime.datetime.fromtimestamp(tstamp, tz=pytz.utc).astimezone(tz)


def get_init_info(init_t, end_t, dir_parent, force_ts = False):
    """Get relevant indices from timestamps.

    Returns the index of file in a folder and
    the index of the spectra in that file corresponding to init_timestamp

    Parameters
    ----------
    init_t: int or float
        Start timestamp of time we are interested in. (ctime)
    end_t: int or float
        End of time window we are interested in. (ctime)
    parent_dir: str
        The directory we search in.

    Returns
    -------
    idxstart: int
        The index of the starting row of interest to us within a
        baseband data file.
    fileidx: int
        The index of the file in the sorted directory.
    files: list of str
        Sorted list of path strings to all files in 'parent_dir'.
    """
    f1,idx=get_file_from_timestamp(init_t,dir_parent,'f', force_ts=force_ts)
    f2,_=get_file_from_timestamp(end_t,dir_parent,'f', force_ts=force_ts)
    files=time2fnames(get_tstamp_from_filename(f1),get_tstamp_from_filename(f2),dir_parent,'f')
    return files,idx

def get_plot_lims(pol, acclen):
    """Get limits for display settings for pretty plots!

    Parameters
    ----------
    pol: np.ndarray
        Baseband from a baseline. (channelized data.) *Is this a 2d array??*
    acclen: int
        *Depricated, doesn't get used.*

    Returns
    -------
    med: float
        Mean power in pol.
    vmin: float
        Two standard deviations below mean.
    vmax: float
        Two standard deviations above mean.
    """
    # numpy percentile method ignores mask and may generate garbage with 0s (missing specs).
    # Pivot to using mean if acclen too small.

    # if(acclen>250000):
    #     med = np.mean(pol)
    #     xx=np.ravel(pol).copy()
    #     u=np.percentile(xx,99)
    #     b=np.percentile(xx,1)
    #     xx_clean=xx[(xx<=u)&(xx>=b)] # remove some outliers for better plotting
    #     stddev = np.std(xx_clean)
    # else:
    #     med = np.mean(pol)
    #     stddev = np.std(pol)
    med = np.ma.mean(pol)
    stddev = np.ma.std(pol)
    vmin = med - 2 * stddev
    vmax = med + 2 * stddev
    print("med and plot lims", med, vmin, vmax)
    return med, vmin, vmax


def plot_4bit(
    pol00,
    pol11,
    pol01,
    channels,
    acclen,
    time_start,
    vmin,
    vmax,
    opath,
    minutes=False,
    logplot=True,
):
    """Waterfall plotting routine for 4-bit spectrum integrated data.

    Plots and saves figure.

    Parameters
    ----------
    pol00: np.ndarray
        Channelized autocorr data from pol0.
    pol11: np.ndarray
        Channelized autocorr data from pol1.
    pol01: np.ndarray
        Channelized x-corr data (pol0 x pol1).
    channels: array like
        Indices of channels of interest.
    acclen: int
        ??
    time_start: int or float
        ??In what format??
    vmin: float
        Plotting parameter. Two standard deviations below mean. Used
        to set colorbar scale.
    vmax: float
        Plotting parameter. Two standard deviations above mean. Used
        to set colorbar scale.
    opath: str
        Ouput path. Path to image of figure to be saved.
    minutes: bool
        Whether to display time in minutes. Defaults to False,
        displaying seconds.
    logplot: bool
        Defaults to True. Logarithmic scale on y-axis.
    """
    freq = channels * 125 / 2048  # MHz
    pol00_med = np.ma.median(pol00, axis=0)
    pol11_med = np.ma.median(pol11, axis=0)
    pol00_mean = np.ma.mean(pol00, axis=0)
    pol11_mean = np.ma.mean(pol11, axis=0)
    pol00_max = np.ma.max(pol00, axis=0)
    pol11_max = np.ma.max(pol11, axis=0)
    pol00_min = np.ma.min(pol00, axis=0)
    pol11_min = np.ma.min(pol11, axis=0)
    if (vmin is None) and (vmax is None):
        med, vmin, vmax = get_plot_lims(
            pol00, acclen
        )  # use of acclen in get_plot_lims is depricated
        med2, vmin2, vmax2 = get_plot_lims(
            pol11, acclen
        )  # use of acclen in get_plot_lims is depricated
    else:
        print("SETTING VMIN AND VMAX")
        vmin = 10**vmin
        vmax = 10**vmax
        vmin2 = vmin
        vmax2 = vmax
    pol01_mag = np.abs(pol01)
    if logplot:
        print("IN LOGPLOT")
        pol00 = np.log10(pol00)
        pol11 = np.log10(pol11)
        pol00_med = np.log10(pol00_med)
        pol11_med = np.log10(pol11_med)
        pol00_mean = np.log10(pol00_mean)
        pol11_mean = np.log10(pol11_mean)
        pol00_max = np.log10(pol00_max)
        pol11_max = np.log10(pol11_max)
        pol00_min = np.log10(pol00_min)
        pol11_min = np.log10(pol11_min)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
        vmin2 = np.log10(vmin2)
        vmax2 = np.log10(vmax2)
        pol01_mag = np.log10(pol01_mag)

    plt.figure(figsize=(18, 10), dpi=200)
    t_acclen = acclen * 2048 / 125e6  # seconds
    t_end = pol01.shape[0] * t_acclen
    tag = "Seconds"  # Warning: confusing variable name, tag is used in SNAPfiletools in very different way/meaning
    if minutes:
        t_end = t_end / 60
        tag = "Minutes"
    myext = np.array(
        [np.min(channels) * 125 / 2048, np.max(channels) * 125 / 2048, t_end, 0]
    )
    plt.suptitle(f"{tag} since {time_start}")
    plt.subplot(2, 3, 1)
    plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect="auto", extent=myext)
    plt.title("pol00")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(tag)
    cb00 = plt.colorbar()
    cb00.ax.plot([0, 1], [7.0] * 2, "w")

    plt.subplot(2, 3, 4)
    plt.imshow(pol11, vmin=vmin2, vmax=vmax2, aspect="auto", extent=myext)
    plt.title("pol11")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(tag)
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title("Basic stats for frequency bins")
    plt.plot(freq, pol00_max, "r-", label="Max")
    plt.plot(freq, pol00_min, "b-", label="Min")
    plt.plot(freq, pol00_mean, "k-", label="Mean")
    plt.plot(freq, pol00_med, color="#666666", linestyle="-", label="Median")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("pol00")

    plt.subplot(2, 3, 5)
    plt.plot(freq, pol11_max, "r-", label="Max")
    plt.plot(freq, pol11_min, "b-", label="Min")
    plt.plot(freq, pol11_mean, "k-", label="Mean")
    plt.plot(freq, pol11_med, color="#666666", linestyle="-", label="Median")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("pol11")
    plt.legend(loc="lower right", fontsize="small")

    plt.subplot(2, 3, 3)
    plt.imshow(pol01_mag, aspect="auto", extent=myext)
    plt.title("pol01 magnitude")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(tag)
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(
        np.angle(pol01),
        vmin=-np.pi,
        vmax=np.pi,
        aspect="auto",
        extent=myext,
        cmap="RdBu",
    )
    plt.ylabel(tag)
    plt.xlabel("Frequency (MHz)")
    plt.title("pol01 phase")
    plt.colorbar()
    plt.savefig(opath)
    return


def plot_1bit(pol01, channels, acclen, time_start, opath, minutes=False, logplot=False):
    """Waterfall plotting routine for 1-bit spectrum integrated data.

    Plots and saves figure.

    Parameters
    ----------
    pol01: np.ndarray
        Channelized x-corr data (pol0 x pol1).
    channels: array like
        Indices of channels of interest.
    acclen: int
        ??
    time_start: ??
        ??In what format; is this relative time or absolute timestamp?
    opath: str
        Ouput path. Path to image of figure to be saved.
    minutes: bool
        Whether to display time in minutes. Defaults to False,
        displaying seconds.
    logplot: bool
        *Depricated.*
        Defaults to True. Logarithmic scale on y-axis.
    """
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 4)
    t_acclen = acclen * 2048 / 125e6  # seconds
    t_end = pol01.shape[0] * t_acclen
    tag = "Seconds"
    if minutes:
        t_end = t_end / 60
        tag = "Minutes"
    myext = np.array(
        [np.min(channels) * 125 / 2048, np.max(channels) * 125 / 2048, t_end, 0]
    )

    plt.suptitle(f"{tag} since {time_start}")
    img1 = ax[0].imshow(
        np.real(pol01), aspect="auto", vmin=-0.005, vmax=0.005, extent=myext
    )
    ax[0].set_title("pol01 real part")
    img2 = ax[1].imshow(
        np.imag(pol01), aspect="auto", vmin=-0.005, vmax=0.005, extent=myext
    )
    ax[1].set_title("pol01 imag part")
    plt.colorbar(img1, ax=ax[0])
    plt.colorbar(img2, ax=ax[1])
    plt.savefig(opath)
    return

def get_pfb_chans(channels,osamp):
    pfbchans = np.array([],dtype='int64')
    bd = np.where(np.diff(channels)!=1)[0]
    bd = np.append(bd,[len(channels)-1])
    start=0
    for end in bd:
        print(start, end)
        temp=np.arange((channels[start]-1)*osamp,(channels[end]+1)*osamp)
        pfbchans=np.append(pfbchans,temp)
        start=end+1
    return pfbchans

def load_antenna_power(init_t, end_t, dir_parent, chans=None, tags=['pol00', 'pol11']):
    """
    Load direct spectra for a given antenna. Internally, this function finds all files
    between two timestamps, loads specified pols (00,11, 01r, 01i) for each file
    and returns an array of dimensions (npol,nchan,ntime), 
    where ntime ~ (end_t - init_t)/6.44.

    Parameters
    ----------
    init_t: float
        Unix C-time for data start.
    end_t: float
        Indices of channels of interest.
    dir_parent: str
        Parent directory for direct data (should contain 5 digit dirs)
    chans: array-like, optional
        List of frequency channels in range [0, 2048) that need to be loaded.
        Default None, (set to the full range)
    tags: list, optional
        List of polarizations that need to be read. Default ['pol00', 'pol11']
    Returns
    ----------
    large_arr: np.ndarray
        (npol,nchan,ntime) array of all data found between two timestramps
    ctime_arr: np.ndarray
        (ntime,) array of C timestamps corresponding to each of the ntime spectra in the data.
    """

    if chans is None:
        chans = np.arange(0,2048)
    nchans = len(chans)
    file1,idx1=butils.get_file_from_timestamp(init_t, dir_parent, 'd')
    file2,idx2=butils.get_file_from_timestamp(end_t, dir_parent, 'd')
    fnames=butils.time2fnames(butils.get_tstamp_from_filename(file1),butils.get_tstamp_from_filename(file2),dir_parent, "d" )
    print(f"Found {len(fnames)} files between {init_t} and {end_t}")
    #fnames is actually the path to the 10-digit direct spectra dir
    for tagnum, tag in enumerate(tags):
        print(f"Reading {tag}")
        new_fnames = [os.path.join(ff, tag + ".scio.bz2") for ff in fnames]
        files=[]
        for ff in new_fnames:
            print(ff)
            files.append(scio.read(ff))
        flags=[False]*len(files)
        if tagnum==0:
            #for the first tag, find out number of rows in files and allocate return array
            numrows=0
            for ii,myfile in enumerate(myfiles):
                if myfile is None:
                    flags[ii]=True
                    print(f"{myfile} is None")
                    continue
                if myfile.shape[0] < 10 or np.min(myfile) == 0:
                    flags[ii]=True
                    print(f"{myfile} has too few rows or has 0 power somewhere")
                    continue
                numrows += myfile.shape[0]  # determine total number of rows
            ctime_arr = np.zeros(numrows, dtype="float64")
            large_arr = np.zeros((len(tags), nchans, numrows), dtype="float64")
        #Now fill up the return arrays
        curidx=0
        for fnum, myfile in enumerate(myfiles):
            if flags[fnum]:
                continue
            ss = myfile.shape[0]
            tt = butils.get_tstamp_from_filename(fnames[fnum])
            large_arr[tagnum, :, curidx : curidx + ss] = myfile[:, chans].T
            ctime_arr[curidx : curidx + ss] = (
                tt + np.arange(ss) * 6.44
            )  # acclen 393216 = 6.44s. Each row of direct data is 6.44s long.
            curidx += ss
    return large_arr,ctime_arr

def get_present_files(t_start, t_end, ant_path_list, T_SCAN = 10, tolerance = 70):
    ''' 
    Given a path to several antenna, will tell you when data is present for each antenna

    Args:
        t_start (int): starting time of when we're scanning
        t_end (int): ending time of when we're scanning
        ant_path_list (list of strings): just the directory location for all antenna data
        T_SCAN (int): the time interval between scans, basically just the dt
        tolerance (int): maximum time that we tolerate between a point in time and the last present file

    Returns:
        arr (array): array of binary entries, of shape (ntimes, nants).
                     1 means data is present, 0 means it is absent
                     ntimes is in units of T_SCAN

        fig (figure): just visualizes arr, with time on y axis, in human time 
    '''

    t_start_human = datetime.fromtimestamp(t_start, tz=timezone.utc).strftime('%H:%M:%S, %d/%m')
    t_end_human = datetime.fromtimestamp(t_end, tz=timezone.utc).strftime('%H:%M:%S, %d/%m')
    ant_name_list = []
    ant2files = {}
    for i, ant_path in enumerate(ant_path_list):
        path_name = os.path.basename(ant_path)
        ant_name = i
        #ant_name = os.path.splitext(path_name)[0]
        print(ant_name)
        ant_name_list.append(ant_name)

    for ant_idx, ant_path in enumerate(ant_path_list):
        ant_name = ant_name_list[ant_idx]
        files_raw = []
        tstamps = []
        try:
            files_raw = time2fnames(t_start-tolerance, t_end, ant_path, 'f') #want to look backwards a bit from t_start also
        except FileNotFoundError:
            files_raw = []
        print(f'raw file ant {ant_name}', files_raw)

        for file in files_raw:
            filename = os.path.basename(file)
            tstamp_str = os.path.splitext(filename)[0]
            tstamp = int(tstamp_str)
            tstamps.append(tstamp)
        
        tstamps = np.array(tstamps)
        print(f'tstamps ant {ant_name}', tstamps)
        ant2files[ant_name] = tstamps

    rounded = int(np.ceil((t_end-t_start)/T_SCAN))*T_SCAN + t_start +1 #round UP to the nearest dt.
    times = np.arange(t_start, rounded, T_SCAN)
    times_human = [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%d/%m, %H:%M:%S') for t in times]
    arr = np.zeros((len(times), len(ant_name_list)))
    for t_idx, time in enumerate(times):
        for ant_idx, ant_name in enumerate(ant_name_list):
            data = ant2files[ant_name]
            if np.any((data >= time - tolerance) & (data <= time)):
                arr[t_idx][ant_idx] = 1

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"File Presence ({t_start_human}) to ({t_end_human})")

    im = ax.imshow(arr, aspect='auto', interpolation='none', cmap='Oranges')

    ax.set_xticks(range(len(ant_name_list)))
    ax.set_xticklabels(ant_name_list)

    step = len(times) // 10
    ax.set_yticks(range(0, len(times), step))
    ax.set_yticklabels([times_human[i] for i in range(0, len(times), step)])

    for x in range(1, arr.shape[1]):
        ax.axvline(x - 0.5, color='black', linewidth=0.5)

    return arr, fig

def get_simul_files(arr, time_start, dt, desired_ant_indices):
    ''' 
    Returns a list of [t_start, t_end] for times during which data is present for all antenna in desired_ant_indices

    Parameters
    ----------
    arr : array 
        array shape (ntimes, nants) of binary entries, with 1 meaning data is present and 0 meaning data is not present
    time_start: int 
        starting unix time of array, point at which we start counting
    dt : int 
        difference in time (seconds) between entries in array
    desired_ant_indices : list 
        the antenna (index on arr) for which we want data to be present. 
        If it's -1, it considers ALL antenna in arr.

    Returns
    -------
    runs : list
        list of two-element lists [t_start, t_end] which indicate the times of simultaneous continuous data in all desired antenna. 
    '''
    in_run = False
    runs = []
    current_time = time_start
    for row in arr:
        all_present = all(row[x] == 1 for x in desired_ant_indices)
        #all four cases:
        if (all_present) and (in_run):
            pass

        elif (not all_present) and (not in_run):
            pass

        elif (all_present) and (not in_run):
            in_run = True
            start_run = current_time
            
        elif (not all_present) and (in_run):
            in_run = False
            end_run = current_time - 20
            runs.append([start_run, end_run])

        else:
            raise ValueError('something broke!')
        
        current_time += dt

    if in_run:
        runs.append([start_run, current_time])

    return runs


def check_data_holes(t_start, t_end, dir, filesize=500001224, tol = 60, verbose=False, force_ts = False):
    '''
    This is a measure for abundance of caution when opening up files.
    Returns True if there is data missing, or any holes between the data.
    Returns False if there are no problems (i.e. no data holes)
    '''
    #start by trying to open up the files
    try:
        files, _ = get_init_info(t_start, t_end, dir, force_ts=force_ts)
    except Exception as e:
        print(e)
        print(f"literally zero files here")
        return True
    
    tstamps = []
    #now iterate through the present files
    for f in files:
        #check that the file is full
        size = os.path.getsize(f)
        if size != filesize:
            print('ERROR: Something wrong near file:', f)
            if not verbose:
                return True
        #add timestamp to list
        num = int(os.path.splitext(os.path.basename(f))[0])
        tstamps.append(num)
    #make into actual array
    tstamps = np.array(tstamps)
    #check enough space at front
    diff_to_start = tstamps[0]-t_start
    print('diff to start', diff_to_start)
    if diff_to_start>tol:
        print('ERROR: Something wrong at start')
        if not verbose:
            return True
    #check enough space at back
    diff_to_end = t_end -tstamps[-1]
    print('diff to end', diff_to_end)
    if diff_to_end>tol:
        print('ERROR: Something wrong at end')
        if not verbose:
            return True
    #check enough space between files
    diff_between = np.diff(tstamps)
    print('diff between', diff_between)
    if np.any(diff_between>tol):
        print('ERROR: Something wrong bewteen files')
        if not verbose:
            return True
    #otherwise all good
    print('No holes in data')
    return False



def get_windows_oneant(json_path, 
                        batch_start,
                        ant,
                        min_SNR,
                        min_nchunks,
                        max_nchunks=None,
                        interval=None
                        ):

    windows = []
    with open(json_path, 'r') as f:
        data = json.load(f) 
    ant_data = data[ant]

    for pulse in ant_data:
        t0, t1 = pulse["times"]
        if interval is not None:
            int_start = batch_start + interval[0]
            int_end   = batch_start + interval[1]
            if t1<int_start or t0>int_end:
                continue

        chunks = pulse["SNR, Chan, Sat"]
        if not chunks:
            continue

        snrs  = np.array([c[0] for c in chunks])
        chans = np.array([c[1] for c in chunks])
        sats  = np.array([c[2] for c in chunks])

        nchunks = len(snrs)
        chunk_dt = (t1 - t0) / nchunks

        above = snrs > min_SNR

        # ---------------------------------------------
        # Find contiguous True segments
        # ---------------------------------------------
        segments = []
        start = None

        for i, flag in enumerate(above):
            if flag:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i))
                    start = None

        if start is not None:
            segments.append((start, len(above)))

        if not segments:
            continue

        best_global_mean = -np.inf
        best_start = None
        best_len = 0

        # ---------------------------------------------
        # Evaluate each segment
        # ---------------------------------------------
        for s, e in segments:
            seg_len = e - s

            if seg_len < min_nchunks:
                continue

            segment_snrs = snrs[s:e]

            # Case 1: no max limit → take whole segment
            if max_nchunks is None or seg_len <= max_nchunks:
                seg_mean = segment_snrs.mean()

                if seg_mean > best_global_mean:
                    best_global_mean = seg_mean
                    best_start = s
                    best_len = seg_len

            else:
                # Case 2: need best subwindow of length max_nchunks
                k = max_nchunks

                # cumulative sum for fast sliding mean
                csum = np.cumsum(segment_snrs)
                csum = np.insert(csum, 0, 0)

                # compute window sums
                window_sums = csum[k:] - csum[:-k]
                idx = np.argmax(window_sums)

                seg_mean = window_sums[idx] / k

                if seg_mean > best_global_mean:
                    best_global_mean = seg_mean
                    best_start = s + idx
                    best_len = k

        if best_start is None:
            continue
        end_idx = best_start + best_len

        t_start = int(t0 + best_start * chunk_dt)
        t_end = int(t0 + end_idx * chunk_dt)

        windows.append({
            "antenna": ant,
            "sat": int(sats[best_start]),
            "channel": int(chans[best_start]),
            "t_start": t_start,
            "t_end": t_end,
            "len": t_end - t_start
        })
    return windows
