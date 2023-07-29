import os, sys
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scio import scio # module scio with class called scio, packeged as pbio, make sure `pip import pbio` not `scio`
import argparse
import time, re
from datetime import datetime 
import matplotlib.dates as mdates
from multiprocessing import Pool
from functools import partial
import pytz

if __name__=="__main__":
    import SNAPfiletools as sft
else:
    import albatros_analysis.SNAPfiletools as sft


def get_ts_from_name(f):
    return int(f.split("/")[-1])


def get_localtime_from_UTC(tstamp, mytz):
    return datetime.fromtimestamp(int(tstamp), tz=pytz.utc).astimezone(tz=mytz)


# ============================================================
def get_data_arrs(data_dir: str, ctime_start: str, ctime_stop: str, chunk_time, blocklen, mytz):
    """
    Given the path to a Big data directory (i.e. directory contains the directories
    labeled by the first 5 digits of the ctime date), gets all the data in some time interval.

    Parameters:
    -----------

    data_dir: str
        path to data directory

    ctime_start, ctime_stop: str
        desired start and stop time in ctime

    chunk_time: ??
    
    blocklen: (probably int)
    
    mytz: (what kind of object is this? probably pytz.tzfile)
        Timezone (of the dish at collection?)

    Returns:
    --------

    cimte_start, ctime_stop: int
        start and stop times in ctime

    pol00,pol11,pol01r,pol01i: array
        2D arrays containing the data for given time interval for autospectra
        as well as cross spectra. pol00 corresponds to adc0 and pol11 to adc3
    """
    print("\n################### READING DATA ###################")
    print(f"Files requested between timestamps {ctime_start} to {ctime_stop}")
    print(
        f"Corresponding UTC time: {datetime.utcfromtimestamp(ctime_start)} to {datetime.utcfromtimestamp(ctime_stop)}"
    )

    # all the dirs between the timestamps. read all, append, average over chunk length
    data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)
    print("total data subdirs", len(data_subdirs))
    print("First and last subdirs:", data_subdirs[0], data_subdirs[-1])
    data_subdirs.sort()

    if len(data_subdirs) == 0:
        print("NOTHING WAS READ. CHECK TSTAMPS")
        sys.exit(1)

    # rough estimate of number of rows we'll read
    nrows_guess = len(data_subdirs) * ((int(3600 / chunk_time / blocklen) + 1) + 1)
    # print("Starting with a guess of ", nrows_guess)
    print("guessed rows", nrows_guess)
    pol00 = np.zeros((nrows_guess + 500, 2048))

    nrows = 0

    t1 = time.time()
    new_dirs = [d + "/pol00.scio.bz2" for d in data_subdirs]
    datpol00 = scio.read_files(new_dirs)
    new_dirs = [d + "/pol11.scio.bz2" for d in data_subdirs]
    datpol11 = scio.read_files(new_dirs)
    new_dirs = [d + "/pol01r.scio.bz2" for d in data_subdirs]
    datpol01r = scio.read_files(new_dirs)
    new_dirs = [d + "/pol01i.scio.bz2" for d in data_subdirs]
    datpol01i = scio.read_files(new_dirs)
    print(time.time() - t1, f"Read {len(data_subdirs)} files")

    # average everything if blocklen>1
    # print("first row must be in:", data_subdirs[0])

    myavgfunc = partial(get_avg, block=blocklen)
    if blocklen > 1:
        t1 = time.time()
        with Pool(os.cpu_count()) as p:
            avgpol00 = p.map(myavgfunc, datpol00)
            avgpol11 = p.map(myavgfunc, datpol11)
            avgpol01r = p.map(myavgfunc, datpol01r)
            avgpol01i = p.map(myavgfunc, datpol01i)
        print(time.time() - t1, "averaged everything")
    else:
        avgpol00 = datpol00
        avgpol11 = datpol11
        avgpol01r = datpol01r
        avgpol01i = datpol01i

    print(len(avgpol00))
    t1 = time.time()
    tstart = 0
    tend = 0
    for i, d in enumerate(avgpol00):
        # print("Mean, median are", np.mean(d,axis=0),np.median(d,axis=0))
        print("working on", data_subdirs[i])
        if d is None:
            continue
        if i == 0:
            pol00[: d.shape[0]] = d
            nrows += d.shape[0]
            ts = get_ts_from_name(data_subdirs[i])

            tstart = ts  # save starting time for user output
            ts = ts + d.shape[0] * chunk_time * blocklen
            continue
        newts = get_ts_from_name(data_subdirs[i])
        diff = int((newts - ts) / chunk_time / blocklen)
        # each cell in the plot represents a minimum time of blocklen * chunktime.
        # That's the time resolution for the plot. Can't catch gaps < resolution.
        if diff > 0:
            print(f"significant diff b/w files {tstart} and {newts} of:", diff, "rows")
            pol00[nrows : nrows + diff, :] = np.nan
            pol00 = np.append(pol00, np.zeros((diff, 2048)), axis=0)
            nrows += diff
        # print(nrows, d.shape)
        # print(nrows,nrows+d.shape[0],pol00.shape,"heh")
        pol00[nrows : nrows + d.shape[0], :] = d
        nrows += d.shape[0]
        # print("reading", data_subdirs[i], "with size ", d.shape[0], "NROWS", oldnrows,nrows)
        tstart = newts
        ts = newts + d.shape[0] * chunk_time * blocklen
    tend = ts
    # print("HERE")
    # once we have pol00, we know the exact size. use it
    tstart = get_ts_from_name(
        data_subdirs[0]
    )  # tstart was replaced above for missing gap info
    print("############################################################")
    print(
        f"First file at: {tstart}, Last file at: {get_ts_from_name(data_subdirs[-1])}"
    )
    print(f"Plotting all data starting {tstart} and ending {int(tend)}")
    ts, te = list(map(partial(get_localtime_from_UTC, mytz=mytz), [tstart, tend]))
    print(
        f"In Local time: {ts.strftime('%b-%d %H:%M:%S')} to {te.strftime('%b-%d %H:%M:%S')} in {mytz.zone}"
    )
    print("Final nrows:", nrows)

    pol00 = pol00[:nrows].copy()
    # print(pol00.shape)

    pol11 = np.zeros((nrows, 2048))
    pol01r = np.zeros((nrows, 2048))
    pol01i = np.zeros((nrows, 2048))

    nrows = 0
    for i in range(len(avgpol00)):
        if (avgpol11[i] is None) or (avgpol01i[i] is None) or (avgpol01r[i] is None):
            continue
        if i == 0:
            r = avgpol11[i].shape[0]
            pol11[:r, :] = avgpol11[i]
            pol01r[:r, :] = avgpol01r[i]
            pol01i[:r, :] = avgpol01i[i]
            nrows += r
            ts = get_ts_from_name(data_subdirs[i]) + r * chunk_time * blocklen
            continue
        newts = get_ts_from_name(data_subdirs[i])
        diff = int((newts - ts) / chunk_time / blocklen)
        if diff > 0:
            pol11[nrows : nrows + diff, :] = np.nan
            pol01r[nrows : nrows + diff, :] = np.nan
            pol01i[nrows : nrows + diff, :] = np.nan
            nrows += diff
        r = avgpol11[i].shape[0]
        pol11[nrows : nrows + r, :] = avgpol11[i]
        pol01r[nrows : nrows + r, :] = avgpol01r[i]
        pol01i[nrows : nrows + r, :] = avgpol01i[i]
        nrows += r
        ts = newts + r * chunk_time * blocklen

    t2 = time.time()
    # print('Time taken to concatenate data:',t2-t1)
    # print("pol00, pol11,pol01r, pol01i shape:", pol00.shape,pol11.shape,pol01r.shape,pol01i.shape)
    pol00 = np.ma.masked_invalid(pol00)
    pol11 = np.ma.masked_invalid(pol11)
    pol01r = np.ma.masked_invalid(pol01r)
    pol01i = np.ma.masked_invalid(pol01i)
    return pol00, pol11, pol01r, pol01i, tstart, tend


def get_avg(arr, block=10):
    """
    Fast average array over a given block size.
    """
    if arr is None:
        return None
    iters = arr.shape[0] // block
    leftover = arr.shape[0] % block
    # print(iters,leftover)
    nrows = iters + int(leftover > 0)
    ncols = arr.shape[1]
    newarr = np.zeros((nrows, ncols), dtype=arr.dtype)
    cmp1 = np.median(arr[0, :]) / np.median(
        arr[1, :]
    )  # temporary fix, assuming only first row is expected to be faulty
    if cmp1 > 1e2:
        # skip first row before averaging for first block
        newarr[0, :] = np.mean(arr[1:block, :], axis=0)
        newarr[1:iters, :] = np.mean(
            arr[block : iters * block, :].reshape(-1, block, ncols), axis=1
        )
    else:
        # print(f"Shape of passed arr {arr.shape} and shape of new arr {newarr.shape}")
        newarr[:iters, :] = np.mean(
            arr[: iters * block, :].reshape(-1, block, ncols), axis=1
        )
    if leftover:
        newarr[iters, :] = np.mean(arr[iters * block :, :], axis=0)
    return newarr


def get_stats(data_arr):
    """
    Given a 2D array containing some data chunk, returns the
    min, median, mean, and max over that chunk.
    """
    # print("WHERE MIN ZERO",np.where(np.min(data_arr,axis=0)==0))
    # print("WHERE MEDIAN ZERO",np.where(np.median(data_arr,axis=0)==0))
    # print("MEDIAN",np.median(data_arr,axis=0))
    # print("MEDIAN MA",np.ma.median(data_arr,axis=0))
    if logplot:
        stats = {
            "min": np.log10(np.ma.min(data_arr, axis=0)),
            "median": np.log10(np.ma.median(data_arr, axis=0)),
            "mean": np.log10(np.ma.mean(data_arr, axis=0)),
            "max": np.log10(np.ma.max(data_arr, axis=0)),
        }
    else:
        stats = {
            "min": np.ma.min(data_arr, axis=0),
            "median": np.ma.median(data_arr, axis=0),
            "mean": np.ma.mean(data_arr, axis=0),
            "max": np.ma.max(data_arr, axis=0),
        }
    return stats


def get_vmin_vmax(data_arr):
    """
    Automatically gets vmin and vmax for colorbar
    """
    # print("shape of passed array", data_arr.shape, data_arr.dtype)
    xx = data_arr[~data_arr.mask].data
    med = np.percentile(xx, 50)
    # print(med, "median")
    u = np.percentile(xx, 99)
    b = np.percentile(xx, 1)
    xx_clean = xx[(xx <= u) & (xx >= b)]  # remove some outliers for better plotting
    stddev = np.std(xx_clean)
    vmin = max(med - 2 * stddev, 10**7)
    vmax = med + 2 * stddev
    # print("vmin, vmax are", vmin, vmax)
    return vmin, vmax


def get_ylim_times(t_i, t_f):
    """
    Gets the y limits in matplotlib's date format for a given initial time
    and final time. t_i and t_f must be given in ctime
    """
    # getlocaltime = lambda tstamp: datetime.fromtimestamp(int(tstamp),tz=pytz.utc).astimezone(tz=mytz)
    y_lims = list(map(datetime.utcfromtimestamp, [t_i, t_f]))
    y_lims_plt = mdates.date2num(y_lims)
    # date2num is NOT tz aware.
    # will return same value regardless of tz of passed datetime object.
    # pass tz to formatter and tick locators
    return y_lims_plt


# ================= plotting functions =======================
def full_plot(data_arrs, mytz, chunk_time):
    """
    Makes a plot that contains autospectra waterfalls for each pol, as well
    as some statistics (min,max,med,mean spectra), and cross spectra
    """
    global vmin, vmax, vmin2, vmax2
    pol00, pol11, pol01, tstart, tend = data_arrs
    print("Generating stats for pol00")
    pol00_stats = get_stats(pol00)
    print("Generating stats for pol11")
    pol11_stats = get_stats(pol11)
    # print("WHERE POL11 ZERO", np.where(pol11==0))
    # print("Pol00 median", pol00_stats['median'])
    if logplot is True:
        pol00 = np.log10(pol00)
        pol11 = np.log10(pol11)

    if rescale:
        scaling_pol00 = np.tile(pol00_stats["median"], pol00.shape[0]).reshape(
            *pol00.shape
        )
        scaling_pol11 = np.tile(pol11_stats["median"], pol11.shape[0]).reshape(
            *pol11.shape
        )
        pol00[:] = 10 * (
            pol00 - scaling_pol00
        )  # - instead of / for type 2 scaling: log(pol00/pol00_median)
        pol11[:] = 10 * (pol11 - scaling_pol11)
        vmin = -1
        vmax = 1
        vmin2 = vmin
        vmax2 = vmax

    y_extent = get_ylim_times(tstart, tend)
    ticks = np.linspace(y_extent[0], y_extent[1], 10)
    # print(y_extent)

    myext = np.array([freq[0], freq[-1], y_extent[1], y_extent[0]])

    plt.figure(figsize=(18, 10), dpi=200)
    plt.subplot(2, 3, 1)

    plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect="auto", extent=myext)
    plt.title("pol00")
    cb00 = plt.colorbar()
    cb00.ax.set_ylabel("Uncalibrated log(power)", rotation=90)
    plt.xlabel("Frequency (MHz)")
    plt.yticks(ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(datetimefmt)

    # this makes the code slow on Lab laptop.

    # nticks = 15 #desired number of ticks on the plot
    # hourinterval = int(pol00.shape[0]*chunk_time*blocksize/3600/nticks)
    # locator=mdates.HourLocator(interval=hourinterval,tz=mytz)
    # ax.yaxis.set_major_locator(locator)

    plt.subplot(2, 3, 4)
    plt.imshow(pol11, vmin=vmin2, vmax=vmax2, aspect="auto", extent=myext)
    plt.title("pol11")
    plt.xlabel("Frequency (MHz)")
    cb00 = plt.colorbar()
    cb00.ax.set_ylabel("Uncalibrated log(power)", rotation=90)
    plt.yticks(ticks)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(datetimefmt)
    # ax.yaxis.set_major_locator(locator)

    plt.subplot(2, 3, 2)
    plt.title("Median power in frequency bins")
    plt.plot(freq, pol00_stats["max"], "r-", label="Max")
    plt.plot(freq, pol00_stats["min"], "b-", label="Min")
    plt.plot(freq, pol00_stats["mean"], "k-", label="Mean")
    plt.plot(
        freq, pol00_stats["median"], color="#666666", linestyle="-", label="Median"
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("pol00")
    plt.legend(loc="lower right", fontsize="small")
    plt.ylim(vmin, vmax)

    plt.subplot(2, 3, 5)
    plt.plot(freq, pol11_stats["max"], "r-", label="Max")
    plt.plot(freq, pol11_stats["min"], "b-", label="Min")
    plt.plot(freq, pol11_stats["mean"], "k-", label="Mean")
    plt.plot(
        freq, pol11_stats["median"], color="#666666", linestyle="-", label="Median"
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("pol11")
    plt.ylim(vmin2, vmax2)

    plt.legend(loc="lower right", fontsize="small")

    plt.subplot(2, 3, 3)
    plt.imshow(np.log10(np.abs(pol01)), vmin=3, vmax=8, aspect="auto", extent=myext)
    plt.title("pol01 magnitude")
    plt.xlabel("Frequency (MHz)")
    cb00 = plt.colorbar()
    cb00.ax.set_ylabel("Uncalibrated power", rotation=90)
    plt.gca().set_yticklabels([])

    plt.subplot(2, 3, 6)
    plt.imshow(
        np.angle(pol01),
        vmin=-np.pi,
        vmax=np.pi,
        aspect="auto",
        extent=myext,
        cmap="RdBu",
    )
    plt.title("pol01 phase")
    plt.xlabel("Frequency (MHz)")
    cb00 = plt.colorbar()
    cb00.ax.set_ylabel("Radian", rotation=90)
    plt.gca().set_yticklabels([])

    range_localtime = list(
        map(partial(get_localtime_from_UTC, mytz=mytz), [tstart, tend])
    )
    print("start and end times are", tstart, tend)
    plt.suptitle(
        f'Plotting {range_localtime[0].strftime("%b-%d %H:%M:%S")} to {range_localtime[1].strftime("%b-%d %H:%M:%S")} in {mytz.zone} \nAveraged over {blocksize} chunks ~ {blocksize*chunk_time/60:4.2f} minutes.'
    )
    plt.tight_layout()
    outfile = os.path.join(
        outdir,
        "direct_overnight_output"
        + "_"
        + str(ctime_start)
        + "_"
        + str(ctime_stop)
        + ".jpg",
    )
    plt.savefig(outfile)

    print("Wrote " + outfile)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.set_usage('python plot_overnight_data.py <data directory> <start time as YYYYMMDD_HHMMSS or ctime> <stop time as YYYYMMDD_HHMMSS or ctime> [options]')
    # parser.set_description(__doc__)
    parser.add_argument("data_dir", type=str, help="Direct data directory")
    parser.add_argument(
        "time_start", type=str, help="Start time YYYYMMDD_HHMMSS or ctime. Both in UTC."
    )
    parser.add_argument(
        "time_stop", type=str, help="Stop time YYYYMMDD_HHMMSS or ctime. Both in UTC."
    )
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default=".",
        help="Output plot directory [default: .]",
    )

    parser.add_argument(
        "-a",
        "--avglen",
        dest="blocksize",
        default=10,
        type=int,
        help="number of chunks (rows) of direct spectra to average over. One chunk is roughly 6 seconds.",
    )
    parser.add_argument(
        "-n",
        "--acclen",
        dest="acclen",
        type=int,
        default=393216,
        help="Accumulation length to calculate accumulation time. Default 393216 ~ 6.44s",
    )
    parser.add_argument(
        "-l",
        "--logplot",
        dest="logplot",
        default=True,
        action="store_true",
        help="Plot in logscale",
    )
    parser.add_argument(
        "-p",
        "--plottype",
        dest="plottype",
        default="full",
        type=str,
        help="Type of plot to generate. 'full': pol00 and pol11 waterfall autospectra, min/max/mean/med autospectra, waterfall cross spectra. 'waterfall': same as 1, but no stats",
    )
    parser.add_argument(
        "-tz",
        "--timezone",
        type=str,
        default="US/Eastern",
        help="Valid timezone of the telescope recognized by pytz. E.g. US/Eastern. Default is US/Eastern.",
    )
    parser.add_argument(
        "-vmi",
        "--vmin",
        dest="vmin",
        default=None,
        type=float,
        help="minimum for colorbar. if nothing is specified, vmin is automatically set",
    )
    parser.add_argument(
        "-vma",
        "--vmax",
        dest="vmax",
        default=None,
        type=float,
        help="maximum for colorbar. if nothing is specified, vmax is automatically set",
    )
    parser.add_argument(
        "-d",
        "--datetimefmt",
        dest="datetimefmt",
        default="%m/%d %H:%M",
        type=str,
        help="Format for dates on axes of plots",
    )
    parser.add_argument(
        "-fma",
        "--fmax",
        dest="fmax",
        default=None,
        type=float,
        help="maximum for frequency to plot",
    )
    parser.add_argument(
        "-fmi",
        "--fmin",
        dest="fmin",
        default=None,
        type=float,
        help="minimum for frequency to plot",
    )
    parser.add_argument(
        "-r",
        "--rescale",
        dest="rescale",
        default=False,
        action="store_true",
        help="Rescale autospectra using median power",
    )
    parser.add_argument(
        "-c", "--common", action="store_true", help="Common colorbar for both pols"
    )
    args = parser.parse_args()

    # =============== defining some global variables ===============#
    global freq, timezone, logplot, vmin, vmax, vmin2, vmax2, ctime_start, ctime_stop, blocksize, outdir, datetimefmt, rescale

    timezone = args.timezone
    vmin = args.vmin
    vmax = args.vmax
    logplot = args.logplot
    blocksize = args.blocksize
    outdir = args.outdir
    rescale = args.rescale
    mytz = pytz.timezone(args.timezone)
    datetimefmt = mdates.DateFormatter(
        args.datetimefmt, tz=mytz
    )  # formatter needs to be tz aware

    # =============================================================#

    # figuring out if human time or ctime was passed with pattern matching
    rx_human = re.compile(r"^\d{8}_\d{6}$")
    rx_ctime = re.compile(r"^\d{10}$")
    m1 = rx_human.search(args.time_start)
    m2 = rx_ctime.search(args.time_start)
    if m1:
        ctime_start = sft.timestamp2ctime(args.time_start)
        ctime_stop = sft.timestamp2ctime(args.time_stop)
    elif m2:
        ctime_start = int(args.time_start)
        ctime_stop = int(args.time_stop)
    else:
        raise ValueError("INVALID time format entered.")

    chunk_time = args.acclen * 4096 / 250e6

    # ================= reading data =================#
    pol00, pol11, pol01r, pol01i, tstart, tend = get_data_arrs(
        args.data_dir, ctime_start, ctime_stop, chunk_time, args.blocksize, mytz
    )
    # import sys
    # sys.exit(0)

    fmin, fmax = 0, 125
    if args.fmin:
        fmin = args.fmin
    if args.fmax:
        fmax = args.fmax

    cstart = int(np.floor(fmin / (250 / 4096)))
    cend = int(np.floor(fmax / (250 / 4096)))

    pol00 = pol00[:, cstart:cend]
    pol11 = pol11[:, cstart:cend]
    pol01r = pol01r[:, cstart:cend]
    pol01i = pol01i[:, cstart:cend]

    pol01 = pol01r + 1j * pol01i
    freq = np.arange(cstart, cend) * 250 / 4096  # 125 MHz is max frequency

    # ============ setting vmin and vmax ============#
    # setting vmin and vmax
    if vmin == None and vmax == None:
        vmin, vmax = get_vmin_vmax(pol00)
        vmin2, vmax2 = get_vmin_vmax(pol11)
        if logplot == True:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)
            vmin2 = np.log10(vmin2)
            vmax2 = np.log10(vmax2)
    if args.common:
        vmin = min(vmin, vmin2)
        vmax = max(vmax, vmax2)
        vmin2 = vmin
        vmax2 = vmax

    # ============ and finally: plotting! ============#
    if args.plottype == "full":
        full_plot([pol00, pol11, pol01, tstart, tend], mytz, chunk_time)


