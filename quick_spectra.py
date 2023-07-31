"""A script to plot direct spectra products (pol0 mag, pol1 mag, pol0xpol1 mag and phase) for any given direct data folder.

Needs a directory that contains the following files:

**pol00.scio.bz2 pol01i.scio.bz2 pol01r.scio.bz2 pol11.scio.bz2**

Ideally, the directory name should be a timestamp (as is the convention for albatros data dumps), so that the output image has the timestamp in its name. This timestamp indicates the starting time for this direct data file. FPGA code is set to create a new file roughly every hour. Usual directory structure is something like:

`~/data_auto_cross/snap1/16272/1627202093/`

Use -l (ell) for logscale plots, which are more useful. -o for specifying output dir. Default output dire is ./ (dir from which code is called). Output plots indicate the localtime at telescope's location. By default, the timezone of the telescope is assumed to be US/Eastern. Timezone can be specified as shown below.

`python quick_spectra.py ~/1627202093/ -o ~/outputs/ -l -tz US/Pacific`

*All scripts support -h option for help.*
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from scio import scio
import argparse
import pytz
import datetime as dt


def _parse_slice(s):
    a = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*a)


def get_slice(s, acctime):
    tstart, tstop, tstep = s.start, s.stop, s.step
    if tstart is not None:
        tstart = int(np.floor(tstart * 60 / acctime))
    if tstop is not None:
        tstop = int(np.floor(tstop * 60 / acctime))
    if tstep is not None:
        tstep = int(np.floor(tstep * 60 / acctime))
    return slice(tstart, tstop, tstep)


def get_acctime(fpath):
    dat = np.fromfile(fpath, dtype="uint32")
    diff = np.diff(dat)
    acctime = np.mean(
        diff[(diff > 0) & (diff < 100)]
    )  # sometimes timestamps are 0, which causes diff to be huge. could also use np. median
    return acctime


def get_vmin_vmax(data_arr):
    """
    Automatically gets vmin and vmax for colorbar
    """
    # print("shape of passed array", data_arr.shape, data_arr.dtype)
    xx = np.ravel(data_arr).copy()
    med = np.percentile(xx, 50)
    # print(med, "median")
    u = np.percentile(xx, 99)
    b = np.percentile(xx, 1)
    xx_clean = xx[(xx <= u) & (xx >= b)]  # remove some outliers for better plotting
    stddev = np.std(xx_clean)
    vmin = max(med - 2 * stddev, 10**7)
    vmax = med + 2 * stddev
    if vmin > vmax:
        vmin = 10 ** (np.log10(vmin) - 1)
    print(vmin, vmax)
    # print("vmin, vmax are", vmin, vmax)
    return vmin, vmax


if __name__ == "__main__":
    "Example usage: python quick_spectra.py ~/data_auto_cross/16171/1617100000"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Auto/cross-spectra location. Ex: ~/data_auto_cross/16171/161700000",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./", help="Output directory for plots"
    )
    parser.add_argument("-l", "--logplot", action="store_true", help="Plot in logscale")
    parser.add_argument(
        "-c", "--common", action="store_true", help="Common colorbar for both pols"
    )
    parser.add_argument("-s", "--show", action="store_true", help="Show final plot")
    parser.add_argument(
        "-tz",
        "--timezone",
        type=str,
        default="US/Eastern",
        help="Valid timezone of the telescope recognized by pytz. E.g. US/Eastern. Default is US/Eastern.",
    )
    parser.add_argument(
        "-sl",
        "--tslice",
        type=_parse_slice,
        help="Slice on time axis to restrict plot to. Format: -sl=tmin:tmax for timin, tmax in minutes",
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
    args = parser.parse_args()

    # data_dir = pathlib.Path(args.data_dir)
    # output_dir = pathlib.Path(args.output_dir)

    pol00 = scio.read(os.path.join(args.data_dir, "pol00.scio.bz2"))
    pol11 = scio.read(os.path.join(args.data_dir, "pol11.scio.bz2"))
    pol01r = scio.read(os.path.join(args.data_dir, "pol01r.scio.bz2"))
    pol01i = scio.read(os.path.join(args.data_dir, "pol01i.scio.bz2"))
    acctime = get_acctime(os.path.join(args.data_dir, "time_gps_start.raw"))

    fmin, fmax = 0, 125

    if args.fmin:
        fmin = args.fmin
    if args.fmax:
        fmax = args.fmax
    cstart = int(np.floor(fmin / (250 / 4096)))
    cend = int(np.floor(fmax / (250 / 4096)))

    tstart = 0  # for myext below
    if args.tslice is not None:
        tstart = int(args.tslice.start)

    if args.tslice:
        # convert tslice in minutes to samps
        tslice = get_slice(args.tslice, acctime)
        pol00 = pol00[tslice, cstart:cend]
        pol11 = pol11[tslice, cstart:cend]
        pol01r = pol01r[tslice, cstart:cend]
        pol01i = pol01i[tslice, cstart:cend]
    else:
        # Remove starting row since it's sometimes bad :(
        pol00 = pol00[1:, cstart:cend]
        pol11 = pol11[1:, cstart:cend]
        pol01r = pol01r[1:, cstart:cend]
        pol01i = pol01i[1:, cstart:cend]

    # Add real and image for pol01
    pol01 = pol01r + 1j * pol01i

    freq = np.linspace(fmin, fmax, np.shape(pol00)[1])

    pol00_med = np.median(pol00, axis=0)
    pol11_med = np.median(pol11, axis=0)
    pol00_mean = np.mean(pol00, axis=0)
    pol11_mean = np.mean(pol11, axis=0)
    pol00_max = np.max(pol00, axis=0)
    pol11_max = np.max(pol11, axis=0)
    pol00_min = np.min(pol00, axis=0)
    pol11_min = np.min(pol11, axis=0)

    if args.vmin is None and args.vmax is None:
        vmin, vmax = get_vmin_vmax(pol00)
        vmin2, vmax2 = get_vmin_vmax(pol11)

    pmax = np.max(pol00)
    axrange = [fmin, fmax, 0, pmax]

    if args.common:
        vmin_common = min(vmin, vmin2)
        vmax_common = max(vmax, vmax2)
        vmin = vmin_common
        vmax = vmax_common
        vmin2 = vmin_common
        vmax2 = vmax_common

    if args.logplot == True:
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
        pmax = np.log10(pmax)
        axrange = [fmin, fmax, 6.5, pmax]

    print("Estimated accumulation time from timestamp file: ", acctime)
    tot_minutes = int(np.ceil(acctime * pol00.shape[0] / 60))
    myext = np.array([fmin, fmax, tstart + tot_minutes, tstart])

    plt.figure(figsize=(18, 10), dpi=200)

    plt.subplot(2, 3, 1)
    plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect="auto", extent=myext)
    plt.title("pol00")
    cb00 = plt.colorbar()
    plt.xlabel("Frequency (MHz)")
    cb00.ax.plot([0, 1], [7.0] * 2, "w")
    cb00.ax.set_ylabel("Uncalibrated log(Power)")

    plt.subplot(2, 3, 4)
    plt.imshow(pol11, vmin=vmin2, vmax=vmax2, aspect="auto", extent=myext)
    plt.title("pol11")
    cb = plt.colorbar()
    cb.ax.set_ylabel("Uncalibrated log(Power)")
    plt.xlabel("Frequency (MHz)")

    plt.subplot(2, 3, 2)
    plt.title("Basic stats for frequency bins")
    plt.plot(freq, pol00_max, "r-", label="Max")
    plt.plot(freq, pol00_min, "b-", label="Min")
    plt.plot(freq, pol00_mean, "k-", label="Mean")
    plt.plot(freq, pol00_med, color="#666666", linestyle="-", label="Median")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("pol00")
    plt.ylim(vmin, vmax)
    plt.xlim(fmin, fmax)

    plt.subplot(2, 3, 5)
    plt.plot(freq, pol11_max, "r-", label="Max")
    plt.plot(freq, pol11_min, "b-", label="Min")
    plt.plot(freq, pol11_mean, "k-", label="Mean")
    plt.plot(freq, pol11_med, color="#666666", linestyle="-", label="Median")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("pol11")
    plt.ylim(vmin2, vmax2)
    plt.xlim(fmin, fmax)
    plt.legend(loc="lower right", fontsize="small")

    plt.subplot(2, 3, 3)
    plt.imshow(np.log10(np.abs(pol01)), vmin=3, vmax=8, aspect="auto", extent=myext)
    plt.title("pol01 magnitude")
    cb = plt.colorbar()
    cb.ax.set_ylabel("Uncalibrated log(Power)")
    plt.xlabel("Frequency (MHz)")

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
    cb = plt.colorbar()
    cb.ax.set_ylabel("Phase (rad)")
    plt.xlabel("Frequency (MHz)")

    args.data_dir = os.path.abspath(args.data_dir)
    timestamp = args.data_dir.split("/")[-1]
    mytz = pytz.timezone(args.timezone)
    utctime = dt.datetime.fromtimestamp(
        int(timestamp), tz=pytz.utc
    )  # this might be a safer way compared to passing tz directly to fromtimestamp
    localtimestr = utctime.astimezone(tz=mytz).strftime("%b-%d %H:%M:%S")
    plt.suptitle(f"Minutes since {localtimestr} localtime. File ctime {timestamp}")

    outfile = os.path.normpath(args.output_dir + "/" + timestamp + "_quick" + ".png")
    plt.savefig(outfile)
    print("Wrote " + outfile)
    if args.show == True:
        plt.show()
    plt.close()
