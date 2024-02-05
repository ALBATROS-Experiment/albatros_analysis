"""The purpose of this script is to let you analyze 4 bit baseband data from a single antenna. It will compute and save pol00, pol11, and pol01 averaged over whatever acclen user specifies. [question from steve to mohan: in what units is acclen measured?]

It requires a parent directory (which has 5 digit timestamp folders [question: what are those digits/ what do they represent?]) for baseband files, a starting timestamp, and an accumulation length. By default it will run for 560 chunks: that's roughly 1 hour for an accumulation length of 393216. You have the option to manually specify the number of chunks, or an end timestamp. In the latter case, the number of chunks will be calculated using (t_end - t_start)/t_acclen. Both cases are shown below. 

You also have the option to control exactly what channels you'd like to examine. In order to specify the channels, use the `-c` option followed by the **indices (not channel numbers)** of starting and end channels. **The channels must be contiguous and the indices must follow numpy convention.** E.g., say you have 100 channels around 40 MHz, and 50 channels around 113 MHz (ORBCOMM), and you'd like to examine ORBCOMM, use: `-c 100 150`

If you wanted to look at a single channel, you'd use something like: `-c 100 101`. This will only unpack the channel at index 100. 

Example usage: 

`python autocorravg.py ~/baseband/snap1/ 1627202094 393216 -t 1627205694 -c 100 150`

`python autocorravg.py ~/baseband/snap1/ 1627202094 393216 -n 560 -c 100 150`

**How to read .npz file tht's generated?**
The npz file stores data and masks for pol00, pol11, and pol01, and the exact channel numbers that were used (as per the indices specified). It can be read as follows

```python
with np.load("~/pols_4bit.npz") as npz:
    pol01 = np.ma.MaskedArray(npz['data01'],npz['maskp01'])
    # pol00 = np.ma.MaskedArray(npz['data00'],npz['maskp00'])
    # pol11 = np.ma.MaskedArray(npz['data11'],npz['maskp11'])
    channels = npz['channels'].copy()
```

The format of the output file is: 

`pols_4bit_{time_start}_{acclen}_{nchunks}_{chanstart}_{chanend}.npz`

By default `chanstart=0` and `chanend=None`, in order to process all channels. So if a file is name "...\*0_None.png\*" it means all channels in the baseband files were included.
"""



import numpy as np
import time
import argparse

if __name__=="__main__":
    from correlations import baseband_data_classes as bdc
    from correlations import correlations as cr
    from utils import baseband_utils as butils
else:
    from .correlations import baseband_data_classes as bdc
    from .correlations import correlations as cr
    from .utils import baseband_utils as butils

# TODO: not sure about types going into and out of get_avg_fast
# best guess out is tuple[ndarray x 3, list]


def get_avg_fast(
    path: str,
    init_t: int,
    end_t: int,
    acclen: int,
    nchunks: int,
    chanstart=0,
    chanend=None,
):
    """Time-averages power in each channel.
    
    TODO: what does this function do? make sure above is correct.

    Parameters
    ----------
    path: str
        Path to baseband data.
    init_t: int
        The start time of ... (in what units/format)
    end_t: int
        The end tine of ... (in what units/format)
    acclen: int
        Accumulation length for averaging (in what units?).
    chanstart: int
        The index of the lowest frequency channel in selection.
        Defaults to 0.
    chanend: int
        Index of the highest frequency channel in selection. Defaults
        to None -> up to highest freq channel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list?]
        What?
    """

    files, idxstart = butils.get_init_info(init_t, end_t, path)
    fileidx = 0
    print("Starting at: ", idxstart, "in filenum: ", fileidx)
    print(files[fileidx])

    ant1 = bdc.BasebandFileIterator(
        files,
        fileidx,
        idxstart,
        acclen,
        nchunks=nchunks,
        chanstart=chanstart,
        chanend=chanend,
    )
    if ant1.obj.bit_mode != 4:
        raise NotImplementedError(
            f"BIT MODE {ant1.obj.bit_mode} IS NOT SUPPORTED BY THIS SCRIPT. Do you want to use autocorravg1bit.py?"
        )
    ncols = ant1.obj.chanend - ant1.obj.chanstart
    pol00 = np.zeros((nchunks, ncols), dtype="float64", order="c")
    pol11 = np.zeros((nchunks, ncols), dtype="float64", order="c")
    pol01 = np.zeros((nchunks, ncols), dtype="complex64", order="c")
    j = ant1.spec_num_start
    m = ant1.spec_num_start
    st = time.time()
    for i, chunk in enumerate(ant1):
        t1 = time.time()
        pol00[i, :] = cr.avg_autocorr_4bit(chunk["pol0"], chunk["specnums"])
        pol11[i, :] = cr.avg_autocorr_4bit(chunk["pol1"], chunk["specnums"])
        pol01[i, :] = cr.avg_xcorr_4bit(chunk["pol0"], chunk["pol1"], chunk["specnums"])
        t2 = time.time()
        print("time taken for one loop", t2 - t1)
        j = ant1.spec_num_start
        print("After a loop spec_num start at:", j, "Expected at", m + (i + 1) * acclen)
        print(i + 1, "CHUNK READ")
    print("Time taken final:", time.time() - st)
    pol00 = np.ma.masked_invalid(pol00)
    pol11 = np.ma.masked_invalid(pol11)
    pol01 = np.ma.masked_invalid(pol01)
    return pol00, pol11, pol01, ant1.obj.channels





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Parent data directory. Should have 5 digit time folders.",
    )
    parser.add_argument("time_start", type=float, help="Start timestamp ctime")
    parser.add_argument("acclen", type=int, help="Accumulation length for averaging")
    parser.add_argument(
        "-n",
        "--nchunks",
        dest="nchunks",
        type=int,
        default=560,
        help="Number of chunks in output file. If stop time is specfied this is overwritten. Default 560 ~ 1 hr.",
    )
    parser.add_argument(
        "-t",
        "--time_stop",
        dest="time_stop",
        type=float,
        default=False,
        help="Stop time. Overwrites nchunks if specified",
    )
    parser.add_argument(
        "-c", "--chans", type=int, nargs=2, help="Indices of start and end channels."
    )
    parser.add_argument("-l", "--logplot", action="store_true", help="Plot in logscale")
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default="/scratch/s/sievers/mohanagr/",
        help="Output directory for data and plots",
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
    args = parser.parse_args()

    if args.time_stop:
        args.nchunks = int(
            np.floor((args.time_stop - args.time_start) * 250e6 / 4096 / args.acclen)
        )
    else:
        args.time_stop = args.time_start + int(
            np.ceil(args.nchunks * args.acclen * 4096 / 250e6)
        )
    if not args.chans:
        args.chans = [0, None]

    print("nchunks is: ", args.nchunks, "and stop time is ", args.time_stop)
    # assert(1==0)
    pol00, pol11, pol01, channels = get_avg_fast(
        args.data_dir,
        args.time_start,
        args.time_stop,
        args.acclen,
        args.nchunks,
        args.chans[0],
        args.chans[1],
    )
    print("RUN 1 DONE")

    import os

    fname = f"pols_4bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.npz"
    fpath = os.path.join(args.outdir, fname)
    np.savez_compressed(
        fpath,
        datap01=pol01.data,
        maskp01=pol01.mask,
        datap00=pol00.data,
        maskp00=pol00.mask,
        datap11=pol11.data,
        maskp11=pol11.mask,
        chans=channels,
    )

    fname = f"pols_4bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.png"
    fpath = os.path.join(args.outdir, fname)
    butils.plot_4bit(
        pol00,
        pol11,
        pol01,
        channels,
        args.acclen,
        args.time_start,
        args.vmin,
        args.vmax,
        fpath,
        minutes=True,
        logplot=args.logplot,
    )
    print(fpath)
