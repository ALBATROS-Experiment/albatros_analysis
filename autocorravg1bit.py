import numpy as np

# from correlations_temp import baseband_data_classes as bdc
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
from utils import baseband_utils as butils
import argparse


def get_avg_fast_1bit(path, init_t, end_t, acclen, nchunks, chanstart=0, chanend=None):
    idxstart, fileidx, files = butils.get_init_info(init_t, end_t, path)
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
    if ant1.obj.bit_mode != 1:
        raise NotImplementedError(
            f"BIT MODE {ant1.obj.bit_mode} IS NOT SUPPORTED BY THIS SCRIPT."
        )
    nchans = ant1.obj.chanend - ant1.obj.chanstart
    pol01 = np.zeros((nchunks, nchans), dtype="complex64", order="c")
    j = ant1.spec_num_start
    m = ant1.spec_num_start
    st = time.time()
    for i, chunk in enumerate(ant1):
        t1 = time.time()
        pol01[i, :] = cr.avg_xcorr_1bit(
            chunk["pol0"], chunk["pol1"], chunk["specnums"], nchans
        )
        t2 = time.time()
        print("time taken for one loop", t2 - t1)
        j = ant1.spec_num_start
        assert j == m + (i + 1) * acclen
        print(i + 1, "CHUNK READ")
    print("Time taken final:", time.time() - st)
    pol01 = np.ma.masked_invalid(pol01)
    return pol01, ant1.obj.channels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average over pol01 of 1 bit baseband files with custom control over acclen/timestamps and channels."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Parent data directory. Should have 5 digit time folders.",
    )
    parser.add_argument("time_start", type=int, help="Start timestamp ctime")
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
        type=int,
        default=False,
        help="Stop time. Overwrites nchunks if specified",
    )
    parser.add_argument(
        "-c",
        "--chans",
        type=int,
        nargs=2,
        help="Indices of start and end channels. Start channel index MUST be even.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default="/scratch/s/sievers/mohanagr/",
        help="Output directory for data and plots",
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
    pol01, channels = get_avg_fast_1bit(
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

    fname = f"pol01_1bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.npz"
    fpath = os.path.join(args.outdir, fname)
    np.savez_compressed(fpath, datap01=pol01.data, maskp01=pol01.mask, chans=channels)
    r = np.real(pol01)
    im = np.imag(pol01)

    fname = f"pol01_1bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.png"
    fpath = os.path.join(args.outdir, fname)
    butils.plot_1bit(
        pol01,
        channels,
        args.acclen,
        args.time_start,
        fpath,
        minutes=True,
        logplot=False,
    )
    print(fpath)
