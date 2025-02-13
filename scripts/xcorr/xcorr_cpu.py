import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os import path
import sys
import helper
sys.path.insert(0,path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
from albatros_analysis.src.utils import baseband_utils as butils


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir1', type=str,help='Parent data directory antenna 1, should have 5 digit time folders.')
    parser.add_argument('data_dir2', type=str,help='Parent data directory antenna 2, should have 5 digit time folders.')
    parser.add_argument("time_start",type=int, help="Start timestamp ctime")
    parser.add_argument("acclen", type=int, help="Accumulation length for averaging")
    parser.add_argument("delay", type=int, help="Specnum difference between two antennas +ve or -ve")
    parser.add_argument('-n', '--nchunks', dest='nchunks',type=int, default=560, help='Number of chunks in output file. If stop time is specfied this is overwritten. Default 560 ~ 1 hr.')
    parser.add_argument('-t', '--time_stop', dest='time_stop',type=int, default=False, help='Stop time. Overwrites nchunks if specified')
    parser.add_argument("-c", '--chans', type=int, nargs=2, help="Indices of start and end channels.")
    parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='/project/s/sievers/mohanagr/',
              help='Output directory for data and plots')
    args = parser.parse_args()

    if(args.time_stop):
        args.nchunks = int(np.floor((args.time_stop-args.time_start)*250e6/4096/args.acclen))
        print("TOTAL CHUNKS:", args.nchunks)
    else:
        args.time_stop = args.time_start + int(np.ceil(args.nchunks*args.acclen*4096/250e6))
    if(not args.chans):
        args.chans=[0,None]

    path1=args.data_dir1
    path2=args.data_dir2
    print(path1,path2)
    init_t = args.time_start #c#1627441379 #1627441542 #1627439234
    acclen=args.acclen
    t_acclen = acclen*4096/250e6
    delay=args.delay
    nchunks=args.nchunks
    end_t = int(init_t + nchunks*t_acclen)
    pols,rowcounts,channels=helper.get_avg_fast(path1, path2, init_t, end_t, delay, acclen, nchunks, chanstart=args.chans[0], chanend=args.chans[1])

    fname = f"xcorr_00_11_4bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{str(args.delay)}_{args.chans[0]}_{args.chans[1]}.npz"
    fpath = path.join(args.outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,rowcounts=rowcounts,chans=channels)

