import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os import path
import sys
import helper
sys.path.insert(0,path.expanduser("~"))
import json


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default=".",
        help="Output plot directory [default: .]",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )
    dir_parents = []
    spec_offsets = []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ref_ant, ant, details)
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])

    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = config["correlation"]["osamp"]
    pfb_size = config["correlation"]["pfb_size"]
    new_acclen = config["correlation"]["new_acclen"]
    cutsize = 16
    print("pfbsize",pfb_size)
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size))
    channels = np.arange(chanstart, chanend)
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)
    print("nchunks", nchunks)
    # print("loaded files", files)
    print("IPFB ROWS", pfb_size, "OSAMP", osamp)
    filt_thresh = 0.2
    # t_acclen = acclen*4096/250e6
    # sys.exit()
    fname = f"xcorr_all_ant_1bit_{str(init_t)}_{str(end_t)}_{str(new_acclen)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}.npy"
    fpath = path.join(args.outdir,fname)
    if osamp > 1:
        t1=time.time()
        pols,new_channels=helper.repfb_xcorr_avg(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,fpath,cutsize=16,filt_thresh=filt_thresh)
        t2=time.time()
    else:
        t1=time.time()
        pols,new_channels=helper.xcorr_avg(idxs,files,pfb_size,nchunks,channels)
        t2=time.time()
    print("Total time taken", t2-t1)

    # fname = f"xcorr_all_ant_4bit_{str(init_t)}_{str(new_acclen)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}_{filt_thresh}.npz"

