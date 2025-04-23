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

    with open("config_gpu.json", "r") as f:
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
    pfb_size = config["correlation"]["accumulation_length"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = 64
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size/osamp))
    outdir = "/project/s/sievers/mohanagr/gpu_all_antenna"
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)
    print(pfb_size, osamp)
    # t_acclen = acclen*4096/250e6
    # sys.exit()
    t1=time.time()
    pols,missing_fraction,channels=helper.repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize=16,filt_thresh=0.45)
    t2=time.time()
    print("Total time taken", t2-t1)

    fname = f"xcorr_all_ant_4bit_{str(init_t)}_{str(pfb_size)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,missing_fraction=missing_fraction,chans=channels)

