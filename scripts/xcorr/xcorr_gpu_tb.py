import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os import path
import sys
import helper as hp
import helper_gpu as hpg
sys.path.insert(0,path.expanduser("~"))
import json

outdir = "/project/s/sievers/thomasb/may26/vis"

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config3.json", help="Path to config file")
    return parser.parse_args()

def run_script(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["offset"],
    )
    dir_parents = []
    spec_offsets = []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ref_ant, ant, details)
        dir_parents.append(details["path"])
        spec_offsets.append(details["offset"])

    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    pfb_size = config["correlation"]["accumulation_length"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]

    osamp = 64
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size/osamp))
    idxs, files = hp.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("total time (in s) of visibility computation", end_t - init_t)
    print("final idxs", idxs)
    print("PFB SIZE:", pfb_size)
    print("OVERSAMPLING", osamp)
    # t_acclen = acclen*4096/250e6
    # sys.exit()
    t1=time.time()
    pols,missing_fraction,channels=hpg.repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize=16,filt_thresh=0.45)
    t2=time.time()
    print("Total time taken", t2-t1)

    return pols, missing_fraction, channels



if __name__ == "__main__":
    args = arguments()
    pols, missing_fraction, channels = run_script(args.config)
    fname = f"xcorr_all_ant_4bit_{str(init_t)}_{str(pfb_size)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,missing_fraction=missing_fraction,chans=channels)

