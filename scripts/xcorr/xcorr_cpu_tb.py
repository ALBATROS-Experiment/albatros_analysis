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
import json

outdir = "/project/s/sievers/thomasb/may26/vis"

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config3.json", help="Path to config file")
    parser.add_argument("-o", '--outdir', type=str, default="/project/sievers/s/thomasb", help="Output file path")
    return parser.parse_args()

def run_script(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )
    dir_parents = []
    spec_offsets = []
    info = {}
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ref_ant, ant, details)
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])
    init_t = config["correlation"]["start_timestamp"]
    end_t  = config["correlation"]["end_timestamp"]
    acclen = config["correlation"]["accumulation_length"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    outdir = "/project/s/sievers/mohanagr/cpu_all_antenna"
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)
    t_acclen = acclen*4096/250e6

    info = {}
    info["init_t"] = init_t
    info["end_t"] = end_t
    info["acclen"] = acclen
    info["chanstart"] = chanstart
    info["chanend"] = chanend
    info["nchunks"] = nchunks
    pols,rowcounts,channels =helper.get_avg_fast2(idxs, files, acclen, nchunks, chanstart, chanend)
    
    return pols, rowcounts, channels, info


if __name__ == "__main__":
    args = arguments()
    pols, rowcounts, channels, info = run_script(args.config)
    init_t, end_t, acclen = info[0], info[1], info[2]
    fname = f"xcorr_00_11_4bit_{str(init_t)}_{str(end_t)}_{str(acclen)}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,rowcounts=rowcounts,chans=channels)

