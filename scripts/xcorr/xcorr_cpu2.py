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

if __name__=="__main__":

    with open("config.json", "r") as f:
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
    acclen = config["correlation"]["accumulation_length"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    outdir = "/project/s/sievers/mohanagr/cpu_all_antenna"
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)
    t_acclen = acclen*4096/250e6

    pols,rowcounts,channels=helper.get_avg_fast2(idxs, files, acclen, nchunks, chanstart, chanend)

    fname = f"xcorr_00_11_4bit_{str(init_t)}_{str(acclen)}_{str(nchunks)}_{chanstart}_{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,rowcounts=rowcounts,chans=channels)

