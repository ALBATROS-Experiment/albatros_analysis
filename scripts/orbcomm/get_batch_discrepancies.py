import os
import sys
sys.path.append(os.path.expanduser('~'))
import numpy as np 
from matplotlib import pyplot as plt
from datetime import datetime as dt
import figures as fgs
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.scripts.xcorr.fine_timing import dump_upchan_baseband
from albatros_analysis.scripts.orbcomm.get_timing_discrepancy_mod import run_from_config
from albatros_analysis.scripts.xcorr import helper as xchelper
import numba as nb
import time
import importlib
import json
from scipy.optimize import minimize
from skyfield.api import load, wgs84
import cupy
import fitting_helper as fh
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    #parser.add_argument("pulse_list", type = list)
    parser.add_argument("-o", "--out_path", type=str, default="/scratch/thomasb")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )
    dir_parents, spec_offsets, ant_coords = [], [], []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ref_ant, ant, details)
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])

    batch_start_ts = config["correlation"]["start_timestamp"]
    batch_end_ts = config["correlation"]["end_timestamp"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = config["correlation"]["osamp"]
    pfb_size = config["correlation"]["pfb_size"]
    new_acclen = config["correlation"]["new_acclen"]
    cutsize = 16
    channels = np.arange(chanstart, chanend)
    filt_thresh = 0.4
    nant = len(dir_parents)
    npol = 2

    module_path = os.path.join(args.out_path, f"full_timing_discrepancies_{batch_start_ts}")
    os.makedirs(module_path, exist_ok=True)

    print("batch start ts", batch_start_ts, "batch end ts", batch_end_ts)
    print("IPFB ROWS", pfb_size, "OSAMP", osamp)

    discrepancy_list = []

    pulse_list = [[1753216650,1753217000,57166]]

    for pulse in pulse_list:
        print(pulse)
        pulse_start_ts, pulse_end_ts = pulse[0], pulse[1]
        satID = pulse[2]
        nchunks = int(np.floor((pulse_end_ts-pulse_start_ts)*250e6/4096/pfb_size))
        idxs, files = xchelper.get_init_info_all_ant(pulse_start_ts, pulse_end_ts, spec_offsets, dir_parents)
        nrows_total = nchunks * pfb_size // (osamp * new_acclen)
        fname = f"data_raw_osamp={osamp}_start={pulse_start_ts}_end={pulse_end_ts}.npy"
        print(fname)
        disk_path = os.path.join(module_path, fname)
        if os.path.exists(disk_path):
            print('Data already exists! Skipping Computation')
        else:
            t1=time.time()
            pols,new_channels=dump_upchan_baseband(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,disk_path,cutsize=16,filt_thresh=filt_thresh)
            t2=time.time()
            print("Total time taken", t2-t1)
        results = run_from_config(args.config_path, disk_path, 
                                pulse_start_ts, pulse_end_ts, satID, 
                                output_path=module_path, osamp = osamp, plot=False)
        discrepancy_list.append(results)

    print(discrepancy_list)



