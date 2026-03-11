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
from albatros_analysis.scripts.orbcomm.get_pulse_discrepancy import get_discrepancy
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
import gc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    #parser.add_argument("pulse_list", type = list)
    parser.add_argument("-o", "--out_path", type=str, default="/scratch/thomasb")
    parser.add_argument('-c', "--coherent", action='store_true')
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

    master_discrepancies = {}

    pulse_path = '/scratch/thomasb/full_timing_discrepancies_1753200150/pulses2.json'
    with open(pulse_path, "r") as f:
        pulse_list = json.load(f)

    for pulse in pulse_list:
        print(pulse)
        pulse_start_ts, pulse_end_ts = pulse['t_start'], pulse['t_end']
        satID = pulse['sat']
        det_chan = channels[pulse['channel']] #if want to compute with fewer channels
        if det_chan%2 == 0:
            compute_chans = np.array([det_chan-2, det_chan-1, det_chan, det_chan+1])
        else:
            compute_chans = np.array([det_chan-1, det_chan, det_chan+1, det_chan+2])

        fname = f"data_raw_osamp={osamp}_start={pulse_start_ts}_end={pulse_end_ts}_chans={compute_chans[0]}:{compute_chans[-1]}.npy"
        print('Looking at file:', fname)
        disk_path = os.path.join(module_path, fname)

        with open("/scratch/thomasb/full_timing_discrepancies_1753200150/cutting.json", "r") as f1:
                cuts = json.load(f1)

        if os.path.exists(disk_path):
            print('Data already exists! Skipping Computation')
            if fname not in cuts:
                cuts[fname] = {'satID': int(satID)}
                print("Wasn't in cutter, adding it!")
        
        else:
            nchunks = int(np.floor((pulse_end_ts-pulse_start_ts)*250e6/4096/pfb_size))
            idxs, files = xchelper.get_init_info_all_ant(pulse_start_ts, pulse_end_ts, spec_offsets, dir_parents)
            nrows_total = nchunks * pfb_size // (osamp * new_acclen)
    
            t1=time.time()
            _,_=dump_upchan_baseband(idxs,files,pfb_size,nchunks,compute_chans,osamp,new_acclen,disk_path,cutsize=16,filt_thresh=filt_thresh)
            t2=time.time()
            print("Total time taken", t2-t1)
            del _
            gc.collect()

            #add it to cut list to know it's been computed. still need to cut manually though
            cuts.setdefault(fname, {})['satID'] = int(satID)

        with open("/scratch/thomasb/full_timing_discrepancies_1753200150/cutting.json", "w") as f2:
            json.dump(cuts, f2, indent=4)

        results = get_discrepancy(args.config_path, 
                                    disk_path, 
                                    satID, 
                                    out_path=module_path, 
                                    osamp=osamp, 
                                    plot=True,
                                    coherent=args.coherent)
        print(results)
        master_discrepancies[fname] = results

    with open(os.path.join(module_path, 'times_all_coherent.json'), 'w') as f:
        json.dump(master_discrepancies, f, indent=4)