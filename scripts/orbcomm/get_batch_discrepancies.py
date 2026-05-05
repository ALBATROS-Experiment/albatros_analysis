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
import helper_discrepancies as hd
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
        ant_coords.append(details["coordinates"])

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
    print("batch start ts", batch_start_ts, "batch end ts", batch_end_ts)
    print("IPFB ROWS", pfb_size, "OSAMP", osamp)

    # set up some paths
    path_batch = os.path.join(args.out_path, f"batch_{batch_start_ts}")
    os.makedirs(path_batch, exist_ok=True)
    path_discrepancies = os.path.join(path_batch, 'timing_discrepancies')
    os.makedirs(path_discrepancies, exist_ok=True)
    path_pulses = os.path.join(path_batch, 'data/pulses.json')
    with open(path_pulses, "r") as f:
        pulse_list = json.load(f)

    master_discrepancies = {}
    for i, pulse in enumerate(pulse_list):
        print(f'\nSTARTING PULSE {i}')
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
        path_disk = os.path.join(path_batch, 'data', fname)

        path_cutter = os.path.join(path_batch, "data/cutting_discrep.json")
        if os.path.exists(path_cutter):
            with open(path_cutter, "r") as f1:
                cuts = json.load(f1)
        else:
            cuts = {}

        if os.path.exists(path_disk) and (fname in cuts):
            print('Data already computed and cut! Skipping Computation')

        elif os.path.exists(path_disk):
            print('Data exists but still have to cut it!')
            print('Loading Data')
            data_all = np.load(path_disk)
            new_chans = np.linspace(compute_chans[0], compute_chans[-1]+1, osamp*4, endpoint=False)
            freqs = 250e6 - (new_chans/(4096/250e6))
            print('Computing V')
            V = hd.efield_to_vis(data_all,
                                pulse_start_ts,
                                pulse_end_ts,
                                [0, 2, 3, 4, 5, 6],        
                                ant_coords,
                                satID,
                                freqs,
                                acclen = new_acclen,
                                osamp = osamp,
                                bb_spectrum_T = 4096/250e6)
            print('Cutting')
            start_spectrum, end_spectrum, cut_chans = hd.discrep_cutting(V, satID, acclen=new_acclen)
            cuts[fname] = {'satID': satID,
                            'spectra_start': start_spectrum,
                            'spectra_end': end_spectrum,
                            'cut_chans': cut_chans}

        else:
            print('Need to make data and cut it!')
            print(pulse_end_ts-pulse_start_ts)
            nchunks = int(np.floor((pulse_end_ts-pulse_start_ts)*250e6/4096/pfb_size))
            print('nchunks:', nchunks)
            idxs, files = xchelper.get_init_info_all_ant(pulse_start_ts, pulse_end_ts, spec_offsets, dir_parents)
            nrows_total = nchunks * pfb_size // (osamp * new_acclen)
            print('nrows total:', nrows_total)
            t1=time.time()
            baseband, _ = dump_upchan_baseband(idxs,files,pfb_size,nchunks,compute_chans,osamp,new_acclen,path_disk,cutsize=16,filt_thresh=filt_thresh)
            t2=time.time()
            print("Total time taken", t2-t1)

            new_chans = np.linspace(compute_chans[0], compute_chans[-1]+1, osamp*4, endpoint=False)
            freqs = 250e6 - (new_chans/(4096/250e6))
            V = hd.efield_to_vis(baseband,
                                pulse_start_ts,
                                pulse_end_ts,
                                [0, 2, 3, 4, 5, 6],        
                                ant_coords,
                                satID,
                                freqs,
                                acclen = new_acclen,
                                osamp = osamp,
                                bb_spectrum_T = 4096/250e6)

            spec_cut_start, spec_cut_end, chans_cut = hd.discrep_cutting(V, satID, acclen=new_acclen)
            cuts[fname] = {'satID': satID,
                            'spec_cut_start': start_spectrum,
                            'spec_cut_end': end_spectrum,
                            'chans_cut': chans_cut}
            del baseband
            gc.collect()
        #save the cutting to file (no matter what happens, even if we don't change it)
        with open(f"/scratch/thomasb/batch_{batch_start_ts}/data/cutting_discrep.json", "w") as f2:
            json.dump(cuts, f2, indent=4)
        print('Saved to Cutting Json')
        #fit for discrepancy once data is set up and cut
        results = get_discrepancy(args.config_path, 
                                    path_disk, 
                                    satID, 
                                    out_path=path_batch, 
                                    osamp=osamp, 
                                    plot=True,
                                    coherent=args.coherent)
        print(results)
        master_discrepancies[fname] = results
    #save all results to file
    with open(os.path.join(path_discrepancies, 'times_all_incoherent.json'), 'w') as f:
        json.dump(master_discrepancies, f, indent=4)