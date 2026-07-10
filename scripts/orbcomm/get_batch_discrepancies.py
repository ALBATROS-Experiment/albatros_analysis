#system stuff
import os
import sys
sys.path.insert(0, "/home/thomasb/")
#general
import numpy as np 
import cupy as cp
from matplotlib import pyplot as plt
from datetime import datetime as dt
import time
import argparse
import json
import gc
import importlib
import json
from scipy.optimize import minimize
from skyfield.api import load, wgs84
#utils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.utils import finetiming_utils as futils
#helper and big functions
from albatros_analysis.scripts.xcorr import helper as xchelper
from albatros_analysis.scripts.orbcomm.streaming_fine_timing import repfb
from albatros_analysis.scripts.orbcomm.get_pulse_discrepancy import get_discrepancy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    #parser.add_argument("pulse_list", type = list)
    parser.add_argument("-o", "--out_path", type=str, default="/scratch/thomasb")
    parser.add_argument('-c', "--coherent", action='store_true')
    parser.add_argument('-m', "--meteors_only", action='store_true')
    parser.add_argument('-t', "--testing", type = str, default=None)

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
    ant_idxs = [0, 1, 2, 3, 4, 5, 6]
    T_SPECTRA = 4096/250e6*osamp

    # set up some paths
    if args.testing is not None:
        path_batch = os.path.join(args.out_path, f"batch_{batch_start_ts}_testing/{args.testing}")
    else:
        path_batch = os.path.join(args.out_path, f"batch_{batch_start_ts}")

    os.makedirs(path_batch, exist_ok=True)
    path_discrepancies = os.path.join(path_batch, 'timing_discrepancies')
    os.makedirs(path_discrepancies, exist_ok=True)
    path_pulses = os.path.join(path_batch, 'data/pulses.json')
    path_results = os.path.join(path_discrepancies, 'times_all.json')
    with open(path_pulses, "r") as f:
        pulse_list = json.load(f)
    print('Number of pulses:', len(pulse_list))

    for pidx, pulse in enumerate(pulse_list):
        print(f'\nSTARTING PULSE {pidx}')
        print(pulse)
        pulse_start_ts, pulse_end_ts = pulse['t_start'], pulse['t_end']
        tle_path = outils.get_tle_file(pulse_start_ts, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
        satID = pulse['sat']

        if args.meteors_only:
            if satID not in {59051, 57166}:
                print('NOT RUSSIAN, SKIPPING!')
                continue

        det_chan = channels[pulse['channel']] #if want to compute with fewer channels
        if det_chan%2 == 0:
            compute_chans = np.array([det_chan-2, det_chan-1, det_chan, det_chan+1])
        else:
            compute_chans = np.array([det_chan-1, det_chan, det_chan+1, det_chan+2])

        fname = f"data_raw_osamp={osamp}_start={pulse_start_ts}_end={pulse_end_ts}_chans={compute_chans[0]}:{compute_chans[-1]}.npy"
        print('Looking at file:', fname)
        path_disk = os.path.join(path_batch, 'data', fname)

        #CASE 1: we already computed the discrepancy
        if os.path.exists(path_results):
            with open(path_results, 'r') as f:
                results_check = json.load(f)
            if fname in results_check.keys():
                print('Already have fit! Skipping!')
                continue

        path_cutter = os.path.join(path_batch, "data/cutting_discrep.json")
        if os.path.exists(path_cutter):
            with open(path_cutter, "r") as f1:
                cuts = json.load(f1)
        else:
            cuts = {}

        #CASE 2: we already computed and cut the data
        if os.path.exists(path_disk) and (fname in cuts):
            print('Data already computed and cut! Skipping Computation')

        #CASE 3: we already computed the data
        elif os.path.exists(path_disk):
            print('Data exists but still have to cut it!')
            print('Loading Data')
            baseband = np.load(path_disk)
            print('baseband shape', baseband.shape)
            new_chans = np.linspace(compute_chans[0], compute_chans[-1]+1, osamp*4, endpoint=False)
            freqs = 250e6 - (new_chans/(4096/250e6))
            print('Computing V')
            V = futils.get_vis(baseband,satID,freqs,pulse_start_ts,pulse_end_ts,ant_coords,ant_idxs,tle_path,T_SPECTRA, new_acclen)
            print('Cutting')
            cut_spectra_start, cut_spectra_end, cut_chans = futils.discrep_cutting(V, satID, acclen=new_acclen)
            cuts[fname] = {'satID': satID,
                            'cut_spectra_start': cut_spectra_start,
                            'cut_spectra_end': cut_spectra_end,
                            'cut_chans': cut_chans}

        #CASE 4: we have fuck all
        else:
            print('Need to make data and cut it!')
            print(pulse_end_ts-pulse_start_ts)
            nchunks = int(np.floor((pulse_end_ts-pulse_start_ts)*250e6/4096/pfb_size))
            print('nchunks:', nchunks)

            overflow_ctr = np.zeros(nant, dtype=int)
            for antidx in range(nant):
                overflow_files = xchelper.get_overflow_files(batch_start_ts, batch_end_ts, dir_parents[antidx])
                overflow_ctr[antidx] = np.sum(overflow_files<pulse_start_ts)
            print(overflow_ctr)
            idxs, files, start_specnum = xchelper.get_init_info_all_ant2(pulse_start_ts, pulse_end_ts, spec_offsets, dir_parents, overflow_ctr)

            #idxs, files = xchelper.get_init_info_all_ant(pulse_start_ts, pulse_end_ts, spec_offsets, dir_parents)
            
            nrows_total = nchunks * pfb_size // (osamp * new_acclen)
            print('nrows total:', nrows_total)
            t1=time.time()
            baseband = repfb(idxs,files,pfb_size,nchunks,compute_chans,osamp,new_acclen,path_disk,cutsize=16,filt_thresh=filt_thresh)
            t2=time.time()
            print("Total time taken", t2-t1)

            print('starting spectrum number', start_specnum)
            pulse_list[pidx]['start_specnum'] = int(start_specnum)
            with open(path_pulses, 'w') as f:
                json.dump(pulse_list, f, indent=4)

            new_chans = np.linspace(compute_chans[0], compute_chans[-1]+1, osamp*4, endpoint=False)
            freqs = 250e6 - (new_chans/(4096/250e6))
            V = futils.get_vis(baseband,satID,freqs,pulse_start_ts,pulse_end_ts,ant_coords,ant_idxs,tle_path,T_SPECTRA, new_acclen)

            #START TEMP FIGURE
            #======================================================
            # fig,ax = plt.subplots(5,3, constrained_layout=True)
            # fig.set_size_inches(10,20)
            # ax=np.ravel(ax)
            # nblines, ntimes, nchans = V.shape
            # blnum = 0
            # for i in range(len(ant_idxs)):
            #     for j in range(i+1, len(ant_idxs)):
            #         ai = ant_idxs[i]
            #         aj = ant_idxs[j]
            #         ax[blnum].set_title(f"{ai}-{aj} (id {blnum})")
            #         img=ax[blnum].imshow(np.angle(V[blnum,:,:]),aspect='auto',interpolation='none',cmap='RdBu')
            #         cbar=plt.colorbar(img,ax=ax[blnum])
            #         blnum+=1
            # fig.savefig(os.path.join(path_discrepancies, 'error_vis.png'))
            #======================================================
            #END TEMP FIGURE

            cut_spectra_start, cut_spectra_end, cut_chans = futils.discrep_cutting(V, satID, acclen=new_acclen)
            cuts[fname] = {'satID': satID,
                            'cut_spectra_start': cut_spectra_start,
                            'cut_spectra_end': cut_spectra_end,
                            'cut_chans': cut_chans}
            del baseband
            gc.collect()
            
        #save the cutting to file (no matter what happens, even if we don't change it)
        with open(path_cutter, "w") as f2:
            json.dump(cuts, f2, indent=4)
        print('Saved to Cutting Json')
        #fit for discrepancy once data is set up and cut
        results = get_discrepancy(args.config_path, 
                                path_disk, 
                                satID, 
                                path_batch, 
                                osamp=osamp, 
                                plot=True,
                                coherent=args.coherent)
        print(results)

        if os.path.exists(path_results):
            with open(path_results, "r") as f:
                master = json.load(f)
        else:
            master = {}
        master[fname] = results
        with open(path_results, "w") as f:
            json.dump(master, f, indent=4)