import os
import sys
from os import path
sys.path.insert(0, "/home/thomasb/")
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from albatros_analysis.src.utils import baseband_utils as butils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("-ref", "--ref_ant", type=str, default="MARS2", help='The antenna that was used as reference for satdet')
    parser.add_argument("-sec", '--secondary_ant', type=str, default="MARS7", help='The other antenna that makes up the baseline over which to check SNR')
    parser.add_argument('-c', '--min_chunks', default=3, help='Minimum number of chunks')
    parser.add_argument('-snr', '--min_snr', default=100, help='Minimum SNR')
    parser.add_argument('-l', '--acclen', type=int, default=3000000, help='coarse acclen for satdet cxcorr')
    parser.add_argument('-t', '--testing', type=str, default=None)
    parser.add_argument('-w', '--write_to_file', action='store_true', help='see if we want to write a new file for pulses')
    args = parser.parse_args()


    #OPEN CONFIG-------------------------------------------------------------------------------
    with open(args.config_file, "r") as f:
        config = json.load(f)
        ant_paths, coords, ant_names = [], [], []

        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            print(ant, details)
            ant_paths.append(details["path"])
            ant_names.append(details["name"])
        batch_start_ts = config["correlation"]["start_timestamp"]

    if args.testing is not None:
        path_batch = f'/scratch/thomasb/batch_{batch_start_ts}_testing/{args.testing}'
    else:
        path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
    
    path_satdet = os.path.join(path_batch, 'satdet')
    path_data = os.path.join(path_satdet, f'satdet_{int(args.acclen/1e6)}M_ref{args.ref_ant}.json') #gets us to the data that uses ref ant
    os.makedirs(os.path.join(path_batch, 'data'), exist_ok=True)
    path_out = os.path.join(path_batch, f'data/pulses.json')

    #========================================================================================
    #get the windows
    w_all = butils.get_windows_oneant(path_data, 
                                batch_start_ts, 
                                args.secondary_ant,  #gives us the entry that uses non-ref
                                args.min_snr,
                                args.min_chunks,
                                max_nchunks=None,
        #                       interval=[6e4, 9e4]  #seconds in batch time you can look at
                                )
    print('Number of good pulses', len(w_all)) 

    #========================================================================================
    #check to see what antenna are missing the data
    for pidx, p in enumerate(w_all):
        print(f'\n==========Starting Pulse {pidx}=========')
        print(p)
        t_start = p['t_start']
        print(t_start)
        t_end = p['t_end']
        print(t_end)
        p['missing'] = []
        for antidx, dir in enumerate(ant_paths):
            print(f'lookin at {ant_names[antidx]}')
            missing = butils.check_data_holes(t_start, t_end, dir,)
            if missing:
                print(f'missing {ant_names[antidx]}')
                p['missing'].append(ant_names[antidx])
        print(p)

    print('\n==========ALL PULSES=========') 
    print('Number of good pulses', len(w_all)) 
    for p in w_all:
        print(p)
    #========================================================================================
    #save to json
    if args.write_to_file:
        print('writing to file!')
        with open(path_out, 'w') as f:
            json.dump(w_all, f, indent=4)
        print('Saved to', path_out)