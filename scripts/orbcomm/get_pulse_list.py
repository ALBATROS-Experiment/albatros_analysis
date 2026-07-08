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
    parser.add_argument("batch_start_ts", type=int)
    parser.add_argument("-r", "--ref_ant", type=str, default="MARS2", help='The antenna that was used as reference for satdet')
    parser.add_argument("-n", '--nref_ant', type=str, default="MARS7", help='The other antenna that makes up the baseline over which to check SNR')
    parser.add_argument('-c', '--min_chunks', default=3, help='Minimum number of chunks')
    parser.add_argument('-s', '--min_snr', default=100, help='Minimum SNR')
    parser.add_argument('-l', '--acclen', type=int, default=3000000, help='coarse acclen for satdet cxcorr')
    parser.add_argument('-w', '--write_to_file', action='store_true', help='see if we want to write a new file for pulses')
    args = parser.parse_args()

    #========================================================================================
    #some hardcoded stuff to eventually make modular
    ant_names = ['MARS1', 'MARS2', 'MARS3', 'MARS4', 'MARS5', 'MARS6', 'MARS7', 'MARS8']

    ant_paths = np.array([
    '/project/rrg-sievers/albatros/mars/202507/mars1/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars2/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars3/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars4/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars5',
    '/project/rrg-sievers/albatros/mars/202507/mars6/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars7/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars8/baseband'
])

#     ant_paths = np.array([
#     "/scratch/mohanagr/summer_2025/baseband/mars1",
#     "/scratch/mohanagr/summer_2025/baseband/mars2",
#     "/scratch/mohanagr/summer_2025/baseband/mars3",
#     "/scratch/mohanagr/summer_2025/baseband/mars4",
#     "/scratch/mohanagr/summer_2025/baseband/mars5",
#     "/scratch/mohanagr/summer_2025/baseband/mars6",
#     "/scratch/mohanagr/summer_2025/baseband/mars7",
#     "/scratch/mohanagr/summer_2025/baseband/mars8"
# ])

    #========================================================================================
    #set basic parameters
    batch_start_ts = args.batch_start_ts
    path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
    path_satdet = os.path.join(path_batch, 'satdet')
    path_data = os.path.join(path_satdet, f'satdet_{int(args.acclen/1e6)}M_ref{args.ref_ant}.json') #gets us to the data that uses ref ant
    os.makedirs(os.path.join(path_batch, 'data'), exist_ok=True)
    path_out = os.path.join(path_batch, f'data/pulses.json')

    #========================================================================================
    #get the windows
    w_all = butils.get_windows_oneant(path_data, 
                                batch_start_ts, 
                                args.nref_ant,  #gives us the entry that uses non-ref
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