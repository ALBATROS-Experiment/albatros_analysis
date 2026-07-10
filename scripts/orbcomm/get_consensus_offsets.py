import numpy as np 
import matplotlib.pyplot as plt
import json
import os
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('-r', '--ref_ant', type=str, default='MARS2', help='determines the reference antenna')
    parser.add_argument('-l', '--acclen', type=int, default=3000000, help='determines the reference antenna')
    parser.add_argument('-t', '--testing', type=str, default=None)
    parser.add_argument('-f', '--write_to_file', action='store_true', help='writes the consensus offsets to the satdet json file')
    parser.add_argument('-c', '--write_to_config', action='store_true', help='writes the consensus offsets to the config file')
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)
        batch_start_ts = config["correlation"]["start_timestamp"]

    if args.testing is not None:
        path = f'/scratch/thomasb/batch_{batch_start_ts}_testing/{args.testing}/satdet'
    else:
        path = f'/scratch/thomasb/batch_{batch_start_ts}/satdet'

    with open(os.path.join(path, f'satdet_{int(args.acclen/1e6)}M_ref{args.ref_ant}.json'), 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        if key == 'summary':
            continue
        print(f'\n============{key}==============')
        baseline_data = data[key]

        offsets_all = []
        offsets_russian = []
        offsets_perpulse = []
        
        weights_all = []

        #add each offset and SNR to their lists
        for pulse in baseline_data:
            offsets_pulse = []
            for off, (snr, chan, sat) in zip(pulse["specnumoffsets"], pulse["SNR, Chan, Sat"]):
                if off == 0:
                    continue

                offsets_all.append(off)
                offsets_pulse.append(off)
                weights_all.append(snr)
                if sat in {59051, 57166}:
                    offsets_russian.append(off)
            offsets_perpulse.append(np.median(offsets_pulse))

        
        
        #turn them into numpy arrays
        offsets_all = np.array(offsets_all)
        offsets_russian = np.array(offsets_russian)
        offsets_perpulse = np.array(offsets_perpulse)
        weights_all = np.array(weights_all)

        print('offsets', offsets_perpulse)

        #determine the median and MAD of the offsets
        med_all = np.median(offsets_all)
        med_russian = np.median(offsets_russian)
        med_perpulse = np.median(offsets_perpulse)
        
        mad_all = np.median(np.abs(offsets_all - med_all))
        mad_russian = np.median(np.abs(offsets_russian - med_russian))
        mad_perpulse = np.median(np.abs(offsets_perpulse - med_perpulse))

        #mask thing
        # mask = np.abs(offsets - med) <= 3 * mad
        # offsets_masked = offsets[mask]
        # weights_masked = weights[mask]

        # std = np.std(offsets_masked)



        print('\nmedian (A)', med_all)
        print('median (R)', med_russian)
        print('median (P)', med_perpulse)

        print('MAD (A)', mad_all)
        print('MAD (R)', mad_russian)
        print('MAD (P)', mad_perpulse)
    

        data['summary'][key] = {'consensus_offset': int(med_russian),
                            'mad': float(mad_russian)}
    
    #print(data['summary'])
    #sys.exit()         
    if args.write_to_file:       
        print('writing to satdet file!')
        with open(os.path.join(path, f'satdet_{int(args.acclen/1e6)}M_ref{args.ref_ant}.json'), 'w') as f:
            json.dump(data, f, indent=4)

    if args.write_to_config:       
        print('writing to config file!')
        offset_data = data['summary']
        print(offset_data)
        with open(args.config_path, 'r') as f:
            config_all = json.load(f)
            for key, items in offset_data.items():
                config_all['antenna'][key]['clock_offset'] = items['consensus_offset']
                #print(config_all['antenna'][key])
        with open(args.config_path, 'w') as f:
            json.dump(config_all, f, indent=4)


    # idx = np.argsort(offsets)
    # offsets_sorted = offsets[idx]
    # weights_sorted = weights[idx]
    # cdf = np.cumsum(weights_sorted)
    # cutoff = 0.5 * np.sum(weights_sorted)
    # consensus = offsets[np.searchsorted(cdf, cutoff)]
    # print('consensus offset', consensus)

    # spread = np.std(offsets)
    # n = len(offsets)

    # print({'consensus_offset': int(consensus),
    #     'spread' : float(spread),
    #     'chunks detected': int(n)})
    #     break