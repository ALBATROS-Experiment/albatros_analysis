import numpy as np 
import matplotlib.pyplot as plt
import json
import os
import sys

batch_start_ts = 1753200150

path = f'/scratch/thomasb/batch_{batch_start_ts}/satdet'

with open(os.path.join(path, 'satdet_3M.json'), 'r') as f:
    data = json.load(f)

for key, value in data.items():
    if key == 'summary':
        continue
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


    continue


    idx = np.argsort(offsets)
    offsets_sorted = offsets[idx]
    weights_sorted = weights[idx]
    cdf = np.cumsum(weights_sorted)
    cutoff = 0.5 * np.sum(weights_sorted)
    consensus = offsets[np.searchsorted(cdf, cutoff)]
    print('consensus offset', consensus)

    spread = np.std(offsets)
    n = len(offsets)

    print({'consensus_offset': int(consensus),
        'spread' : float(spread),
        'chunks detected': int(n)})
    break