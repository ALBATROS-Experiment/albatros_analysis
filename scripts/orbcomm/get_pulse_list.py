import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

def get_windows_oneant(json_path, 
                       batch_start, 
                       ant,
                       min_SNR,
                       min_nchunks,
                       max_nchunks=None,
                       interval=None
                       ):

    windows = []
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = data[f'{batch_start}']
    ant_data = data[ant]

    for pulse in ant_data:
        t0, t1 = pulse["times"]
        if interval is not None:
            int_start = batch_start + interval[0]
            int_end   = batch_start + interval[1]
            if t1<int_start or t0>int_end:
                continue

        chunks = pulse["SNR, Chan, Sat"]
        if not chunks:
            continue

        snrs  = np.array([c[0] for c in chunks])
        chans = np.array([c[1] for c in chunks])
        sats  = np.array([c[2] for c in chunks])

        nchunks = len(snrs)
        chunk_dt = (t1 - t0) / nchunks

        above = snrs > min_SNR

        # ---------------------------------------------
        # Find contiguous True segments
        # ---------------------------------------------
        segments = []
        start = None

        for i, flag in enumerate(above):
            if flag:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i))
                    start = None

        if start is not None:
            segments.append((start, len(above)))

        if not segments:
            continue

        best_global_mean = -np.inf
        best_start = None
        best_len = 0

        # ---------------------------------------------
        # Evaluate each segment
        # ---------------------------------------------
        for s, e in segments:
            seg_len = e - s

            if seg_len < min_nchunks:
                continue

            segment_snrs = snrs[s:e]

            # Case 1: no max limit → take whole segment
            if max_nchunks is None or seg_len <= max_nchunks:
                seg_mean = segment_snrs.mean()

                if seg_mean > best_global_mean:
                    best_global_mean = seg_mean
                    best_start = s
                    best_len = seg_len

            else:
                # Case 2: need best subwindow of length max_nchunks
                k = max_nchunks

                # cumulative sum for fast sliding mean
                csum = np.cumsum(segment_snrs)
                csum = np.insert(csum, 0, 0)

                # compute window sums
                window_sums = csum[k:] - csum[:-k]
                idx = np.argmax(window_sums)

                seg_mean = window_sums[idx] / k

                if seg_mean > best_global_mean:
                    best_global_mean = seg_mean
                    best_start = s + idx
                    best_len = k

        if best_start is None:
            continue
        end_idx = best_start + best_len

        t_start = int(t0 + best_start * chunk_dt)
        t_end = int(t0 + end_idx * chunk_dt)

        windows.append({
            "antenna": ant,
            "sat": int(sats[best_start]),
            "channel": int(chans[best_start]),
            "t_start": t_start,
            "t_end": t_end,
            "len": t_end - t_start
        })
    return windows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_start_ts", type=int)
    parser.add_argument('-c', '--min_chunks', default=4, help='Minimum number of chunks')
    parser.add_argument('-s', '--min_snr', default=300, help='Minimum SNR')
    args = parser.parse_args()

    batch_start_ts = args.batch_start_ts
    min_chunks = args.min_chunks
    min_snr = args.min_snr

    path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
    path_satdet = os.path.join(path_batch, 'satdet')
    path_data = os.path.join(path_satdet, 'satdet_data_1753200150_3M_len_86260_1769307199.json') #hard-coded for the time being
    path_out = os.path.join(path_batch, f'data/pulses1.json')

    w_all = get_windows_oneant(path_data, 
                            batch_start_ts, 
                            "Antenna 2",
                            min_snr,
                            min_chunks,
                            max_nchunks=None,
        #                   interval=[6e4, 9e4]  #seconds in batch time you can look at
                            )
    print('Number of good pulses', len(w_all)) 
    
    with open(path_out, 'w') as f:
        json.dump(w_all, f, indent=4)
    print('Saved to', path_out)