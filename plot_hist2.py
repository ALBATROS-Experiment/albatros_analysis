from correlations import baseband_data_classes as bdc
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import os
from palettable.colorbrewer.sequential import GnBu_9 as mycmap

def pretty_print_statistics(array, stats_names, axis, labels):
    """
    Pretty print statistics of a 2-D array.

    Parameters:
    - array: 2-D array of statistics.
    - stats_names: List of statistics names (e.g., ['mean', 'median', 'min', 'max']).
    - axis: The axis along which the statistics are calculated (0 for rows, 1 for columns).
    - labels: List of labels corresponding to the rows or columns of the original data.
    """
    print(labels)
    if len(labels) != array.shape[1]:
        print("Length of labels must match the shape of the unreduced axis.")
        return

    # Transpose array if needed to print row-wise
    if axis == 0:
        array = array.T

    # Print header
    header = " | ".join(f"{str(lb):>8}" for lb in labels)
    print(f"{'Label':>8} | {header}")
    print("-" * (17 + 13 * len(stats_names)))

    # Print data
    num_stats = 4
    for i in range(num_stats):
        row=array[i,:]
        row_str = " | ".join(f"{value:>8.0f}" for value in row)
        print(f"{stats_names[i]:>8} | {row_str}")

if(__name__=='__main__'):
    "Example usage: python quick_spectra.py ~/data_auto_cross/16171/1617100000"
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="Baseband file location. Ex: ~/snap1/16171/161700000/161700026.raw")
    parser.add_argument("-o", "--output_dir", type=str, default="./", help="Output directory for plots")
    parser.add_argument("-m", "--mode", type=int, default=-1, help="0 for pol0, 1 for pol1, -1 for both")
    parser.add_argument("-r", "--rescale", action="store_true", help="Map bit values (0-15 for 4 bit data) to -ve to +ve levels.")
    parser.add_argument("-c", '--chans', type=int, nargs=2, help="Channel numbers for start and end")
    args = parser.parse_args()

    obj = bdc.Baseband(args.filepath)
    hist = obj.get_hist(mode=args.mode)
    ch0 = obj.channels[0]
    ch1 = obj.channels[-1]
    assert args.chans[0] in obj.channels and args.chans[1] in obj.channels
    if(args.chans):
        ch0 = args.chans[0]
        ch1 = args.chans[1]
        chidx0 = ch0 - obj.channels[0]
        chidx1 = ch1 - obj.channels[0]
        hist = hist[:, chidx0:chidx1]
        channels = obj.channels[chidx0:chidx1]
    
    print(f"Hist vals shape:\n{hist.shape}")
    # np.savetxt('./hist_dump_mohan_laptop.txt',hist) this was to check output against code on niagara. all match.
    
    nlevels = 2**obj.bit_mode
    if(args.rescale and obj.bit_mode == 4):
        bins = np.arange(-7, 8)
        hist = np.fft.fftshift(hist, axes=0)  # first row would correspond to -8 which is 0
        assert np.all(hist[0, :] == 0)
        hist = hist[1:, :].copy()
    elif(args.rescale and obj.bit_mode == 1):
        bins = [-1, 1]
    else:
        bins = np.arange(0, nlevels)
    
    print(f"Bins: {bins}")
    print(f"Total data points: {hist.sum()}")
    
    snap, five_digit, timestamp = args.filepath.split('/')[-3:]
    timestamp = timestamp.split('.')[0]

    f = plt.gcf()
    f.set_size_inches(10, 4)
    if(args.mode in (0, 1)):
        tag = 'pol' + str(args.mode)
    else:
        tag = 'both_pols'
    
    start_chan = channels[0]
    end_chan = channels[-1]
    print(f"hist.shape: {hist.shape}")
    plt.suptitle(f'Histogram for {snap} {timestamp} {tag}')
    plt.subplot(121)
    
    print(f"Per chan hist:\n{hist}")
    print(f"Min:\n{np.min(hist, axis=1)}\nMax:\n{np.max(hist, axis=1)}\nStd:\n{np.std(hist, axis=1)}\nMean:\n{np.mean(hist, axis=1)}\n")
    
    plt.imshow(hist, aspect="auto", interpolation='none', cmap=mycmap.mpl_colormap)
    
    freqs = channels
    locs = np.arange(0, len(channels))
    labels = [str(x) for x in channels]
    
    osamp = max(int(len(channels) // 32), 1)
    print(f"OSAMP IS {osamp}, {locs[::osamp]}")
    plt.xticks(locs[::osamp], labels[::osamp], rotation=-50)
    
    locs = np.arange(0, len(bins))
    labels = bins
    plt.yticks(locs, labels)
    plt.colorbar()
    plt.xlabel('channels')

    plt.subplot(122)
    hist_total = np.sum(hist, axis=1)
    plt.bar(bins, hist_total, label=f'mode={args.mode}')
    plt.xticks(bins, bins)
    plt.tight_layout()

    nowstamp = int(time.time())
    fname = os.path.join(args.output_dir, f'hist_{snap}_{timestamp}_{tag}_{ch0}_{ch1}_{nowstamp}.png')
    plt.savefig(fname)
    print(f"Saved plot as: {fname}")

    mean = np.mean(hist, axis=1)
    median = np.median(hist, axis=1)
    min_val = np.min(hist, axis=1)
    max_val = np.max(hist, axis=1)

    stats = np.array([mean, median, min_val, max_val])
    print(bins.shape)
    print(median.shape)
    labels = bins
    pretty_print_statistics(stats, ['mean', 'median', 'min', 'max'], axis=1, labels=labels)
