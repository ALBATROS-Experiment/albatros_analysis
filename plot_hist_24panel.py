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
        print("channels", obj.channels)
        chidx0 = np.where(obj.channels == ch0)[0][0]
        chidx1 = np.where(obj.channels == ch1)[0][0] + 1
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
    f.set_size_inches(15, 10)
    if(args.mode in (0, 1)):
        tag = 'pol' + str(args.mode)
    else:
        tag = 'both_pols'
    
    start_chan = channels[0]
    end_chan = channels[-1]
    print(f"hist.shape: {hist.shape}")
    plt.suptitle(f'Histogram for {snap} {timestamp} {tag}', fontsize=20)
    
    print(f"Per chan hist:\n{hist}")
    print(f"Min:\n{np.min(hist, axis=1)}\nMax:\n{np.max(hist, axis=1)}\nStd:\n{np.std(hist, axis=1)}\nMean:\n{np.mean(hist, axis=1)}\n")
    
    # Grid parameters
    rows, cols = 4, 6
    max_panels = rows * cols
    num_channels = len(channels)
    
    if num_channels > max_panels:
        print(f"WARNING: Selected {num_channels} channels but can only display {max_panels}. Only showing the first {max_panels} channels.")
        display_channels = channels[:max_panels]
    else:
        display_channels = channels
    
    if args.mode == -1:
        # Split the data for two polarizations
        obj_pol0 = bdc.Baseband(args.filepath)
        obj_pol1 = bdc.Baseband(args.filepath)
        
        hist_pol0 = obj_pol0.get_hist(mode=0)
        hist_pol1 = obj_pol1.get_hist(mode=1)
        
        # Check if gain coefficients exist
        has_coeffs = hasattr(obj, 'coeffs')
        
        if args.chans:
            chidx0 = np.where(obj_pol0.channels == ch0)[0][0]
            chidx1 = np.where(obj_pol0.channels == ch1)[0][0] + 1
            hist_pol0 = hist_pol0[:, chidx0:chidx1]
            hist_pol1 = hist_pol1[:, chidx0:chidx1]
        
        # Apply the same rescaling if needed
        if(args.rescale and obj.bit_mode == 4):
            hist_pol0 = np.fft.fftshift(hist_pol0, axes=0)[1:, :]
            hist_pol1 = np.fft.fftshift(hist_pol1, axes=0)[1:, :]
        
        # Create subplots for each channel
        for i, chan in enumerate(display_channels):
            if i >= max_panels:
                break
                
            plt.subplot(rows, cols, i+1)
            # Get index of this channel in the display_channels array
            chan_idx = i
            
            # Get the histogram for this specific channel
            hist_chan_pol0 = hist_pol0[:, chan_idx]
            hist_chan_pol1 = hist_pol1[:, chan_idx]
            
            # Get original channel index for coefficient lookup
            orig_chan_idx = np.where(obj.channels == chan)[0][0]
            
            # Add gain coefficient to title if available
            title = f'Channel {chan}'
            labels = []
            
            # Calculate RMS for each polarization
            total_counts_pol0 = np.sum(hist_chan_pol0)
            total_counts_pol1 = np.sum(hist_chan_pol1)
            
            # Calculate RMS (sqrt of sum(bin_value^2 * count) / total_count)
            rms_pol0 = np.sqrt(np.sum(bins**2 * hist_chan_pol0) / total_counts_pol0) if total_counts_pol0 > 0 else 0
            rms_pol1 = np.sqrt(np.sum(bins**2 * hist_chan_pol1) / total_counts_pol1) if total_counts_pol1 > 0 else 0
            
            # Quantization delta is 1.0 for rescaled data
            quant_delta = 1.0
            delta_std_ratio_pol0 = quant_delta / rms_pol0 if rms_pol0 > 0 else 0
            delta_std_ratio_pol1 = quant_delta / rms_pol1 if rms_pol1 > 0 else 0
            
            # Create labels with gain information if available
            if has_coeffs:
                try:
                    gain_pol0 = np.log2(obj.coeffs[0, orig_chan_idx]) if obj.coeffs[0, orig_chan_idx] > 0 else 0
                    gain_pol1 = np.log2(obj.coeffs[1, orig_chan_idx]) if obj.coeffs[1, orig_chan_idx] > 0 else 0
                    labels.append(f'pol0 (gain={gain_pol0:.1f}, Δ/σ={delta_std_ratio_pol0:.3f})')
                    labels.append(f'pol1 (gain={gain_pol1:.1f}, Δ/σ={delta_std_ratio_pol1:.3f})')
                except (IndexError, AttributeError):
                    labels = [f'pol0 (Δ/σ={delta_std_ratio_pol0:.3f})', f'pol1 (Δ/σ={delta_std_ratio_pol1:.3f})']
            else:
                labels = [f'pol0 (Δ/σ={delta_std_ratio_pol0:.3f})', f'pol1 (Δ/σ={delta_std_ratio_pol1:.3f})']
            
            # Calculate max value for ylim with headroom
            max_val = max(np.max(hist_chan_pol0), np.max(hist_chan_pol1))
            plt.ylim(0, max_val * 1.25)  # Add 25% headroom
            
            plt.bar(bins, hist_chan_pol0, alpha=0.6, color='royalblue', label=labels[0])
            plt.bar(bins, hist_chan_pol1, alpha=0.6, color='orangered', label=labels[1])
            
            plt.title(title, fontsize=14)
            plt.xticks(bins, bins, fontsize=8)
            plt.yticks([])  # Remove y-axis ticks
            plt.legend(loc='upper right', fontsize=8, framealpha=0.8)  # Add legend to each subplot
                
    else:
        # Create subplots for each channel with single polarization
        for i, chan in enumerate(display_channels):
            if i >= max_panels:
                break
                
            plt.subplot(rows, cols, i+1)
            # Get index of this channel in the display_channels array
            chan_idx = i
            
            # Get the histogram for this specific channel
            hist_chan = hist[:, chan_idx]
            
            # Get original channel index for coefficient lookup
            orig_chan_idx = np.where(obj.channels == chan)[0][0]
            
            # Add gain coefficient to title if available
            title = f'Channel {chan}'
            label = f'pol{args.mode}'
            
            # Check if gain coefficients exist
            has_coeffs = hasattr(obj, 'coeffs')
            
            # Calculate RMS
            total_counts = np.sum(hist_chan)
            # Calculate RMS (sqrt of sum(bin_value^2 * count) / total_count)
            rms = np.sqrt(np.sum(bins**2 * hist_chan) / total_counts) if total_counts > 0 else 0
            
            # Quantization delta is 1.0 for rescaled data
            quant_delta = 1.0
            delta_std_ratio = quant_delta / rms if rms > 0 else 0
            
            # Create label with gain information if available
            if has_coeffs:
                try:
                    gain = np.log2(obj.coeffs[args.mode, orig_chan_idx]) if obj.coeffs[args.mode, orig_chan_idx] > 0 else 0
                    label = f'pol{args.mode} (gain={gain:.1f}, Δ/σ={delta_std_ratio:.3f})'
                except (IndexError, AttributeError):
                    label = f'pol{args.mode} (Δ/σ={delta_std_ratio:.3f})'
            else:
                label = f'pol{args.mode} (Δ/σ={delta_std_ratio:.3f})'
            
            # Calculate max value for ylim with headroom
            max_val = np.max(hist_chan)
            plt.ylim(0, max_val * 1.25)  # Add 25% headroom
            
            plt.bar(bins, hist_chan, label=label)
            plt.title(title, fontsize=14)
            plt.xticks(bins, bins, fontsize=8)
            plt.yticks([])  # Remove y-axis ticks
            plt.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.25, left=0.05, right=0.95, top=0.92, bottom=0.05)  # Make room for the suptitle

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


