### IMPORTS ###

# Array-handling
import numpy as np

# Timing
from datetime import datetime
from datetime import timezone as tz
import time

# System utils
import sys, os
# Adds the home directory to the path so that we can import from albatros_analysis
# This won't work if albatros_analysis isn't in your home directory
sys.path.insert(0, os.path.expanduser("~"))

# Nicely packaged up parameters
from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde.ionogram_plotting import ionogram, plot_traces

import logging
logger = logging.getLogger(__name__)

def offset(chan_idx, args = default_args):
    """Compute the time-sample offset between a given channel and the baseline.

    The ionosonde transmits different frequency channels in a staggered
    (non-simultaneous) sequence, so each channel's correlation trace is
    shifted in time relative to the others. This function returns that
    shift, in number of timesamples, for a given channel index.

    Parameters
    ----------
    chan_idx : int
        Index of the frequency channel.

    Returns
    -------
    int
        Number of timesamples by which this channel's data is offset
        from the baseline (channel 0).
    """
    return int(args.trans_len * chan_idx * args.code_baudrate)

def find_approx_ref_idx(corr, best_channel, args = default_args):
    """Find a rough estimate of the reference index (the index of the time at
    which the signal is recieved in best_channel).

    This is a first-pass, coarse estimate obtained by simply locating the
    global maximum of the power trace on the best channel, then correcting
    for that channel's timing offset. It is intended to be refined later
    by find_ref_idx.

    Parameters
    ----------
    corr : ndarray, shape (n_channels, 2, n_samples)
        Correlation data for each channel and polarization.
    best_channel : int
        Index of the channel to use for locating the peak (typically the
        output of pick_best_channel).

    Returns
    -------
    int
        Approximate reference index (in samples), corrected for the
        channel's timing offset.
    """
    corr_sq = np.abs(corr[best_channel, 0, :])**2 + np.abs(corr[best_channel, 1, :])**2

    approx_ref_idx = np.argmax(corr_sq) - offset(best_channel, args=args)

    return approx_ref_idx

def peak_height(corr_power_sq, center_idx, cutoff_inner = 100, cutoff_outer = 500, args = default_args):
    """Computes some measure of how much of a peak there is around center_idx.

    Currently, this is implemented by finding the mean power within
    cutoff_inner indicies and subtracting from that the median peak power that
    lies within cutoff_outer but not cutoff_inner.

    Parameters
    ----------
    corr_power_sq : ndarray
        1D array of squared power values (for a single channel/polarization combo).
    center_idx : int
        Index around which to measure the peak.
    cutoff_inner : int, optional
        Half-width (in samples) of the window treated as "the peak" (default 100).
    cutoff_outer : int, optional
        Outer half-width (in samples) defining the background region, which
        spans from cutoff_inner to cutoff_outer on either side of center_idx
        (default 500).

    Returns
    -------
    float
        Mean peak-region power minus median background power.
    """
    mean_peak_power = np.mean(corr_power_sq[center_idx - cutoff_inner: center_idx + cutoff_inner + 1])
    to_left = corr_power_sq[center_idx - cutoff_outer: center_idx - cutoff_inner]
    to_right = corr_power_sq[center_idx + cutoff_inner + 1: center_idx + cutoff_outer + 1]
    median_offpeak_power = np.median(np.concat([to_left, to_right]))
    return mean_peak_power - median_offpeak_power

def find_ref_idx(corr, best_channel, approx_ref_idx, plotting = False, cutoff_plot = 500, args = default_args):
    """
    Refine the approximate reference index into a precise one.

    The transmitted code repeats many times (params.code_repeat_num
    repeats), each separated by the inter-pulse period (params.ipp). This
    function searches a window around approx_ref_idx for the true peak,
    then scores every subsequent expected repeat location using
    peak_height, and finds the contiguous block of repeats (of length
    2 * code_repeat_num) whose peak heights sum to the largest total. This
    identifies which block of repeats corresponds to the real, well-formed
    received signal (as opposed to noise or sidelobes), and returns the
    corresponding reference index.

    Parameters
    ----------
    corr : ndarray, shape (n_channels, 2, n_samples)
        Correlation data for each channel and polarization.
    best_channel : int
        Index of the channel to use (typically the output of pick_best_channel).
    approx_ref_idx : int
        Coarse estimate of the reference index (e.g. from find_approx_ref_idx).
    plotting : bool, optional
        If True, plot each candidate repeat's power trace in a grid, with
        its computed peak_height annotated, and highlight the block of
        repeats chosen as the best match (default False).
    cutoff_plot : int, optional
        Half-width (in samples) of the window plotted around each candidate
        repeat when plotting is True (default 500).

    Returns
    -------
    int
        Refined reference index, corrected for best_channel's timing offset.
    """
    corr_sq = ( np.abs(corr[best_channel, 0, :])**2
                 + np.abs(corr[best_channel, 1, :])**2 )

    low_idx = ( int(approx_ref_idx - 2 * args.trans_len * args.code_baudrate)
              + offset(best_channel, args=args) )
    high_idx = ( int(approx_ref_idx + 2 * args.trans_len * args.code_baudrate)
               + offset(best_channel, args=args) )

    max_idx = np.argmax(corr_sq[low_idx:high_idx]) + low_idx
    new_low_idx = int(max_idx - 2 * args.trans_len * args.code_baudrate)

    if plotting:
        fig, axs = plt.subplots(5, 8, sharey = True, layout = "constrained",
                                figsize = (9, 6))
        flat_axs = axs.flatten()
    peak_height_arr = []

    for i in range(4 * args.code_repeat_num - 3):
        center_idx = int((i + 3) * (2 * args.ipp * args.code_baudrate)) + new_low_idx
        peak_height_arr.append(peak_height(corr_sq, center_idx))

        if plotting:
            flat_axs[i].plot(corr_sq[center_idx - cutoff_plot: center_idx + cutoff_plot],
                             c = (0.5, 0, 0))
            flat_axs[i].text(x = 0.1, y = 0.9, s = f"{peak_height_arr[-1]:.2f}", 
                             transform=flat_axs[i].transAxes)

    cumsum = np.cumsum(peak_height_arr)
    argmax = np.argmax(cumsum[2 * args.code_repeat_num - 1:] 
                       - cumsum[:-2 * args.code_repeat_num + 1])

    if plotting:
        for j in range(argmax + 1, argmax + 2 * args.code_repeat_num):
            flat_axs[j].set_facecolor((1.0, 0.95, 0.8))
        for k in range(4 * args.code_repeat_num - 3, 40):
            flat_axs[k].axis("off")

        plt.show()
        os.makedirs(args.out_dir, exist_ok=True)
        fig.savefig(os.path.join(args.out_dir, "finding_ref_idx.png"), dpi = 200)
        # Make sure the figure gets cleaned up once we're done with it.
        plt.close(fig)

    final_idx = int((argmax + args.code_repeat_num + 3)
                    * (2 * args.ipp * args.code_baudrate)) + new_low_idx

    return final_idx - offset(best_channel, args=args)


def extract_traces(freqs, corr, ref_idx, med_arr, dist_range = [-2000, 2000], chan_indices = None, args = default_args):
    """Extract each channel's windowed trace around ref_idx.

    ...

    Parameters
    ----------
    ...
    chan_indices : array-like of int, optional
        Indices into freqs/corr specifying which channels to extract. If
        None (default), all channels are extracted. Indices are always
        interpreted as positions in the *original* freqs/corr arrays, so
        that each channel's staggered-transmission timing offset (which
        depends on its absolute position in the transmit sequence) is
        computed correctly even when extracting only a subset. Do not
        pre-slice freqs/corr to select a subset — pass chan_indices
        instead, or offsets will be wrong.

    Returns
    -------
    normalized : ndarray, shape (n_samples, len(chan_indices))
        Median-normalized power for each selected channel.
    pol0, pol1 : ndarray, complex, shape (n_samples, len(chan_indices))
        Raw polarization correlation values for each selected channel.
    dists : ndarray
        Distance axis (km) relative to ref_idx.
    """
    if chan_indices is None:
        chan_indices = np.arange(len(freqs))
    else:
        chan_indices = np.asarray(chan_indices)

    if dist_range:
        ref_idx_low = int(dist_range[0] / args.c * args.code_baudrate + ref_idx)
        ref_idx_high = int(dist_range[1] / args.c * args.code_baudrate + ref_idx)
    else:
        ref_idx_low = - offset(len(freqs) - 1, args=args)
        ref_idx_high = corr.shape[2]

    n_samples = ref_idx_high - ref_idx_low
    n_chan = len(chan_indices)

    pol0 = np.full([n_samples, n_chan], np.nan, dtype=np.float16)
    pol1 = np.full([n_samples, n_chan], np.nan, dtype=np.float16)

    for out_i, chan_i in enumerate(chan_indices):
        # NOTE: offset() is computed from chan_i, the channel's index in
        # the *original* freqs/corr arrays -- this is what keeps the
        # staggered-transmission timing correct when only a subset of
        # channels is being extracted.
        low_idx = ref_idx_low + offset(chan_i, args=args)
        high_idx = ref_idx_high + offset(chan_i, args=args)

        valid_low = max(low_idx, 0)
        valid_high = min(high_idx, corr.shape[2])

        if valid_low >= valid_high:
            continue

        out_start = valid_low - low_idx
        out_end = out_start + (valid_high - valid_low)

        pol0[out_start:out_end, out_i] = np.abs(corr[chan_i, 0, valid_low:valid_high])
        pol1[out_start:out_end, out_i] = np.abs(corr[chan_i, 1, valid_low:valid_high])

    dists = np.arange(ref_idx_low - ref_idx, ref_idx_high - ref_idx) / args.code_baudrate * args.c

    corr_sq = (pol0**2 + pol1**2)
    if med_arr:
        total_corr_db = 5 * np.log(corr_sq / med_arr)
    else:
        total_corr_db = 5 * np.log(corr_sq / np.median(corr_sq, axis = 0))
        print(np.median(corr_sq, axis = 0))

    return total_corr_db, pol0, pol1, dists

def find_plasma_freq(freqs, total_corr_db, snr_threshold_db = 6, args = default_args):
    """Estimate the plasma frequency from an ionogram's normalized traces.
 
    The plasma frequency is approximated here as the highest frequency
    channel that shows a detectable reflected signal: a channel counts as
    detectable if any sample in its median-normalized power trace exceeds
    snr_threshold_db, using the same 5*log10(normalized) dB convention as
    the ionogram plot. Above the true plasma frequency, radio waves are no
    longer reflected back by the ionosphere, so the highest frequency that
    still shows a return is a reasonable estimate of it.
 
    Parameters
    ----------
    freqs : ndarray
        Array of channel frequencies, in Hz.
    normalized : ndarray, shape (n_samples, n_channels)
        Median-normalized power for each channel, e.g. as returned by
        extract_traces (NaNs, from missing data, are ignored).
    snr_threshold_db : float, optional
        Minimum SNR, in dB, for a channel to be considered to have a
        detectable signal (default 6 dB).
 
    Returns
    -------
    float or None
        Average of the highest-frequency channel with a detectable signal
        and the next channel's frequency (i.e. the midpoint between the
        last "detectable" channel and the first "undetectable" one above
        it). Returns None if no channel shows a detectable signal, or if
        every channel shows a detectable signal (since there is then no
        higher, undetectable channel to average against).
    """
 
    detectable = np.zeros(len(freqs), dtype=bool)
    for i in range(len(freqs)):
        col = total_corr_db[:, i]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            continue
        detectable[i] = np.any(valid >= snr_threshold_db)

    # If there are no detectable channels or the highest channel has a detectable signal
    if not np.any(detectable) or detectable[-1]:
        return None
  
    detected_positions = np.where(detectable)[0]
    highest_pos = detected_positions.max()
 
    # Guaranteed to exist since the highest channel is not detectable
    next_pos = highest_pos + 1
 
    return (sorted_freqs[highest_pos] + sorted_freqs[next_pos]) / 2

def process_and_plot(ref_idx_plotting = False, cutoff_plot = 500, dist_range = [-2000, 2000],
                      which_pol = "total", pcm_kw = {"vmin": 0, "vmax": 30, "cmap": "jet"},
                      figsize = None, plasma_snr_threshold = 12, args=default_args):
    """Run the full ionogram pipeline on a saved correlation file and plot it.

    Loads frequency and correlation data from an .npz file, automatically
    selects the clearest channel, locates the true reference (transmit)
    index, and produces an ionogram plot.

    Parameters
    ----------
    ref_idx_plotting : bool, optional
        Passed through to find_ref_idx: if True, plots each candidate
        repeat's power trace while refining the reference index
        (default False).
    cutoff_plot : int, optional
        Passed through to find_ref_idx: half-width (in samples) of the
        window plotted around each candidate repeat when plotting is
        True (default 500).
    dist_range : list of float, optional
        Passed through to ionogram: [min, max] distance range (km) to
        display around the reference (default [-2000, 2000]).
    which_pol : str, optional
        Passed through to ionogram (currently an unused placeholder for
        selecting a polarization; default "total").
    pcm_kw : dict, optional
        Passed through to ionogram: extra keyword arguments for
        ax.pcolormesh (e.g. vmin, vmax, cmap).
    figsize : tuple of float, optional
        Passed through to ionogram: size of the resulting figure, in
        inches.
    plasma_snr_threshold : float, optional
        Passed through to ionogram: SNR threshold (dB) used by
        find_plasma_freq to decide whether a channel has a detectable
        signal (default 6 dB).
    """
    data = np.load(os.path.join(args.out_dir, args.corr_name))
    freqs, corr = data["freqs"], data["corr"]

    # TODO: Benchmark the following and see if there is a faster alternative
    corr_power_sq = np.abs(corr[:, 0, :])**2 + np.abs(corr[:, 1, :])**2
    med_arr = np.median(corr_power_sq, axis = 1)
    print(med_arr)
    max_arr = np.max(corr_power_sq, axis = 1)
    snr_ratios = max_arr / med_arr
    best_channel =  np.argmax(snr_ratios)
    best_six_chan = np.argpartition(snr_ratios, -6)[-6:]
    
    approx_ref_idx = find_approx_ref_idx(corr, best_channel, args=args)
    
    ref_idx = find_ref_idx(corr, best_channel, approx_ref_idx, plotting=ref_idx_plotting,
                            cutoff_plot=cutoff_plot, args=args)
    
    total_corr_db, pol0, pol1, dists = extract_traces(freqs, corr, ref_idx, med_arr = None,
                                                   dist_range=dist_range, args=args)
    
    plot_traces(freqs, dists, total_corr_db, ref_idx, best_six_chan, args = args)
    
    est_plasma_freq = find_plasma_freq(freqs, total_corr_db, snr_threshold_db=plasma_snr_threshold, args = args)

    ionogram(freqs, dists, total_corr_db, ref_idx, plasma_freq = est_plasma_freq,
             pcm_kw=pcm_kw, figsize=figsize, ref_freq = best_channel, args=args)

    data_to_save = {
        f"{args.start_time}_mars{args.which_ant}_pol0": pol0,
        f"{args.start_time}_mars{args.which_ant}_pol1": pol1,
        f"{args.start_time}_mars{args.which_ant}_chan_medians": med_arr,
        f"{args.start_time}_mars{args.which_ant}_chan_maxima": max_arr,
        f"{args.start_time}_mars{args.which_ant}_total_corr_db": total_corr_db
    }

    return freqs[best_channel], est_plasma_freq, data_to_save

if __name__ == "__main__":
    process_and_plot() 