### IMPORTS ###

import logging
logger = logging.getLogger(__name__)

# Array-handling
import numpy as np
# Peak finding with distance seperation
from scipy.signal import find_peaks

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# System utils
import sys, os
# Adds the home directory to the path so that we can import from albatros_analysis
# This won't work if albatros_analysis isn't in your home directory
sys.path.insert(0, os.path.expanduser("~"))

# Nicely packaged up parameters
from albatros_analysis.scripts.ionosonde.params import default_args


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

def pick_best_channel(corr, args = default_args):
    """Pick the frequency channel with the clearest (most prominent) signal.

    "Clearest" is judged with a crude signal-to-noise ratio: for each
    channel, the ratio of its peak power to its median power. Channels
    that are mostly noise will have a peak power close to the median
    (low ratio), while channels with a strong, well-defined return will
    have a much larger peak-to-median ratio.

    Parameters
    ----------
    corr : ndarray, shape (n_channels, 2, n_samples)
        Correlation data for each channel and polarization (2 = polarizations).

    Returns
    -------
    int
        Index of the channel with the highest peak/median power ratio.
    """
    # Many of the frequencies may be pretty noisy, so we want to select the
    # ones with the most prominent peaks
    # We do this by finding a crude signal-to-noise ratio
    corr_power_sq = np.abs(corr[:, 0, :])**2 + np.abs(corr[:, 1, :])**2
    
    max_arr = np.max(corr_power_sq, axis = 1)
    med_arr = np.median(corr_power_sq, axis = 1)
    snr_ratios = max_arr / med_arr
    
    return np.argmax(snr_ratios)

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
    corr_power = np.abs(corr[best_channel, 0, :])**2 + np.abs(corr[best_channel, 1, :])**2

    approx_ref_idx = np.argmax(corr_power) - offset(best_channel, args=args)

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
    median_offpeak_power = np.median([to_left, to_right])
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
    corr_power_sq = ( np.abs(corr[best_channel, 0, :])**2
                 + np.abs(corr[best_channel, 1, :])**2 )

    low_idx = ( int(approx_ref_idx - 2 * args.trans_len * args.code_baudrate)
              + offset(best_channel, args=args) )
    high_idx = ( int(approx_ref_idx + 2 * args.trans_len * args.code_baudrate)
               + offset(best_channel, args=args) )

    max_idx = np.argmax(corr_power_sq[low_idx:high_idx]) + low_idx
    new_low_idx = int(max_idx - 2 * args.trans_len * args.code_baudrate)

    if plotting:
        fig, axs = plt.subplots(4, 10, sharey = True, layout = "constrained",
                                figsize = (10, 6))
        flat_axs = axs.flatten()
    peak_height_arr = []

    for i in range(4 * args.code_repeat_num - 3):
        center_idx = int((i + 3) * (2 * args.ipp * args.code_baudrate)) + new_low_idx
        peak_height_arr.append(peak_height(corr_power_sq, center_idx))

        if plotting:
            flat_axs[i].plot(corr_power_sq[center_idx - cutoff_plot: center_idx + cutoff_plot],
                             c = (0.5, 0, 0))
            flat_axs[i].text(x = 0.1, y = 0.9, s = f"{peak_height_arr[-1]:.2f}", 
                             transform=flat_axs[i].transAxes)

    cumsum = np.cumsum(peak_height_arr)
    argmax = np.argmax(cumsum[2 * args.code_repeat_num - 1:] 
                       - cumsum[:-2 * args.code_repeat_num + 1])

    if plotting:
        for i in range(argmax + 1, argmax + 2 * args.code_repeat_num):
            flat_axs[i].set_facecolor((1.0, 0.95, 0.8))

        plt.show()

    final_idx = int((argmax + args.code_repeat_num + 3)
                    * (2 * args.ipp * args.code_baudrate)) + new_low_idx

    return final_idx - offset(best_channel, args=args)
    

def plot_all_traces(freqs, corr, ref_idx):
    pass

def ionogram(freqs, corr, ref_idx, dist_range = [-2000, 2000], which_pol = "total",
             pcm_kw = {"vmin": 0, "vmax": 15, "cmap": "jet"}, args = default_args):
    """Build and plot an ionogram: signal power vs. frequency and distance.

    For each frequency channel, extracts a window of the power trace
    around the reference index (adjusted for the channel's timing
    offset), converts distance-from-reference into physical distance
    using the speed of light and baud rate, and plots the whole 2D array
    (distance vs. frequency) as a colormesh in dB.

    Parameters
    ----------
    freqs : ndarray
        Array of channel frequencies, in Hz.
    corr : ndarray, shape (n_channels, 2, n_samples)
        Correlation data for each channel and polarization.
    ref_idx : int
        Reference (baseline) index, e.g. from find_ref_idx, marking the
        effective transmit time in baseline samples.
    dist_range : list of float, optional
        [min, max] distance range (km) to display around the reference,
        default [-500, 500].
    which_pol : str, optional
        Currently unused placeholder for selecting a polarization
        (default "total", meaning both polarizations are combined).
    pcm_kw : dict, optional
        Extra keyword arguments passed to ax.pcolormesh (e.g. vmin, vmax,
        cmap).
    """
    ref_idx_low = int(dist_range[0] / args.c * args.code_baudrate + ref_idx)
    ref_idx_high = int(dist_range[1] / args.c * args.code_baudrate + ref_idx)
    
    corr_power = np.abs(corr[:, 0, :])**2 + np.abs(corr[:, 1, :])**2

    to_plot = np.zeros([ref_idx_high - ref_idx_low, len(freqs)])

    for i in range(len(freqs)):
        if ref_idx_low + offset(i) < 0:
            to_plot[:, i] = np.nan
        elif ref_idx_high + offset(i) > corr.shape[2]:
            to_plot[:, i] = np.nan
        else:
            # print(0, ref_idx_low + offset(i), ref_idx_high + offset(i), corr.shape[2])
            to_plot[:, i] = corr_power[i, ref_idx_low + offset(i, args=args): ref_idx_high + offset(i, args=args)] / np.median(corr_power[i, ref_idx_low + offset(i, args=args): ref_idx_high + offset(i, args=args)])

    fig, ax = plt.subplots()

    dists = np.arange(ref_idx_low - ref_idx, ref_idx_high - ref_idx) / args.code_baudrate * args.c
    im = ax.pcolormesh(freqs/1e6, dists / 2, 5 * np.log10(to_plot), **pcm_kw)
    fig.colorbar(im, label='SNR (dB)')
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Range (km)") # Range is distance / 2
    ax.set_title(f"0 km is {args.start_time + ref_idx / args.code_baudrate} @ {freqs[0]/1e6:.2f} MHz")

    fig.savefig(os.path.join(args.out_dir, "std_ionogram.png"))
    np.savetxt(os.path.join(args.out_dir, "ionogram.csv"), to_plot, delimiter=",")
    logger.info("Ionogram saved to %s", os.path.join(args.out_dir, "std_ionogram.png"))

def process_and_plot(args=default_args):
    """Run the full ionogram pipeline on a saved correlation file and plot it.

    Loads frequency and correlation data from an .npz file, automatically
    selects the clearest channel, locates the true reference (transmit)
    index, and produces an ionogram plot.

    Parameters
    ----------
    file_name : str
        Path to an .npz file containing "freqs" and "corr" arrays."""
    data = np.load(os.path.join(args.out_dir, args.corr_name))
    
    freqs, corr = data["freqs"], data["corr"]
    best_channel = pick_best_channel(corr, args=args)
    approx_ref_idx = find_approx_ref_idx(corr, best_channel, args=args)
    ref_idx = find_ref_idx(corr, best_channel, approx_ref_idx, args=args)
    ionogram(freqs, corr, ref_idx, args=args)

if __name__ == "__main__":
    # print("Plotting")

    # file_name = "/scratch/mayas/ionosphere/output/iono_corr_2pols_1746818100_to_1746818115.npz"

    # process_and_plot(file_name)
    
    # default_args.out_dir = "/scratch/mayas/ionograms_testing"
    # file_name = "/scratch/mayas/ionograms_testing/iono_corr_2pols_B_1746818100_to_1746818115.npz"

    process_and_plot()