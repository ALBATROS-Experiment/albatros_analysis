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
        # Make sure the figure gets cleaned up once we're done with it.
        plt.close(fig)

    final_idx = int((argmax + args.code_repeat_num + 3)
                    * (2 * args.ipp * args.code_baudrate)) + new_low_idx

    return final_idx - offset(best_channel, args=args)


def extract_traces(freqs, corr, ref_idx, dist_range = [-2000, 2000], args = default_args):
    """Extract each channel's windowed trace around ref_idx.

    For each frequency channel, extracts a window of the correlation data
    around the reference index (adjusted for that channel's timing
    offset), and normalizes the power by the channel's median power over
    that window.

    Unlike a naive implementation that discards an entire channel to NaN
    whenever any part of its window falls outside the bounds of the
    available data, this function pulls in whatever real data does exist
    for a channel (clipping the requested window to the available range)
    and only pads the genuinely missing samples with NaN. This minimizes
    the total number of NaNs in the output.

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
        [min, max] distance range (km) to extract around the reference,
        default [-2000, 2000].

    Returns
    -------
    normalized : ndarray, shape (n_samples, n_channels)
        Median-normalized power for each channel, padded with NaN only
        where no real data was available.
    pol0 : ndarray, complex, shape (n_samples, n_channels)
        Raw polarization-0 correlation values for each channel.
    pol1 : ndarray, complex, shape (n_samples, n_channels)
        Raw polarization-1 correlation values for each channel.
    dists : ndarray
        Distance axis (km) relative to ref_idx, corresponding to the
        sample window used for every channel.
    """
    ref_idx_low = int(dist_range[0] / args.c * args.code_baudrate + ref_idx)
    ref_idx_high = int(dist_range[1] / args.c * args.code_baudrate + ref_idx)

    corr_power = np.abs(corr[:, 0, :])**2 + np.abs(corr[:, 1, :])**2

    n_samples = ref_idx_high - ref_idx_low
    n_chan = len(freqs)

    normalized = np.full([n_samples, n_chan], np.nan)
    pol0 = np.full([n_samples, n_chan], np.nan, dtype=np.complex64)
    pol1 = np.full([n_samples, n_chan], np.nan, dtype=np.complex64)

    for i in range(n_chan):
        low_idx = ref_idx_low + offset(i, args=args)
        high_idx = ref_idx_high + offset(i, args=args)

        # Clip the requested window to the range of data that actually
        # exists, so we pull in as much real data as possible instead of
        # throwing away the whole channel when only part of its window
        # lies outside corr's bounds.
        valid_low = max(low_idx, 0)
        valid_high = min(high_idx, corr.shape[2])

        if valid_low >= valid_high:
            # No overlap at all with existing data for this channel;
            # it stays NaN.
            continue

        # Where the valid data lands within the (fixed-size) output window.
        out_start = valid_low - low_idx
        out_end = out_start + (valid_high - valid_low)

        med = np.median(corr_power[i, valid_low:valid_high])

        normalized[out_start:out_end, i] = corr_power[i, valid_low:valid_high] / med
        pol0[out_start:out_end, i] = corr[i, 0, valid_low:valid_high]
        pol1[out_start:out_end, i] = corr[i, 1, valid_low:valid_high]

    dists = np.arange(ref_idx_low - ref_idx, ref_idx_high - ref_idx) / args.code_baudrate * args.c

    return normalized, pol0, pol1, dists


def plot_all_traces(freqs, corr, ref_idx, dist_range = [-2000, 2000], figsize = None, args = default_args):
    """Plot the power trace of every frequency channel around ref_idx.

    Uses extract_traces to pull each channel's windowed trace (with as
    little data loss as possible), then plots the combined (both
    polarizations) power for each channel against distance, in its own
    subplot within a single grid figure.

    Parameters
    ----------
    freqs : ndarray
        Array of channel frequencies, in Hz.
    corr : ndarray, shape (n_channels, 2, n_samples)
        Correlation data for each channel and polarization.
    ref_idx : int
        Reference (baseline) index, e.g. from find_ref_idx.
    dist_range : list of float, optional
        [min, max] distance range (km) to display around the reference,
        default [-2000, 2000].
    figsize : tuple of float, optional
        Size of the resulting figure, in inches. Defaults to a size that
        scales with the number of channels.

    Returns
    -------
    fig, axs
        The created matplotlib figure and array of axes (already saved
        to disk and closed by the time this function returns).
    """
    normalized, pol0, pol1, dists = extract_traces(freqs, corr, ref_idx,
                                                     dist_range=dist_range, args=args)

    n_chan = len(freqs)
    n_cols = int(np.ceil(np.sqrt(n_chan)))
    n_rows = int(np.ceil(n_chan / n_cols))

    if figsize is None:
        figsize = (n_cols * 2.2, n_rows * 1.8)

    fig, axs = plt.subplots(n_rows, n_cols, sharex = True, sharey = True,
                            layout = "constrained", figsize = figsize)
    flat_axs = np.atleast_1d(axs).flatten()

    power = np.abs(pol0)**2 + np.abs(pol1)**2

    for i in range(n_chan):
        flat_axs[i].plot(dists / 2, power[:, i], c = (0.5, 0, 0))
        flat_axs[i].set_title(f"{freqs[i]/1e6:.2f} MHz", fontsize = 8)

    # Hide any unused axes in the grid (when n_chan doesn't fill it exactly).
    for j in range(n_chan, len(flat_axs)):
        flat_axs[j].axis("off")

    fig.supxlabel("Range (km)")
    fig.supylabel("Power")

    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "all_traces.png"))
    logger.info("All-traces plot saved to %s", os.path.join(args.out_dir, "all_traces.png"))

    plt.close(fig)

    return fig, axs


def find_plasma_freq(freqs, normalized, snr_threshold_db = 6):
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
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = 5 * np.log10(normalized)
 
    detectable = np.zeros(len(freqs), dtype=bool)
    for i in range(len(freqs)):
        col = power_db[:, i]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            continue
        detectable[i] = np.any(valid >= snr_threshold_db)
 
    if not np.any(detectable) or np.all(detectable): # TODO: Change this to look at highest freq
        return None
 
    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    sorted_detectable = detectable[order]
 
    detected_positions = np.where(sorted_detectable)[0]
    highest_pos = detected_positions.max()
 
    # Guaranteed to exist since not all channels are detectable.
    next_pos = highest_pos + 1
 
    return (sorted_freqs[highest_pos] + sorted_freqs[next_pos]) / 2


def ionogram(freqs, corr, ref_idx, dist_range = [-2000, 2000], which_pol = "total",
             pcm_kw = {"vmin": 0, "vmax": 15, "cmap": "jet"}, figsize = None,
             plasma_snr_threshold = 6, args = default_args):
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
        default [-2000, 2000].
    which_pol : str, optional
        Currently unused placeholder for selecting a polarization
        (default "total", meaning both polarizations are combined).
    pcm_kw : dict, optional
        Extra keyword arguments passed to ax.pcolormesh (e.g. vmin, vmax,
        cmap).
    figsize : tuple of float, optional
        Size of the resulting figure, in inches. Defaults to matplotlib's
        default figure size.
    plasma_snr_threshold : float, optional
        SNR threshold (dB) passed to find_plasma_freq to decide whether a
        channel has a detectable signal (default 6 dB).

    Returns
    -------
    normalized, pol0, pol1 : ndarray
        As returned by extract_traces.
    plasma_freq : float or None
        Estimated plasma frequency (Hz), from find_plasma_freq, or None
        if no channel showed a detectable signal.
    """
    normalized, pol0, pol1, dists = extract_traces(freqs, corr, ref_idx,
                                                     dist_range=dist_range, args=args)

    plasma_freq = find_plasma_freq(freqs, normalized, snr_threshold_db=plasma_snr_threshold)

    fig, ax = plt.subplots(figsize=figsize, layout = "constrained")

    im = ax.pcolormesh(freqs/1e6, dists / 2, 5 * np.log10(normalized), **pcm_kw)
    fig.colorbar(im, label='SNR (dB)')
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Range (km)") # Range is distance / 2
    ax.set_title(f"MARS {args.which_ant}")

    if plasma_freq is not None:
        ax.axvline(plasma_freq / 1e6, color="white", linestyle="--", linewidth=1,
                   label=f"$f_p$ \u2248 {plasma_freq/1e6:.2f} MHz")
        ax.legend(loc="upper right", fontsize=8)
        logger.info("Estimated plasma frequency: %.3f MHz", plasma_freq / 1e6)
    else:
        logger.info("No channel showed a detectable signal; plasma frequency not found.")

    # Secondary y-axis showing the UNIX time corresponding to each range,
    # referenced to the lowest frequency channel (freqs[0], which has zero
    # timing offset relative to ref_idx).
    base_time = args.start_time + ref_idx / args.code_baudrate
    base_time_floor = int(base_time)
    base_time_diff = base_time - base_time_floor
    
    def _range_to_time(range_km):
        two_way_dist_km = np.asarray(range_km) * 2
        return base_time_diff + two_way_dist_km / args.c

    def _time_to_range(unix_time):
        two_way_dist_km = (np.asarray(unix_time) - base_time_diff) * args.c
        return two_way_dist_km / 2

    secax = ax.secondary_yaxis("right", functions=(_range_to_time, _time_to_range))
    secax.set_ylabel(f"UNIX Time (s) @ {freqs[0]/1e6:.2f} MHz – {base_time_floor}")

    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, "std_ionogram.png"))
    np.savez(os.path.join(args.out_dir, "ionogram.npz"),
             normalized=normalized, pol0=pol0, pol1=pol1)
    logger.info("Ionogram saved to %s", os.path.join(args.out_dir, "std_ionogram.png"))

    plt.show()
    plt.close(fig)

    return normalized, pol0, pol1, plasma_freq

def process_and_plot(ref_idx_plotting = False, cutoff_plot = 500, dist_range = [-2000, 2000],
                      which_pol = "total", pcm_kw = {"vmin": 0, "vmax": 15, "cmap": "jet"},
                      figsize = None, plasma_snr_threshold = 6, args=default_args):
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
    best_channel = pick_best_channel(corr, args=args)
    approx_ref_idx = find_approx_ref_idx(corr, best_channel, args=args)
    ref_idx = find_ref_idx(corr, best_channel, approx_ref_idx, plotting=ref_idx_plotting,
                            cutoff_plot=cutoff_plot, args=args)
    ionogram(freqs, corr, ref_idx, dist_range=dist_range, which_pol=which_pol,
             pcm_kw=pcm_kw, figsize=figsize, plasma_snr_threshold=plasma_snr_threshold,
             args=args)

if __name__ == "__main__":
    # print("Plotting")

    # file_name = "/scratch/mayas/ionosphere/output/iono_corr_2pols_1746818100_to_1746818115.npz"

    # process_and_plot(file_name)
    
    # default_args.out_dir = "/scratch/mayas/ionograms_testing"
    # file_name = "/scratch/mayas/ionograms_testing/iono_corr_2pols_B_1746818100_to_1746818115.npz"

    process_and_plot() 