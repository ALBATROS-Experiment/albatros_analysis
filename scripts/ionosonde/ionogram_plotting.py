### IMPORTS ###

# Array-handling
import numpy as np

# Timing
from datetime import datetime
from datetime import timezone as tz
import time

# System utils
import sys, os

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use(f'{os.path.expanduser("~")}/albatros_analysis/scripts/ionosonde/iono.mplstyle')

# Adds the home directory to the path so that we can import from albatros_analysis
# This won't work if albatros_analysis isn't in your home directory
sys.path.insert(0, os.path.expanduser("~"))

# Nicely packaged up parameters
from albatros_analysis.scripts.ionosonde.params import default_args

import logging
logger = logging.getLogger(__name__)

def plot_traces(freqs, dists, total_corr_db, chan_indices = None,
                figsize = None, args = default_args):
    """Plot the power trace of a set of frequency channels around ref_idx.

    Parameters
    ----------
    freqs : ndarray
        Array of channel frequencies, in Hz.
    dists : ndarray
        Distance axis (km) relative to the reference index. The spacing should be c/code_baudrate.
    total_corr_db : ndarray, shape (n_samples, n_channels)
        Median-normalized power (dB).

    chan_indices : array-like of int, optional
        Indices (into the full freqs/total_corr_db) of the channels to
        plot. If None (default), all channels are plotted. Selecting a
        subset this way (rather than slicing freqs/total_corr_db
        yourself) keeps each channel's timing offset correct.
    figsize : tuple of float, optional
        Size of the resulting figure, in inches. Defaults to a size that
        scales with the number of channels being plotted.

    Returns
    -------
    None
        The figure is saved to disk (under args.out_dir) and closed
        before this function returns; nothing is returned.
    """

    if chan_indices is None:
        chan_indices = np.arange(len(freqs))
        file_name = "traces.png"
    else:
        chan_indices = np.asarray(chan_indices)
        file_name = f"traces_{'_'.join(chan_indices.astype(str))}.png"

    n_chan = len(chan_indices)
    n_cols = min(3, n_chan)
    n_rows = int(np.ceil(n_chan / n_cols))

    if figsize is None:
        figsize = (n_cols * 2.5, n_rows * 2)

    fig, axs = plt.subplots(n_rows, n_cols, sharex = True, sharey = True,
                            layout = "constrained", figsize = figsize)
    flat_axs = np.atleast_1d(axs).flatten()

    for i, chan in enumerate(chan_indices):
        flat_axs[i].plot(dists / 2, total_corr_db[:, chan], c = (0.5, 0, 0))
        flat_axs[i].set_title(f"{freqs[chan]/1e6:.2f} MHz")

    # Hide any unused axes in the grid (when n_chan doesn't fill it exactly).
    for j in range(n_chan, len(flat_axs)):
        flat_axs[j].axis("off")

    fig.supxlabel("Relative range (km)")
    flat_axs[0].set(ylim = (0, None))
    fig.supylabel("Correlated signal (dB)")

    os.makedirs(args.out_dir, exist_ok=True)
    file_path = os.path.join(args.out_dir, file_name)
    fig.savefig(file_path)
    logger.info("Traces plot saved to %s", file_path)

    plt.show()
    plt.close(fig)

def ionogram(freqs, dists, total_corr_db, ref_idx,
             pcm_kw = {"vmin": 0, "vmax": 15, "cmap": "jet"}, figsize = None,
            plasma_freq = None, ref_freq = None, args = default_args):
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
    dists : ndarray
        Distance axis (km) relative to the reference index, e.g. as
        returned by extract_traces.
    total_corr_db : ndarray, shape (n_samples, n_channels)
        Median-normalized power (dB) for each channel, e.g. as returned
        by extract_traces.
    ref_idx : int
        Reference (baseline) index, e.g. from find_ref_idx, marking the
        effective transmit time in baseline samples.
    pcm_kw : dict, optional
        Extra keyword arguments passed to ax.pcolormesh (e.g. vmin, vmax,
        cmap).
    figsize : tuple of float, optional
        Size of the resulting figure, in inches. Defaults to matplotlib's
        default figure size.
    plasma_freq : float or None, optional
        Estimated plasma frequency (Hz), e.g. from find_plasma_freq. If
        given, it's marked with a dashed vertical line on the plot
        (default None, in which case no line is drawn).
    ref_freq : int, optional
        Index of the channel used as the timing reference (e.g.
        best_channel from process_and_plot). Used only to label the
        y-axis.

    Returns
    -------
    None
        The figure is saved to disk (under args.out_dir) and closed
        before this function returns; nothing is returned.
    """ 
    fig, ax = plt.subplots(figsize=figsize, layout = "constrained")

    im = ax.pcolormesh(freqs/1e6, dists / 2, total_corr_db, **pcm_kw)
    fig.colorbar(im, label='SNR (dB)')
    ax.set_xlabel("Frequency (MHz)")
    if ref_freq is None:
        ax.set_ylabel("Range (km)")
    else:
        ax.set_ylabel(f"Range (km) relative to {freqs[ref_freq]/1e6:.2f} MHz peak") # Range is distance / 2

    # Round start time to nearest five minutes
    time_to_five_min = round(args.start_time / 300) * 300
    ax.set_title(f"{datetime.fromtimestamp(time_to_five_min, tz = tz.utc)}, MARS {args.which_ant}")

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

    logger.info("Ionogram saved to %s", os.path.join(args.out_dir, "std_ionogram.png"))

    plt.show()
    plt.close(fig)