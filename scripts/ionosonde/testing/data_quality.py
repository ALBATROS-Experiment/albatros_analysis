import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde import data


def get_data(specnum_start, specnum_end, args = default_args):
    """
    Returns
    final_channels : (n_chan,) array of channel numbers
    pols_by_ant    : list (one entry per antenna) of complex arrays with shape
                     (2, n_spec, n_chan), where axis 0 is the polarization
    """
    final_channels, antenna_objs = data.get_antenna_objs(args=args)
    chunks = zip(*antenna_objs).__next__()

    pols_by_ant = []
    for ant_idx in range(args.num_ant):
        chunk = chunks[ant_idx]
        expected_start_specnum = antenna_objs[ant_idx].spec_num_start

        # int() because the specnums can be floats/numpy scalars, which can't be used to slice
        start_idx = int(specnum_start - expected_start_specnum)
        end_idx = int(specnum_end - expected_start_specnum)

        pols = []
        for pol_name in ("pol0", "pol1"):
            pol = bdc.make_continuous_gpu(
                chunk[pol_name],
                chunk["specnums"] - expected_start_specnum, # Indicies of present spectra
                xp.arange(0, len(final_channels)),
                args.read_size,
                len(final_channels),
            )
            pols.append(pol[start_idx:end_idx].get())

        pols_by_ant.append(np.stack(pols))

    return xp.asnumpy(final_channels), pols_by_ant


def make_grid(n_chan):
    """One axes per channel, at most 6 columns. Unused axes are turned off."""
    n_cols = min(6, n_chan)
    n_rows = (n_chan - 1) // n_cols + 1

    # A bit wider than before to leave room for a colorbar next to each axes
    figsize = (n_cols * 2.5, n_rows * 2)

    fig, axs = plt.subplots(n_rows, n_cols, sharex = True, sharey = True,
                            layout = "constrained", figsize = figsize)
    flat_axs = np.atleast_1d(axs).flatten()

    for ax in flat_axs[n_chan:]:
        ax.axis("off")

    return fig, flat_axs


def get_components(pols):
    """Real and imaginary parts of both polarizations, each with shape (n_spec, n_chan)."""
    return {
        "Re(pol0)": pols[0].real,
        "Im(pol0)": pols[0].imag,
        "Re(pol1)": pols[1].real,
        "Im(pol1)": pols[1].imag,
    }


def plot_hists(pols, final_channels, ant_idx = 0, args = default_args):
    n_chan = len(final_channels)

    for pol_idx in range(2):
        fig, flat_axs = make_grid(n_chan)

        for chan_idx in range(n_chan):
            freq = final_channels[chan_idx] * args.chan_res_init

            pol_chan = pols[pol_idx, :, chan_idx]
            # Count each complex value
            counts = np.zeros((15, 15), dtype=int)
            for z in pol_chan:
                counts[int(z.imag) + 7, int(z.real) + 7] += 1

            # Plot
            im = flat_axs[chan_idx].imshow(counts, origin="lower", extent=[-7.5, 7.5, -7.5, 7.5])
            fig.colorbar(im, ax = flat_axs[chan_idx])
            flat_axs[chan_idx].set_title(f"{freq/1e6:.2f} MHz")

        fig.supxlabel("Real")
        fig.supylabel("Imaginary")
        fig.suptitle(f"pol{pol_idx}")

        fig.savefig(os.path.join(args.out_dir, f"baseband_hist_pol{pol_idx}_ant{ant_idx}.png"), transparent = False)


def plot_value_hists(pols, final_channels, ant_idx = 0, args = default_args):
    """Four heat maps: frequency on x, value of Re/Im of each pol on y, counts as color."""
    freqs = final_channels * args.chan_res_init
    values = np.arange(-7, 8)

    fig, axs = plt.subplots(2, 2, sharex = True, sharey = True,
                            layout = "constrained", figsize = (10, 6))
    flat_axs = axs.flatten()

    for ax, (label, comp) in zip(flat_axs, get_components(pols).items()):
        # counts[value, chan]
        counts = np.zeros((len(values), len(freqs)), dtype=int)
        for chan_idx in range(len(freqs)):
            counts[:, chan_idx] = np.bincount(comp[:, chan_idx].astype(int) + 7, minlength = 15)

        im = ax.pcolormesh(freqs / 1e6, values, counts, shading = "nearest")
        fig.colorbar(im, ax = ax)
        ax.set_title(label)

    fig.supxlabel("Frequency (MHz)")
    fig.supylabel("Value")

    fig.savefig(os.path.join(args.out_dir, f"baseband_value_hist_ant{ant_idx}.png"), transparent = False)


def plot_timestreams(pols, final_channels, ant_idx = 0, args = default_args):
    """One axes per frequency; Re/Im of both pols as functions of time."""
    n_chan = len(final_channels)
    # chan_res_init is the number of spectra per second
    t = np.arange(pols.shape[1]) / args.chan_res_init

    fig, flat_axs = make_grid(n_chan)
    components = get_components(pols)

    for chan_idx in range(n_chan):
        freq = final_channels[chan_idx] * args.chan_res_init

        for label, comp in components.items():
            flat_axs[chan_idx].plot(t, comp[:, chan_idx], lw = 0.5, label = label)
        flat_axs[chan_idx].set_title(f"{freq/1e6:.2f} MHz")

    # Put the single legend in the first empty axes, or below the plots if there isn't one
    handles, labels = flat_axs[0].get_legend_handles_labels()
    if n_chan < len(flat_axs):
        flat_axs[n_chan].legend(handles, labels, loc = "center")
    else:
        fig.legend(handles, labels, loc = "outside lower center", ncols = 4)
    fig.supxlabel("Time (s)")
    fig.supylabel("Value")

    fig.savefig(os.path.join(args.out_dir, f"baseband_timestream_ant{ant_idx}.png"), transparent = False)


def plot_waterfall(pols, final_channels, ant_idx = 0, args = default_args):
    """Re/Im of each pol as a function of frequency (x) and time (y)."""
    freqs = final_channels * args.chan_res_init
    t = np.arange(pols.shape[1]) / args.chan_res_init

    fig, axs = plt.subplots(2, 2, sharex = True, sharey = True,
                            layout = "constrained", figsize = (10, 8))

    for ax, (label, comp) in zip(axs.flatten(), get_components(pols).items()):
        im = ax.pcolormesh(freqs / 1e6, t, comp, shading = "nearest") # comp is (n_spec, n_chan)
        fig.colorbar(im, ax = ax)
        ax.set_title(label)

    fig.supxlabel("Frequency (MHz)")
    fig.supylabel("Time (s)")

    fig.savefig(os.path.join(args.out_dir, f"baseband_waterfall_ant{ant_idx}.png"), transparent = False)


def make_plots(specnum_start, specnum_end, args = default_args):
    final_channels, pols_by_ant = get_data(specnum_start, specnum_end, args = args)

    for ant_idx, pols in enumerate(pols_by_ant):
        plot_hists(pols, final_channels, ant_idx = ant_idx, args = args)
        plot_value_hists(pols, final_channels, ant_idx = ant_idx, args = args)
        plot_timestreams(pols, final_channels, ant_idx = ant_idx, args = args)
        plot_waterfall(pols, final_channels, ant_idx = ant_idx, args = args)


if __name__ == "__main__":
    central_specnum = 2621794418 + 150 * default_args.chan_res_init
    default_args.start_time += 150

    # make_plots(central_specnum - 1000, central_specnum + 1000, args = default_args)

    default_args.which_ant = 1
    default_args.baseband_dir = "/scratch/mohanagr/drive3_mars_spring2025/baseband/"
    central_specnum = 2734208734 + 150 * default_args.chan_res_init

    make_plots(central_specnum - 1000, central_specnum + 1000, args = default_args)
