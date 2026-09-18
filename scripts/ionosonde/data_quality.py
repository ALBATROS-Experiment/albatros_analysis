import logging
if __name__ == "__main__":
    logger = logging.getLogger("albatros_analysis.scripts.ionosonde")
else:
    logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as bu
from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde import signal_processing as sp
from albatros_analysis.scripts.ionosonde import data
from albatros_analysis.scripts.ionosonde.ionogram_processing import offset

def make_hists_iono_offset(specnum_start, specnum_end, args = default_args):
    final_channels, antenna_objs = data.get_antenna_objs(args=args)
    chunks = zip(*antenna_objs).__next__()
    # Setup IPFB
    ipfb = sp.setup_ipfb(final_channels, args.ipfb_chunk_size)

    for ant_idx in range(args.num_ant):
        chunk = chunks[ant_idx]
        expected_start_specnum = antenna_objs[ant_idx].spec_num_start
        # Would have to have + (chunk_idx) * read_size if we were reading in multiple chunks
        
        pol0 = bdc.make_continuous_gpu(
            chunk["pol0"],
            chunk["specnums"] - expected_start_specnum, # Indicies of present spectra
            xp.arange(0, len(final_channels)),
            args.read_size,
            len(final_channels),
        )

        pol1 = bdc.make_continuous_gpu(
            chunk["pol1"],
            chunk["specnums"] - expected_start_specnum,
            xp.arange(0, len(final_channels)),
            args.read_size,
            len(final_channels),
        )

        n_chan = len(final_channels)
        n_cols = min(6, n_chan)
        n_rows = (n_chan - 1) // n_cols + 1

        figsize = (n_cols * 2.5, n_rows * 2)

        fig, axs = plt.subplots(n_rows, n_cols, sharex = True, sharey = True,
                                layout = "constrained", figsize = figsize)
        flat_axs = np.atleast_1d(axs).flatten()

        for chan_idx in range(n_chan):
            freq = final_channels[chan_idx] * args.chan_res_init
            # Find nearest ionosonde frequency
            iono_idx = np.argmin(np.abs(args.ionosonde_freqs - freq))
            iono_freq = args.ionosonde_freqs[iono_idx]

            start_idx = specnum_start + offset(iono_idx, args.chan_res_init, args = args) - expected_start_specnum
            end_idx = specnum_end + offset(iono_idx, args.chan_res_init, args = args) - expected_start_specnum
            # print(chan_idx, start_idx, end_idx)
            # print(pol0[start_idx:end_idx, chan_idx])

            pol1_chan = pol1[start_idx:end_idx, chan_idx]
            # Count each complex value
            counts = np.zeros((15, 15), dtype=int)
            for z in pol1_chan:
                counts[int(z.imag) + 7, int(z.real) + 7] += 1

            # Plot
            flat_axs[chan_idx].imshow(counts, origin="lower", extent=[-7.5, 7.5, -7.5, 7.5])
            flat_axs[chan_idx].set_title(f"{freq/1e6:.2f} MHz\n(closest freq {iono_freq/1e6:.2f} MHz)")

        fig.supxlabel("Real")
        fig.supylabel("Imaginary")
        # fig.colorbar(label="Count")
        
        fig.savefig(os.path.join(args.out_dir, "baseband_hist.png"))

def make_hists(specnum_start, specnum_end, args = default_args):
    final_channels, antenna_objs = data.get_antenna_objs(args=args)
    chunks = zip(*antenna_objs).__next__()
    # Setup IPFB
    ipfb = sp.setup_ipfb(final_channels, args.ipfb_chunk_size)

    for ant_idx in range(args.num_ant):
        chunk = chunks[ant_idx]
        expected_start_specnum = antenna_objs[ant_idx].spec_num_start
        # Would have to have + (chunk_idx) * read_size if we were reading in multiple chunks
        
        pol0 = bdc.make_continuous_gpu(
            chunk["pol0"],
            chunk["specnums"] - expected_start_specnum, # Indicies of present spectra
            xp.arange(0, len(final_channels)),
            args.read_size,
            len(final_channels),
        )

        pol1 = bdc.make_continuous_gpu(
            chunk["pol1"],
            chunk["specnums"] - expected_start_specnum,
            xp.arange(0, len(final_channels)),
            args.read_size,
            len(final_channels),
        )

        n_chan = len(final_channels)
        n_cols = min(6, n_chan)
        n_rows = (n_chan - 1) // n_cols + 1

        figsize = (n_cols * 2, n_rows * 2)

        fig0, axs0 = plt.subplots(n_rows, n_cols, sharex = True, sharey = True,
                                layout = "constrained", figsize = figsize)
        fig1, axs1 = plt.subplots(n_rows, n_cols, sharex = True, sharey = True,
                                layout = "constrained", figsize = figsize)
                            
        flat_axs0 = np.atleast_1d(axs0).flatten()
        flat_axs1 = np.atleast_1d(axs1).flatten()

        for chan_idx in range(n_chan):
            freq = final_channels[chan_idx] * args.chan_res_init

            start_idx = specnum_start - expected_start_specnum
            end_idx = specnum_end - expected_start_specnum
            
            pol0_chan = pol0[start_idx:end_idx, chan_idx]
            pol1_chan = pol1[start_idx:end_idx, chan_idx]
            # Count each complex value
            counts0 = np.zeros((15, 15), dtype=int)
            for z in pol0_chan:
                counts0[int(z.imag) + 7, int(z.real) + 7] += 1
            counts1 = np.zeros((15, 15), dtype=int)
            for z in pol1_chan:
                counts1[int(z.imag) + 7, int(z.real) + 7] += 1
            
            # Plot
            flat_axs0[chan_idx].imshow(counts0, origin="lower", extent=[-7.5, 7.5, -7.5, 7.5])
            flat_axs0[chan_idx].set_title(f"{freq/1e6:.2f} MHz")

            flat_axs1[chan_idx].imshow(counts1, origin="lower", extent=[-7.5, 7.5, -7.5, 7.5])
            flat_axs1[chan_idx].set_title(f"{freq/1e6:.2f} MHz")

        for chan_idx in range(n_chan, n_rows * n_cols):
            flat_axs0[chan_idx].axis("off")
            flat_axs1[chan_idx].axis("off")


        fig0.supxlabel("Real")
        fig0.supylabel("Imaginary")

        fig1.supxlabel("Real")
        fig1.supylabel("Imaginary")

        fig0.suptitle("pol0")
        fig1.suptitle("pol1")
        
        fig0.savefig(os.path.join(args.out_dir, "baseband_hist_pol0.png"), transparent = False)
        fig1.savefig(os.path.join(args.out_dir, "baseband_hist_pol1.png"), transparent = False)

if __name__ == "__main__":
    central_specnum = 2621794418 + 150 * default_args.chan_res_init
    default_args.start_time += 150

    make_hists(central_specnum - 1000, central_specnum + 1000, args = default_args)