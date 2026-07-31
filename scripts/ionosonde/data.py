import numpy as np
import sys, os

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as bu
from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde import signal_processing as sp
from albatros_analysis.scripts.ionosonde import ionogram

import logging
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

def get_antenna_objs(idxs, files, nchunks, channels, read_size, args = default_args):
    """Get baseband spectra for all antennas x polarizations

    Parameters
    ----------
    idxs : list
        Starting spectrum number for each antenna.
    files : list
        List of files to process that span start and end timestamps
    nchunks : int
        Number of pfb_size blocks to read and process.
    channels : np.ndarray or list
        Channel numbers to feed IPFB [0,2048), should be present in baseband file.
    read_size
    """
    logger.debug("First file: %s", files[0][0])
    header = bdc.get_header(files[0][0])

    channel_indices = np.where(np.isin(header["channels"], channels))[0]  # channels that are in requested channels
    # These assert statements are here because the ALBATROS backend and
    # low-level unpacking is not designed to handle odd starting chan
    # and odd total chans
    logger.debug("Channel indices to be used: %s", channel_indices)

    if channel_indices.size == 0:
        logger.debug(f"The only channels recorded were {header['channels']}, but {channels} were requested.")
        raise ValueError(f"No requested channels recorded.")
    if channel_indices[0] % 2 == 1:
        raise ValueError("Odd starting channel.")
    if len(channel_indices) % 2 == 1:
        raise ValueError("Odd number of channels.")

    logger.debug("Getting antenna objects.")
    
    antenna_objs = []
    for i in range(args.num_ant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0,  # fileidx is 0 = start idx is inside the first file
            idxs[i],
            read_size,
            nchunks=nchunks,
            channels=channel_indices,
            type="float",
        )
        antenna_objs.append(aa)


    logger.info(
        "Channel indices loaded %s to %s (channels %s to %s), rough bandwidth %.3f MHz",
        aa.obj.channel_idxs[0],
        aa.obj.channel_idxs[-1],
        aa.obj.channels[aa.obj.channel_idxs[0]],
        aa.obj.channels[aa.obj.channel_idxs[-1]],
        len(aa.obj.channel_idxs) * 0.061,
    )
    final_channels = aa.obj.channels[aa.obj.channel_idxs].copy()

    logger.info("Final channels: %s", final_channels)
    return final_channels, antenna_objs

def process_from_data(args=default_args):
    files, idx = bu.get_init_info(args.start_time, int(args.start_time + args.corr_time), args.baseband_dir)
    nchunks = 1
    nchan = len(args.channels)

    ipfb_chunk_size = int(args.corr_time * args.chan_res_init)
    read_size = ipfb_chunk_size - 2 * args.cutsize

    logger.debug("Files: %s", files)
    final_channels, antenna_objs = get_antenna_objs([idx], [files], nchunks, args.channels, read_size, args=args)

    # Setup IPFB
    ipfb = sp.setup_ipfb(args.channels, ipfb_chunk_size) # This takes 10 GB of memory for some reason

    len_timestream = read_size * ipfb.lblock
    ncols = args.buf_len - args.filter_len
    nrows = len_timestream // ncols
    dsamp = int(args.adc_samp_freq * ipfb.lblock / args.len_pfb_init / args.code_baudrate)
    Nts_dc = (nrows * ncols + dsamp - 1) // dsamp  # downsampled length after chopping end bits, essentially ceil

    # Get filter
    hf = sp.get_filter()

    # we'll have to store the end phase for all frequencies to downconvert continuously
    phase_cycles = xp.zeros(len(args.ionosonde_freqs), dtype="float64")
    # we'll have to store the last filter state for all frequencies and polarizations to filter continuously
    filter_state = xp.zeros((len(args.ionosonde_freqs), args.num_pol,
                             args.filter_len), dtype="complex64")


    # Pre-allocate temporary timestreams for downconversion to avoid in-place modification and repeated allocations
    filtered_timestreams = xp.zeros((len(args.ionosonde_freqs),
                                     args.num_pol, Nts_dc),
                                     dtype="complex64") # this takes up roughly 5 GB

    logger.debug("len_timestream (Nts): %s", len_timestream)
    ts_pol0_dc = xp.empty(len_timestream, dtype="complex64") # this takes up roughly 32 GB
    ts_pol1_dc = xp.empty(len_timestream, dtype="complex64") # this takes up roughly 32 GB


    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        for ant_idx in range(args.num_ant):
            chunk = chunks[ant_idx]
            expected_start_specnum = antenna_objs[ant_idx].spec_num_start + (chunk_idx) * read_size

            logger.debug("chunk pol0 shape: %s", chunk["pol0"].shape)
            
            pol0 = bdc.make_continuous_gpu(
                chunk["pol0"],
                chunk["specnums"] - expected_start_specnum,
                xp.arange(0, nchan),
                read_size,
                nchan,
            )

            logger.debug("pol0 shape: %s", pol0.shape)
            pol1 = bdc.make_continuous_gpu(
                chunk["pol1"],
                chunk["specnums"] - expected_start_specnum,
                xp.arange(0, nchan),
                read_size,
                nchan,
            )
            corr = sp.process_one_chunk(pol0, pol1, final_channels,
                                        hf, len_timestream, Nts_dc, ts_pol0_dc,
                                        ts_pol1_dc, ipfb, ant_idx, phase_cycles,
                                        filter_state, filtered_timestreams, args=args)
    
    os.makedirs(args.out_dir, exist_ok = True)
    logger.info("Saving to %s", os.path.join(args.out_dir, args.corr_name))
    np.savez(os.path.join(args.out_dir, args.corr_name), corr = corr, freqs = args.ionosonde_freqs)

    return args.ionosonde_freqs, corr.get()

            
if __name__ == "__main__":
    freqs, corr = process_from_data()
    ionogram.process_and_plot()