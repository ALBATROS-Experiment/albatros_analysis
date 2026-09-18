import logging
if __name__ == "__main__":
    logger = logging.getLogger("albatros_analysis.scripts.ionosonde")
else:
    logger = logging.getLogger(__name__)

import numpy as np
import sys, os

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as bu
from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde import signal_processing as sp

def get_antenna_objs(args = default_args):
    """Get baseband spectra for all antennas x polarizations

    Parameters
    ----------

    channels : np.ndarray or list
        Channel numbers to feed IPFB [0,2048), should be present in baseband file.
    read_size
    """
    files_init, idx_init = bu.get_init_info(args.start_time, int(args.start_time + args.corr_time), args.baseband_dir)
    files = [files_init]
    idxs = [idx_init]
    logger.debug("Files: %s", files)
    logger.debug("Indexes: %s", idxs)

    init_channels = np.arange(args.init_chan_bounds[0], args.init_chan_bounds[1])

    logger.debug("First file: %s", files[0][0])
    header = bdc.get_header(files[0][0])

    channel_indices = np.where(np.isin(header["channels"], init_channels))[0]  # channels that are in requested channels
    # These assert statements are here because the ALBATROS backend and
    # low-level unpacking is not designed to handle odd starting chan
    # and odd total chans
    logger.debug("Channel indices to be used: %s", channel_indices)

    if channel_indices.size == 0:
        logger.debug(f"The only channels recorded were {header['channels']}, but {init_channels} were requested.")
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
            args.read_size,
            nchunks=1,
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

    logger.info("Antenna object channels: %s", aa.obj.channels)
    logger.info("Final channels: %s", final_channels)
    logger.debug(f"Number of antenna objects is {len(antenna_objs)}")
    return final_channels, antenna_objs

def process_from_data(args=default_args):
    final_channels, antenna_objs = get_antenna_objs(args=args)
    chunks = zip(*antenna_objs).__next__()
    # Setup IPFB
    ipfb = sp.setup_ipfb(final_channels, args.ipfb_chunk_size)

    for ant_idx in range(args.num_ant):
        chunk = chunks[ant_idx]
        expected_start_specnum = antenna_objs[ant_idx].spec_num_start
        # Would have to have + (chunk_idx) * read_size if we were reading in multiple chunks

        logger.debug("chunk pol0 shape: %s", chunk["pol0"].shape)
        
        pol0 = bdc.make_continuous_gpu(
            chunk["pol0"],
            chunk["specnums"] - expected_start_specnum, # Indicies of present spectra
            xp.arange(0, len(final_channels)),
            args.read_size,
            len(final_channels),
        )

        logger.debug("pol0 shape: %s", pol0.shape)
        pol1 = bdc.make_continuous_gpu(
            chunk["pol1"],
            chunk["specnums"] - expected_start_specnum,
            xp.arange(0, len(final_channels)),
            args.read_size,
            len(final_channels),
        )
        res = sp.process(ant_idx, expected_start_specnum, pol0, pol1, final_channels, ipfb, args=args)
        corr, final_samp_rate, specnums = res

    os.makedirs(args.out_dir, exist_ok = True)
    logger.info("Saving to %s", os.path.join(args.out_dir, args.corr_name))
    np.savez(os.path.join(args.out_dir, args.corr_name), corr = corr,
             freqs = args.ionosonde_freqs, final_samp_rate = final_samp_rate,
             specnums = specnums)

    return args.ionosonde_freqs, corr.get(), final_samp_rate
            
if __name__ == "__main__":
    process_from_data()