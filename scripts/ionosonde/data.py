import numpy as np
import sys, os

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as bu
from albatros_analysis.scripts.ionosonde import params
from albatros_analysis.scripts.ionosonde import signal_processing as sp

def get_antenna_objs(idxs, files, nchunks, channels, verbose = False):
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
    nant : int, optional
        Number of antennas, by default 1
    """
    print(files[0][0])
    header = bdc.get_header(files[0][0])

    channel_indices = np.where(np.isin(header["channels"], channels))[0]  # channels that are in requested channels
    assert channel_indices[0] % 2 == 0
    assert len(channel_indices) % 2 == 0

    if verbose:
        print("Getting timestream")
        print("\tChannel indices to be used", channel_indices)
    
    antenna_objs = []
    for i in range(params.num_ant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0,  # fileidx is 0 = start idx is inside the first file
            idxs[i],
            params.read_size,
            nchunks=nchunks,
            channels=channel_indices,
            type="float",
        )
        antenna_objs.append(aa)


    # print("channels present", aa.obj.channels)
    print(
        "---------------------------------------------\n",
        "Channel indices loaded",
        aa.obj.channel_idxs[0],
        "to",
        aa.obj.channel_idxs[-1],
        "corresponding to channels",
        aa.obj.channels[aa.obj.channel_idxs[0]],
        "to",
        aa.obj.channels[aa.obj.channel_idxs[-1]],
        "rough bandwidth of",
        len(aa.obj.channel_idxs) * 0.061,
        "MHz",
        "\n---------------------------------------------"
    )
    final_channels = aa.obj.channels[aa.obj.channel_idxs].copy()
    
    return final_channels, antenna_objs

def process_from_data(t_start, t_diff = 15,
                      fpath = "/scratch/mohanagr/drive3_mars_spring2025/baseband/",
                      outdir = f"/scratch/{os.environ.get('USER')}/ionosphere/output"):
    t_end = t_start + t_diff
    files, idx = bu.get_init_info(t_start, t_end, fpath)
    nchunks = 1
    channels = np.arange(64,168)
    nchan = len(channels)

    # Setup IPFB
    ipfb = sp.setup_ipfb(channels) # This takes 10 GB of memory for some reason

    len_timestream = params.read_size * ipfb.lblock
    ncols = params.buf_len - params.filter_len
    nrows = len_timestream // ncols
    Nts_dc = (nrows * ncols + params.dsamp - 1) // params.dsamp  # downsampled length after chopping end bits, essentially ceil

    # Get filter
    hf = sp.get_filter()

    # we'll have to store the end phase for all frequencies to downconvert continuously
    phase_cycles = xp.zeros(len(params.ionosonde_freqs), dtype="float64")
    # we'll have to store the last filter state for all frequencies and polarizations to filter continuously
    filter_state = xp.zeros((len(params.ionosonde_freqs), params.num_pol,
                             params.filter_len), dtype="complex64")


    # Pre-allocate temporary timestreams for downconversion to avoid in-place modification and repeated allocations
    filtered_timestreams = xp.zeros((len(params.ionosonde_freqs),
                                     params.num_pol, Nts_dc),
                                     dtype="complex64") # this takes up roughly 5 GB

    print("len_timestream AKA Nts", len_timestream)
    ts_pol0_dc = xp.empty(len_timestream, dtype="complex64") # this takes up roughly 32 GB
    ts_pol1_dc = xp.empty(len_timestream, dtype="complex64") # this takes up roughly 32 GB
    
    print(files)
    final_channels, antenna_objs = get_antenna_objs([idx], [files], nchunks, channels)

    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        for ant_idx in range(params.num_ant):
            chunk = chunks[ant_idx]
            expected_start_specnum = antenna_objs[ant_idx].spec_num_start + (chunk_idx) * params.read_size

            print("chunk pol0", chunk["pol0"].shape)
            
            pol0 = bdc.make_continuous_gpu(
                chunk["pol0"],
                chunk["specnums"] - expected_start_specnum,
                xp.arange(0, nchan),
                params.read_size,
                nchan,
            )

            print("pol0", pol0.shape)
            pol1 = bdc.make_continuous_gpu(
                chunk["pol1"],
                chunk["specnums"] - expected_start_specnum,
                xp.arange(0, nchan),
                params.read_size,
                nchan,
            )
            corr = sp.process_one_chunk(pol0, pol1, final_channels,
                                        hf, len_timestream, Nts_dc, ts_pol0_dc,
                                        ts_pol1_dc, ipfb, ant_idx, phase_cycles,
                                        filter_state, filtered_timestreams)
    
    os.makedirs(outdir, exist_ok = True)
    fname = f"iono_corr_2pols_{t_start}_to_{t_end}"
    print(f"Saving to {os.path.join(outdir, fname)}")
    np.savez(os.path.join(outdir, fname), corr = corr, freqs = params.ionosonde_freqs)

            
if __name__ == "__main__":
    process_from_data(t_start=1746818097)