import logging
logger = logging.getLogger(__name__)

import numpy as np
import copy
import sys, os

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as bu
from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde import signal_processing as sp
from albatros_analysis.scripts.ionosonde import ionogram_processing
from albatros_analysis.scripts.ionosonde import ionogram_plotting
from albatros_analysis.scripts.ionosonde import codes
from albatros_analysis.scripts.ionosonde import data

args = copy.deepcopy(default_args)
out_dir_root = "/scratch/mayas/ionograms_testing"
time = 1746753000
ant = 2

args.start_time = time
args.which_ant = ant
args.baseband_dir = f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/"
args.out_dir = f"{out_dir_root}/{str(time)[:5]}/{time}/mars{ant}/"

files, idx = bu.get_init_info(args.start_time, int(args.start_time + args.corr_time), args.baseband_dir)
nchunks = 1

ipfb_chunk_size = int(args.corr_time * args.chan_res_init)
read_size = ipfb_chunk_size - 2 * args.cutsize

logger.debug("Files: %s", files)
init_channels = np.arange(args.init_chan_bounds[0], args.init_chan_bounds[1])
final_channels, antenna_objs = data.get_antenna_objs([idx], [files], nchunks, init_channels, read_size, args=args)
nchan = len(final_channels)

chunk_idx = 0
chunks = zip(*antenna_objs).__next__()

logger.debug(f"Number of antenna objects is {len(antenna_objs)}")

# Setup IPFB
ipfb = sp.setup_ipfb(final_channels, ipfb_chunk_size) # This takes 10 GB of memory for some reason

len_timestream = read_size * ipfb.lblock
ncols = args.buf_len - args.filter_len
nrows = len_timestream // ncols

new_samp_freq = args.adc_samp_freq * ipfb.lblock / args.len_pfb_init
Nts_dc = int(ncols * nrows * args.code_baudrate / new_samp_freq) # downsampled length

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

ant_idx = 0

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

ts_pol0 = ipfb.ipfb(ant_idx, 0, pol0, thresh=args.filt_thresh)
ts_pol1 = ipfb.ipfb(ant_idx, 1, pol1, thresh=args.filt_thresh)

# begin loop over frequencies
new_samp_rate = args.adc_samp_freq * ipfb.lblock / args.len_pfb_init

fi = 0
freq = args.ionosonde_freqs[fi]

ipfb_start_freq = final_channels[0] * args.adc_samp_freq/args.len_pfb_init
ddc_freq = (freq - ipfb_start_freq) / new_samp_rate #normalized

sp.ddc_kernel(ts_pol0, ddc_freq, phase_cycles[fi], ts_pol0_dc)
sp.ddc_kernel(ts_pol1, ddc_freq, phase_cycles[fi], ts_pol1_dc)

# update the phase for the next chunk. Remember two_pi_t starts from 0.

phase_cycles[fi] += ddc_freq * len_timestream
phase_cycles[fi] -= xp.floor(phase_cycles[fi]) # keep it between 0 and 1
# print("new phase cycles", phase_cycles[fi])

# filter the downconverted ts, sampling rate 5 us
# FIX: Use separate filter states for pol0 and pol1
ts_pol0_filt = sp.apply_filter(ts_pol0_dc, hf, filter_state[fi, 0], args=args)
ts_pol1_filt = sp.apply_filter(ts_pol1_dc, hf, filter_state[fi, 1], args=args)
# print("ts_pol0_filt shape is", ts_pol0_filt.shape, "and ts_pol1_filt shape is", ts_pol1_filt.shape)
filtered_timestreams[fi, 0, :] = sp.downsample(ts_pol0_filt, ipfb.lblock, args=args)
filtered_timestreams[fi, 1, :] = sp.downsample(ts_pol1_filt, ipfb.lblock, args=args)

dsamp = int(args.adc_samp_freq * ipfb.lblock / args.len_pfb_init / args.code_baudrate)

print(dsamp)
print(args.adc_samp_freq * ipfb.lblock / args.len_pfb_init / args.code_baudrate)
print(ts_pol0_filt.shape)
print(filtered_timestreams[fi, 0, :].shape)