import numpy as np
import copy
import sys, os
from cupyx.scipy.signal import decimate

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

import logging
if __name__ == "__main__":
    print("Hello")
    logger = logging.getLogger("albatros_analysis.scripts.ionosonde")
else:
    print(__name__)
    logger = logging.getLogger(__name__)

args = copy.deepcopy(default_args)
out_dir_root = "/scratch/mayas/ionograms_testing"
time = 1746753000
ant = 1
drive_num = 3

args.start_time = time
args.which_ant = ant
args.baseband_dir = f"/scratch/mohanagr/drive{drive_num}_mars_spring2025/baseband/"
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

ant_idx = 0
chunk = chunks[ant_idx]
expected_start_specnum = antenna_objs[ant_idx].spec_num_start + (chunk_idx) * read_size
logger.info(f"expected_start_specnum is {expected_start_specnum}, chunk[\"specnums\"] is {chunk['specnums']}")

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

ts_pol0 = ipfb.ipfb(ant_idx, 0, pol0, thresh=args.ipfb_filt_thresh)
ts_pol1 = ipfb.ipfb(ant_idx, 1, pol1, thresh=args.ipfb_filt_thresh)

len_timestream = len(ts_pol0)
ts_pol0_dc = xp.empty(len_timestream, dtype="complex64")
ts_pol1_dc = xp.empty(len_timestream, dtype="complex64")

new_samp_rate = args.adc_samp_freq * ipfb.lblock / args.len_pfb_init
decimate_factor = int(new_samp_rate / args.code_baudrate)
len_dec_timestream = (len_timestream - 1) // decimate_factor + 1

filtered_timestreams = xp.zeros((len(args.ionosonde_freqs), args.num_pol, len_dec_timestream), dtype="complex64")

logger.info(f"Timestream length is {len_timestream }.")
logger.info(f"Decimation factor is {decimate_factor}.")
logger.info(f"Decimated timestream length should be {len_dec_timestream// decimate_factor}.")
logger.info(f"The lblock used by the IPFB was {ipfb.lblock}.")
logger.info(f"The new sampling rate after the IPFB is {new_samp_rate/1e6:.2f} MHz.")
ipfb_start_freq = final_channels[0] * args.adc_samp_freq/args.len_pfb_init

fi = 0
freq = args.ionosonde_freqs[fi]

ddc_freq = (freq - ipfb_start_freq) / new_samp_rate #normalized

sp.ddc_kernel(ts_pol0, ddc_freq, 0, ts_pol0_dc)
sp.ddc_kernel(ts_pol1, ddc_freq, 0, ts_pol1_dc)

filtered_timestreams[fi, 0, :] = decimate(ts_pol0_dc, q = decimate_factor, ftype='fir')
filtered_timestreams[fi, 1, :] = decimate(ts_pol1_dc, q = decimate_factor, ftype='fir')
