# Goal: check to see that if we change the number of channels we request, the data read won't be any different

import logging
logger = logging.getLogger(__name__)

import numpy as np
import copy
import sys, os
from cupyx.scipy.signal import decimate

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src import xp

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as bu
from albatros_analysis.scripts.ionosonde import data

files, idx = bu.get_init_info(1746795000, 1746795020, "/scratch/mohanagr/drive2_mars_spring2025/baseband/")
print(f"FILES: {files}, IDX: {idx}")
nchunks = 1

read_size = 1_000_000

channels1 = np.arange(64, 168)
channels2 = np.arange(84, 104)

idxs, files = [idx], [files]

header = bdc.get_header(files[0][0])

channel_indices1 = np.where(np.isin(header["channels"], channels1))[0]  # channels that are in requested channels
channel_indices2 = np.where(np.isin(header["channels"], channels2))[0]

aa1 = bdc.BasebandFileIterator(
    files[0],
    0,  # fileidx is 0 = start idx is inside the first file
    idxs[0],
    read_size,
    nchunks=nchunks,
    channels=channel_indices1,
    type="float",
)

antenna_objs1 = [aa1]

final_channels1 = aa1.obj.channels[aa1.obj.channel_idxs].copy()

print("Antenna object channels 1:", aa1.obj.channels)
print("Final channels 1:", final_channels1)

nchan1 = len(final_channels1)

chunk_idx = 0
chunks1 = zip(*antenna_objs1).__next__()

ant_idx = 0
chunk1 = chunks1[ant_idx]

expected_start_specnum = antenna_objs1[ant_idx].spec_num_start + (chunk_idx) * read_size


print(chunk1["specnums"])
print(len(chunk1["specnums"]))
print(expected_start_specnum)