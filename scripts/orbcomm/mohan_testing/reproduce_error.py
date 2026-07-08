import sys
import time
#status as of Feb 14, 2024: after all speed updates, once again compared to jupyter output.
#                           sat delay values match, coarse xcorr values match, SNR matches
sys.path.insert(0, "/home/mohanagr/")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import json
from os import path
import cupy as cp

t1 = 1753133705
t2 = 1753133885
a1_path = '/scratch/mohanagr/summer_2025/baseband/mars1/'
a2_path = '/scratch/mohanagr/summer_2025/baseband/mars5/'
files_a1, idx1 = butils.get_init_info(t1, t2, a1_path)
print("-----init info ant 2--------------")
files_a2, idx2 = butils.get_init_info(t1, t2, a2_path)
print(files_a1, idx1)
print(files_a2, idx2)
channels = np.asarray(bdc.get_header(files_a1[0])["channels"],dtype='int64')
chanstart = np.where(channels == 1834)[0][0]
chanend = np.where(channels == 1852)[0][0]
nchans = chanend - chanstart
size = 1000000
ant1 = bdc.BasebandFileIterator(
    files_a1,
    0,
    idx1,
    size,
    None,
    chanstart=chanstart,
    chanend=chanend,
    type="float",
)
ant2 = bdc.BasebandFileIterator(
    files_a2,
    0,
    idx2,
    size,
    None,
    chanstart=chanstart,
    chanend=chanend,
    type="float",
)

print("---antenna 1----")
chunk1 = ant1.__next__()
print("---antenna 2----")
chunk2 = ant2.__next__()
