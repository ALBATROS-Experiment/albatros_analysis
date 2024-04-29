# in this version I will not store fine xcorr for all mu-chunks. only store the xcorrs' xcorr peak
# later, to align the mu-chunks, we can save a very small slice of the muchunks

import sys
import os
import time

sys.path.insert(0, "/home/s/sievers/mohanagr/albatros_analysis/")

from src.correlations import baseband_data_classes as bdc

# from albatros_analysis.correlations import correlations as cr
from src.utils import baseband_utils as butils
from src.utils import orbcomm_utils as outils
from src.utils import mkfftw as mk
import numpy as np
import datetime
from matplotlib import pyplot as plt
from os import path

def xcorr_parab(arr1,arr2):
    xc = outils.get_coarse_xcorr_fast2(arr1,arr2, 50,Npfb=1)
    m=np.argmax(xc[0,:].real)
    params=np.polyfit(np.arange(10),xc[0,m-5:m+5].real,2)
    return -params[1]/params[0]/2 + m -50


T_SPECTRA = 4096 / 250e6
T_ACCLEN = 393216 * T_SPECTRA
DEBUG = False

deployment_yyyymm = "202210"
# deployment_yyyymm="202107"
ant1_snap = "snap3"
ant2_snap = "snap4"
# ant2_snap = "snap1"
base_path = path.join("/project/s/sievers/albatros/uapishka",deployment_yyyymm)
out_path = "/scratch/s/sievers/mohanagr/"


# ----------------BASEBAND SETTINGS--------------------------------------------#
tstart = 1667015968
tt1 = tstart + 372 * T_ACCLEN  # OG is 150.
tt2 = tstart + 504 * T_ACCLEN
print(f"Start time: {tt1:.3f}, End time: {tt2:.3f}, Duration: {tt2-tt1:.3f}")
files_a1, idx1 = butils.get_init_info(
    tt1, tt2, path.join(base_path,"baseband", ant1_snap)
)
files_a2, idx2 = butils.get_init_info(
    tt1, tt2, path.join(base_path,"baseband", ant2_snap)
)
hdr1 = bdc.get_header(files_a1[0])
chan_1839_idx = np.where(hdr1["channels"] == 1839)[0][0]
# -----------------------------------------------------------------------------#
# ----------------SATELLITE SETTINGS-------------------------------------------#
a1_coords = [51.4646065, -68.2352594, 341.052]  # north antenna
a2_coords = [51.46418956, -68.23487849, 338.32526665]  # south antenna
# satnorads = [40087, 41187]
# sat2chan = {41187: [1840, 1844], 40087: [1846, 1847]}
# chan2sat = {1840: 41187, 1844: 41187, 1846: 40087, 1847: 40087}
# corr_chans = np.asarray([1840, 1844, 1846, 1847], dtype=int)
corr_chans = np.asarray([1840, 1844], dtype=int)
satnorads = [41188]
sat2chan = {41188: [1840,1844]}
chan2sat = {1840: 41188, 1844: 41188}
channel_idx = corr_chans - 1839 + chan_1839_idx
niter = int(tt2 - tt1) + 2  # run it for an extra second to avoid edge effects
delays = {}
og_delays = {}
times = np.arange(0, niter)  # time in seconds
# -----------------------------------------------------------------------------#
# ----------------AVERAGING SETTINGS-------------------------------------------#
size_micro_chunk = 10000
num_micro_chunks_per_block = 61
num_blocks = 1
num_micro_chunks = num_blocks * num_micro_chunks_per_block
if (num_micro_chunks * size_micro_chunk * T_SPECTRA) > (tt2-tt1):
    print("You've loaded less files than needed. load more")
    exit(1)
# -----------------------------------------------------------------------------#
# ---------------UPSAMPLING SETTINGS-------------------------------------------#
osamp = 40
N = 2 * size_micro_chunk
# dN = int(0.5*size_micro_chunk)
dN = min(
    100000, int(0.3 * N)
)  # size of coarse xcorr stamp returned will be -dN to dN, total size 2*dN
dN_interp = 10  # stamp size for calculating upsampled xcorr for each chan.
sample_no = np.arange(0, (2 * dN_interp - 1) * 4096 * osamp, dtype="float64")
coarse_sample_no = np.arange(0, 2 * dN_interp, dtype="float64") * 4096 * osamp
UPSAMP_CENTER = len(sample_no)//2
UPSAMP_dN = dN_interp * 4096 * osamp
interp_xcorr = np.empty((len(corr_chans), len(sample_no)), dtype="complex128")
chunk_nums = np.zeros(num_micro_chunks-1,dtype=int) #store the block num for xcorr xcorr
cur_xcorr_num = 0
maxvals=[]
idx1, idx2 = outils.delay_corrector(idx1, idx2, 106628, 100000)  # 150 => 40429 180 => 43980

print(
    f"Start channel: {1839:d} at index {chan_1839_idx:d}.\n\
    Num channels: {len(channel_idx):d}.\n\
    Block length: {size_micro_chunk:d}.\n\
    len coarse xcorr: {2*N}'\n\
    len stamp: {2*dN}\n\
    nchunks: {num_micro_chunks:d}\n\
    channels:", hdr1["channels"][channel_idx]
)
print("channel indices", channel_idx)
tle_path = outils.get_tle_file(tstart, "/project/s/sievers/mohanagr/OCOMM_TLES")
print(tle_path)
for i, num in enumerate(satnorads):
    print(i, num)
    og_delays[num] = outils.get_sat_delay(
        a1_coords, a2_coords, tle_path, tt1, niter, num
    )
    delays[num] = np.interp(
        np.arange(num_micro_chunks) * size_micro_chunk * T_SPECTRA,
        times,
        og_delays[num],
    )
print(f"loaded TLEs")

ant1 = bdc.BasebandFileIterator(
    files_a1,
    0,
    idx1,
    size_micro_chunk,
    nchunks=num_micro_chunks,
    channels=channel_idx,
    type="float",
)
ant2 = bdc.BasebandFileIterator(
    files_a2,
    0,
    idx2,
    size_micro_chunk,
    nchunks=num_micro_chunks,
    channels=channel_idx,
    type="float",
)

print("osamp is", osamp)
prev_xcorr = None
# ---------------------#
a1_start = ant1.spec_num_start
a2_start = ant2.spec_num_start
print("a1 start", a1_start, "a2_start", a2_start)
tg1=time.time()
for ii, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
    print(f"Processing micro-chunk {ii}...")
    p0_a1 = np.zeros(
        (size_micro_chunk, len(channel_idx)), dtype="complex128"
    )  # could parallelize
    p0_a2 = np.zeros((size_micro_chunk, len(channel_idx)), dtype="complex128")
    perc_missing_a1 = (1 - len(chunk1["specnums"]) / size_micro_chunk) * 100
    perc_missing_a2 = (1 - len(chunk2["specnums"]) / size_micro_chunk) * 100

    print(f"perc missing ant1 and ant2: {perc_missing_a1:.3f}%, {perc_missing_a2:.3f}%")
    if perc_missing_a1 > 10 or perc_missing_a2 > 10:
        a1_start = ant1.spec_num_start
        a2_start = ant2.spec_num_start
        print("TOO MUCH MISSING, CONTINUING")
        continue
    
    outils.make_continuous(p0_a1, chunk1["pol0"], chunk1["specnums"] - a1_start)
    outils.make_continuous(p0_a2, chunk2["pol0"], chunk2["specnums"] - a2_start)
    a1_start = ant1.spec_num_start
    a2_start = ant2.spec_num_start
    xcorr_stamp = outils.get_coarse_xcorr_fast2(p0_a1, p0_a2, dN)

    if True:
        fig2, ax2 = plt.subplots(np.ceil(xcorr_stamp.shape[0] / 3).astype(int), 3)
        fig2.set_size_inches(12, np.ceil(xcorr_stamp.shape[0] / 3) * 3)
        ax2 = ax2.flatten()
        fig2.suptitle(f"30 values around the center of array of size {2*dN}")
        for i in range(xcorr_stamp.shape[0]):
            ax2[i].set_title(
                f"chan {corr_chans[i]} max: {np.argmax(np.abs(np.abs(xcorr_stamp[i,:])))}"
            )
            ax2[i].plot(np.abs(xcorr_stamp[i, :]))
            ax2[i].set_xlim(dN - 50, dN + 50)
        plt.tight_layout()
        fig2.savefig(
            os.path.join(out_path, f"debug_genph_{tstart}_{ii}_{int(time.time())}.jpg")
        )

    # print("starting upsampling...")

    COARSE_CENTER = xcorr_stamp.shape[1] // 2

    for i, chan in enumerate(corr_chans):
        sat = chan2sat[chan]
        # shift = -int(delays[sat][ii] * 250e6 * osamp) #get delay for chunk num ii
        real_freq = 1-chan/4096
        alia_freq = chan/4096
        shift = delays[sat][ii] * 250e6 * osamp * real_freq/alia_freq
        shifted_samples = sample_no - shift
        # print("shift is ", shift, "for sat", sat)
        outils.get_interp_xcorr_fast(
            xcorr_stamp[i,COARSE_CENTER - dN_interp : COARSE_CENTER + dN_interp],
            chan,
            shifted_samples,
            coarse_sample_no,
            out=interp_xcorr[i, :]
        )

    if prev_xcorr is None:
        prev_xcorr = np.sum(interp_xcorr, axis=0)
    else:
        cur_xcorr = np.sum(interp_xcorr, axis=0)
        peak = xcorr_parab(prev_xcorr, cur_xcorr)
        maxvals.append(peak)
        chunk_nums[cur_xcorr_num] = ii
        cur_xcorr_num += 1
        prev_xcorr = cur_xcorr

tg2=time.time()
print("total time", tg2-tg1)
print("avg time", (tg2-tg1)/num_micro_chunks)

print(maxvals)
print(chunk_nums[:cur_xcorr_num].tolist())
    
# fname=os.path.join(out_path, f"debug_genph_{tstart}_{num_micro_chunks_per_block}_{int(time.time())}.npz")
# np.savez(fname,chunks=block_xcorr)
# print("wrote", fname)
# fit a parabola to consecutive xcorrs' xcorr
