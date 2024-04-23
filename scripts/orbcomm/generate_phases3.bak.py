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

T_SPECTRA = 4096 / 250e6
T_ACCLEN = 393216 * T_SPECTRA
DEBUG = False
out_path = "/scratch/s/sievers/mohanagr"

# ----------------BASEBAND SETTINGS--------------------------------------------#
tstart = 1627453681
tt1 = tstart + 150 * T_ACCLEN  # OG is 150.
tt2 = tstart + 230 * T_ACCLEN
print(f"Start time: {tt1:.3f}, End time: {tt2:.3f}, Duration: {tt2-tt1:.3f}")
files_a1, idx1 = butils.get_init_info(
    tt1, tt2, "/project/s/sievers/albatros/uapishka/202107/baseband/snap3/"
)
files_a2, idx2 = butils.get_init_info(
    tt1, tt2, "/project/s/sievers/albatros/uapishka/202107/baseband/snap1/"
)
hdr1 = bdc.get_header(files_a1[0])
chan_1839_idx = np.where(hdr1["channels"] == 1839)[0][0]
# -----------------------------------------------------------------------------#
# ----------------SATELLITE SETTINGS-------------------------------------------#
a1_coords = [51.4646065, -68.2352594, 341.052]  # north antenna
a2_coords = [51.46418956, -68.23487849, 338.32526665]  # south antenna
satnorads = [40087, 41187]
sat2chan = {41187: [1840, 1844], 40087: [1846, 1847]}
chan2sat = {1840: 41187, 1844: 41187, 1846: 40087, 1847: 40087}
# corr_chans = np.asarray([1840, 1844, 1846, 1847], dtype=int)
corr_chans = np.asarray([1846, 1847], dtype=int)
channel_idx = corr_chans - 1839 + chan_1839_idx
niter = int(tt2 - tt1) + 2  # run it for an extra second to avoid edge effects
delays = {}
og_delays = {}
times = np.arange(0, niter)  # time in seconds
# -----------------------------------------------------------------------------#
# ----------------AVERAGING SETTINGS-------------------------------------------#
size_micro_chunk = 10000
num_micro_chunks_per_block = 300
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
dN_interp1 = 10  # stamp size for calculating upsampled xcorr for each chan.
dN_interp2 = 10  # stamp size for storing final summed-all-chans upsampled xcorr.
sample_no = np.arange(0, (2 * dN_interp1 - 1) * 4096 * osamp, dtype="float64")
coarse_sample_no = np.arange(0, 2 * dN_interp1, dtype="float64") * 4096 * osamp
UPSAMP_CENTER = len(sample_no)//2
UPSAMP_dN = dN_interp2 * 4096 * osamp
interp_xcorr = np.empty((len(corr_chans), len(sample_no)), dtype="complex128")
block_xcorr = np.zeros((num_micro_chunks, len(sample_no)), dtype="complex128")
block_noise = np.zeros(num_micro_chunks, dtype="float64")
idx1, idx2 = outils.delay_corrector(idx1, idx2, 40429, 100000)  # 150 => 40429 180 => 43980

print(
    f"Start channel: {1839:d} at index {chan_1839_idx:d}.\n\
    Num channels: {len(channel_idx):d}.\n\
    Block length: {size_micro_chunk:d}.\n\
    len coarse xcorr: {2*N}'\n\
    len stamp: {2*dN}\n\
    nchunks: {num_micro_chunks:d}"
)
print("channel indices", channel_idx)
for i, num in enumerate(satnorads):
    print(i, num)
    og_delays[num] = outils.get_sat_delay(
        a1_coords, a2_coords, "orbcomm_28July21.txt", tt1, niter, num
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
one_sig = []
three_sig = []
chan2freq = lambda chan: 250e6 * (1 - chan / 4096)


print("osamp is", osamp)

# ---------------------#
a1_start = ant1.spec_num_start
a2_start = ant2.spec_num_start
print("a1 start", a1_start, "a2_start", a2_start)
for ii, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
    # tg1=time.time()
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

    if False:
        fig2, ax2 = plt.subplots(np.ceil(xcorr_stamp.shape[0] / 3).astype(int), 3)
        fig2.set_size_inches(12, np.ceil(xcorr_stamp.shape[0] / 3) * 3)
        ax2 = ax2.flatten()
        fig2.suptitle(f"30 values around the center of array of size {2*dN}")
        for i in range(xcorr_stamp.shape[0]):
            ax2[i].set_title(
                f"chan {corr_chans[i]} max: {np.argmax(np.abs(np.abs(xcorr_stamp[i,:])))}"
            )
            ax2[i].plot(np.abs(xcorr_stamp[i, :]))
            ax2[i].set_xlim(dN - 15, dN + 15)
        plt.tight_layout()
        fig2.savefig(
            os.path.join(out_path, f"debug_genph_{tstart}_{ii}_{int(time.time())}.jpg")
        )

    # print("starting upsampling...")

    COARSE_CENTER = xcorr_stamp.shape[1] // 2
    tot_noise = 0
    # t1=time.time()
    for i, chan in enumerate(corr_chans):
        sat = chan2sat[chan]
        # shift = -int(delays[sat][ii] * 250e6 * osamp) #get delay for chunk num ii
        real_freq = 1-chan/4096
        alia_freq = chan/4096
        shift = delays[sat][ii] * 250e6 * osamp * real_freq/alia_freq
        shifted_samples = sample_no - shift
        # print("shift is ", shift, "for sat", sat)
        noise_real = np.std(np.real(xcorr_stamp[i, COARSE_CENTER + 500 :]))
        # print("chan is", chan, "using max at ", COARSE_CENTER, "noise ", noise_real)
        # temp = xcorr_stamp[
        #     i, COARSE_CENTER - dN_interp1 : COARSE_CENTER + dN_interp1
        # ].copy()
        tot_noise += noise_real**2
        outils.get_interp_xcorr_fast(
            xcorr_stamp[i,COARSE_CENTER - dN_interp1 : COARSE_CENTER + dN_interp1],
            chan,
            shifted_samples,
            coarse_sample_no,
            out=interp_xcorr[i, :]
        )
    # t2=time.time()
    # print(f"all upsampling done; {(t2 - t1) / len(corr_chans):5.3f}s per channel, {t2-t1} total")
    # block_xcorr[ii, :] = interp_xcorr[2,UPSAMP_CENTER-UPSAMP_dN:UPSAMP_CENTER+UPSAMP_dN]
    # t1=time.time()
    # print("size of upsampled arr//2 vs UPSAMP_CENTER is", interp_xcorr.shape[1]//2, UPSAMP_CENTER)
    # block_xcorr[ii, :] = np.sum(interp_xcorr[:,
    #     UPSAMP_CENTER - UPSAMP_dN : UPSAMP_CENTER + UPSAMP_dN
    # ], axis=0)
    block_xcorr[ii, :] = np.sum(interp_xcorr, axis=0)
    # tg2=time.time()
    # print(block_xcorr[ii,dN_interp1-10:dN_interp1+10])
    # print("summing and assigning", t2-t1)
    block_noise[ii] = np.sqrt(tot_noise)
    # print(f"mu-chunk {ii} noise = {block_noise[ii]}")
    # tg2=time.time()
    # print("loop took", tg2-tg1)
# fname=os.path.join(out_path, f"debug_genph_{tstart}_{num_micro_chunks_per_block}_{int(time.time())}.npz")
# np.savez(fname,chunks=block_xcorr)
# print("wrote", fname)
# fit a parabola to consecutive xcorrs' xcorr
bigdat = np.hstack([block_xcorr, np.zeros(block_xcorr.shape, dtype=block_xcorr.dtype)])
print("hstack done")
bigft = mk.many_fft_c2c_1d(bigdat,axis=1)
center2=block_xcorr.shape[1]
n_avg = outils.get_weights(center2)
maxvals=[]
for i in range(0,block_xcorr.shape[0]-1):
    t1=time.time()
    xcorr = outils.complex_mult_conj(bigft[i:i+1,:], bigft[i+1:i+2,:])
    xcorr1 = mk.many_fft_c2c_1d(xcorr,axis=1,backward=True)
    xc = outils.get_normalized_stamp(xcorr1, n_avg, 50, 1)
    t2=time.time()
    print(i)
    m=np.argmax(xc[0,:].real)
    params=np.polyfit(np.arange(10),xc[0,50-5:50+5].real,2)
    maxvals.append(-params[1]/params[0]/2)
    # maxvals.append(m)
    # print(f"max at {m}, taking {t2-t1}")
print(maxvals)
exit(1)
avg_xcorr = np.zeros((num_blocks, 2 * UPSAMP_dN), dtype="complex128")
avg_xcorr_noise = np.zeros(num_blocks, dtype="float64")
ll = 2 * UPSAMP_dN
for bn in range(num_blocks):
    avg_xcorr[bn, :] = np.mean(
        block_xcorr[
            bn * num_micro_chunks_per_block : (bn + 1) * num_micro_chunks_per_block, :
        ],
        axis=0,
    )
    avg_xcorr_noise[bn] = (
        np.sqrt(
            np.sum(
                block_noise[
                    bn * num_micro_chunks_per_block : 
                    (bn + 1) * num_micro_chunks_per_block
                ] ** 2
            )
        )/ num_micro_chunks_per_block
    )
if True:
    fig2, ax2 = plt.subplots(np.ceil(num_blocks / 2).astype(int), 2)
    fig2.set_size_inches(10, np.ceil(num_blocks / 2) * 4)
    ax2 = ax2.flatten()
    fig2.suptitle(
        f"per mu-chunk size={size_micro_chunk} spectra, start at {tt1}, block averaged over {num_micro_chunks_per_block} mu-chunks."
    )
    for axnum in range(num_blocks):
        mm = np.argmax(np.real(avg_xcorr[axnum]))
        ax2[axnum].set_title(f" max at {mm}")
        ax2[axnum].plot(np.abs(avg_xcorr[axnum]))
        new_snr = (avg_xcorr[axnum]-np.real(avg_xcorr[axnum])[mm])/(np.sqrt(2)*avg_xcorr_noise[axnum])
        one=np.where(new_snr > -3)[0]
        print(f"For block num {axnum}, noise is {avg_xcorr_noise[axnum]}, num peaks in 3 sig are", len(one))
        print(one)
        one_sig.append(one)
    plt.tight_layout()
    fig2.savefig(
        os.path.join(
            out_path,
            f"debug_upsmp_{tstart}_{size_micro_chunk}_{num_micro_chunks}_{num_blocks}_{int(time.time())}.jpg",
        )
    )
    sets = [set(lst) for lst in one_sig]

    for s in range(num_blocks-1):
        print(f"common {s} and {s+1}", set.intersection(sets[s], sets[s+1]))
    print("common all", set.intersection(*sets))