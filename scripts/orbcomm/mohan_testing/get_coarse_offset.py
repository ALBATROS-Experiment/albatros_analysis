import sys
import time
sys.path.insert(0, "/home/s/sievers/mohanagr/")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
from matplotlib import pyplot as plt
from os import path

T_SPECTRA = 4096 / 250e6
T_ACCLEN = 393216 * T_SPECTRA
DEBUG=True

deployment_yyyymm="202210"
ant1_snap = "snap3"
# ant2_snap = "snap1"
ant2_snap = "snap4"
base_path = path.join("/project/s/sievers/albatros/uapishka",deployment_yyyymm)
out_path = "/scratch/s/sievers/mohanagr/"
tstart = 1667015968
t1 = tstart + 372 * T_ACCLEN
t2 = tstart + 504 * T_ACCLEN
files_a1, idx1 = butils.get_init_info(
                t1, t2, path.join(base_path,"baseband", ant1_snap)
            )
files_a2, idx2 = butils.get_init_info(
                t1, t2,path.join(base_path,"baseband", ant2_snap)
            )
nchunks = 10
size = 1000000
dN = 100000
chanstart=1834
chanend=1848

nchans=chanend-chanstart
h=bdc.get_header(files_a1[0])
chanstart_idx = np.where(h['channels']==chanstart)[0][0]
chanend_idx = np.where(h['channels']==chanend)[0][0]
print(chanstart_idx, chanend_idx)
assert((chanend_idx-chanstart_idx) == nchans)
ant1 = bdc.BasebandFileIterator(
    files_a1,
    0,
    idx1,
    size,
    nchunks,
    chanstart=chanstart_idx,
    chanend=chanend_idx,
    type="float",
)
ant2 = bdc.BasebandFileIterator(
    files_a2,
    0,
    idx2,
    size,
    nchunks,
    chanstart=chanstart_idx,
    chanend=chanend_idx,
    type="float",
)
p0_a1 = np.zeros((size, nchans), dtype="complex128") #remember that BDC returns complex64. wanna do phase-centering in 128.
p0_a2 = np.zeros((size, nchans), dtype="complex128")
a1_start = ant1.spec_num_start
a2_start = ant2.spec_num_start
for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
    perc_missing_a1 = (1 - len(chunk1["specnums"]) / size) * 100
    perc_missing_a2 = (1 - len(chunk2["specnums"]) / size) * 100
    print("missing a1", perc_missing_a1, "missing a2", perc_missing_a2)
    if perc_missing_a1 > 5 or perc_missing_a2 > 5:
        a1_start = ant1.spec_num_start
        a2_start = ant2.spec_num_start
        continue
    print(chunk1['pol0'].shape)
    outils.make_continuous(
        p0_a1, chunk1["pol0"], chunk1["specnums"] - a1_start
    )
    outils.make_continuous(
        p0_a2, chunk2["pol0"], chunk2["specnums"] - a2_start
    )
    # print("p0 and p1 -------")
    # print("p0")
    # print(p0_a1)
    # print(p0_a2)
    break
cx=outils.get_coarse_xcorr_fast2(p0_a1, p0_a2, dN)
# cx=np.fft.fftshift(outils.get_coarse_xcorr(p0_a1, p0_a2),axes=1)
if DEBUG:
    fig2, ax2 = plt.subplots(np.ceil(cx.shape[0]/3).astype(int), 3)
    fig2.set_size_inches(12, np.ceil(cx.shape[0]/3)*3)
    ax2=ax2.flatten()
    dx=50
    for i in range(cx.shape[0]):
        mm=np.argmax(np.abs(cx[i,:]))
        ax2[i].set_title(f"chan {chanstart+i} max: {mm}")
        # print(cx[i,mm-dx:mm+dx])
        ax2[i].plot(np.abs(cx[i,mm-dx:mm+dx]))
    plt.tight_layout()
    figpath=path.join(out_path,f"debug_cxcorr_{tstart}_{int(time.time())}.jpg")
    fig2.savefig(figpath)
    print("wrote", figpath)