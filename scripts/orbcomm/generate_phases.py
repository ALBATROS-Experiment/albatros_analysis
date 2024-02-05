import sys
import os
import time
# sys.path.insert(0, '/home/mohan/Projects/albatros_analysis/')
script_dir = os.path.dirname(
    os.path.abspath(__file__)
)  # Path to the directory where the script is located
project_home_dir = os.path.dirname(script_dir)  # Path to the project_home directory
sys.path.insert(0, project_home_dir)
from correlations import baseband_data_classes as bdc

# from albatros_analysis.correlations import correlations as cr
from utils import baseband_utils as butils
from utils import orbcomm_utils as outils
import numpy as np
import datetime
from matplotlib import pyplot as plt

def delay_corrector(idx1,idx2, delay):
    # convetion is xcorr = <a(t)b(t-delay)>
    # where a = antenna1 and b = antenna2
    delay = delay - 100000
    print("original",idx1,idx2)
    if delay > 0:
        idx1+=delay
    else:
        idx2+=np.abs(delay)
    print("corrected", idx1,idx2)
    return idx1,idx2

tstart = 1627453681
dt = 4096 / 250e6
t1 = tstart + 150 * 6.44
t2 = tstart + 230 * 6.44  # 100 sec of bright data
print(f"Start time: {t1:.3f}, End time: {t2:.3f}, Duration: {t2-t1:.3f}")
files_a1, idx1 = butils.get_init_info(
    t1, t2, "/project/s/sievers/albatros/uapishka/202107/baseband/snap3/"
)
files_a2, idx2 = butils.get_init_info(
    t1, t2, "/project/s/sievers/albatros/uapishka/202107/baseband/snap1/"
)
idx1,idx2 = delay_corrector(idx1,idx2,40429)
hdr1 = bdc.get_header(files_a1[0])
chanstart = np.where(hdr1["channels"] == 1839)[0][0]
nchans = 10
# size=int(50/dt)
size = 25000
# nchunks = int((t2 - t1) / (size * dt))
nchunks=50
print(
    f"Start channel: {1839:d} at index {chanstart:d}. Num channels: {nchans:d}. Block length: {size:d}. nchunks: {nchunks:d}"
)

a1_coords = [51.4646065, -68.2352594, 341.052] #north antenna
a2_coords = [51.46418956, -68.23487849, 338.32526665] #south antenna
satnorads = [40087, 41187]
# satnorads = [28654]
niter = int(t2 - t1) + 2  # run it for an extra second to avoid edge effects
delays = np.zeros((size, len(satnorads)))
og_delays = np.zeros((niter, len(satnorads)))
times = np.arange(0,niter) # time in seconds
for i, num in enumerate(satnorads):
    print(i, num)
    og_delays[:,i] = outils.get_sat_delay(
        a1_coords, a2_coords, "orbcomm_28July21.txt", t1, niter, num
    )
print(f"loaded TLEs")
sat2chan = {41187: [1840, 1844], 40087: [1846, 1847]}
# sat2chan = {28654:[1836]}

ant1 = bdc.BasebandFileIterator(
    files_a1,
    0,
    idx1,
    size,
    nchunks=nchunks,
    chanstart=chanstart,
    chanend=chanstart + nchans,
    type="float",
)
ant2 = bdc.BasebandFileIterator(
    files_a2,
    0,
    idx2,
    size,
    nchunks=nchunks,
    chanstart=chanstart,
    chanend=chanstart + nchans,
    type="float",
)
one_sig = []
three_sig = []
chan2freq = lambda chan: 250e6 * (1 - chan / 4096)
for ii, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
    print(f"Processing block {ii}...")
    assert(chunk2['specnums'][-1]-chunk2['specnums'][0]+1==size)
    p0_a1 = np.zeros((size, nchans), dtype="complex64")  # could parallelize
    p0_a2 = np.zeros((size, nchans), dtype="complex64")
    perc_missing_a1 = (1 - len(chunk1["specnums"]) / size) * 100
    perc_missing_a2 = (1 - len(chunk2["specnums"]) / size) * 100
    print(f"perc missing ant1 and ant2: {perc_missing_a1:.3f}%, {perc_missing_a2:.3f}%")
    if perc_missing_a1 > 10 or perc_missing_a2 > 10:
        print("TOO MUCH MISSING.")
        exit(1)
    outils.make_continuous(
        p0_a1, chunk1["pol0"], chunk1["specnums"] - chunk1["specnums"][0]
    )
    outils.make_continuous(
        p0_a2, chunk2["pol0"], chunk2["specnums"] - chunk2["specnums"][0]
    )
    all_xcorrs = []
    corr_chans = []
    N = 2 * size
    dN = min(100000, int(0.3 * N))
    # dN = int(0.3*N)
    stamp = slice(N // 2 - dN, N // 2 + dN)
    print("N is ", N, "dN is ", dN)
    for satnum in sat2chan.keys():
        id = satnorads.index(satnum)
        print(id,satnum)
        delay = np.interp(np.arange(ii*size, (ii+1)*size) * dt, times, og_delays[:, id]) #update the delays
        print(f"delay from {ii*size} to {(ii+1)*size}")
        for chan in sat2chan[satnum]:
            print(f"Sat #{satnum} in chan #{chan}")
            phase_delay = 2 * np.pi * delay * chan2freq(chan)
            chan_num = int(chan - hdr1["channels"][chanstart])
            corr_chans.append(chan)
            all_xcorrs.append(
                np.fft.fftshift(
                    outils.get_coarse_xcorr(
                        p0_a1[:, chan_num],
                        p0_a2[:, chan_num] * np.exp(1j * phase_delay),
                    ),
                    axes=1,
                )[:, stamp]
            )
    if len(corr_chans)==1:
        fig,ax = plt.subplots(1,1)
        ax = [ax]
    else:
        fig, ax = plt.subplots(2, len(corr_chans) // 2)
        ax = ax.flatten()
    fig.set_size_inches(10, 8)
    plt.suptitle(f"chunk #{ii}")
    for j, xcorr in enumerate(all_xcorrs):
        m = np.argmax(np.abs(all_xcorrs[j][0, :]))
        print("max at", m)
        ax[j].set_title(f"chan {corr_chans[j]}, max @ {m}")
        ax[j].plot(np.abs(all_xcorrs[j][0, m - 1000 : m + 1000]))
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # plt.savefig(f"/scratch/s/sievers/mohanagr/coarse_xcorr_{timestamp}.jpg")
    # plt.savefig(f"./coarse_xcorr_{timestamp}.jpg")
    if ii == 0 :
        COARSE_CENTER = m # fix the center so we can track the peaks
    print("starting upsampling...")
    osamp=10
    dN=100
    xcorr_arr = np.zeros((len(all_xcorrs),2*dN*4096*osamp),dtype='complex128')
    sample_no=np.arange(0,2*dN*4096*osamp)
    coarse_sample_no=np.arange(0,2*dN)*4096*osamp
    print("osamp is", osamp)
    t1=time.time()
    tot_noise = 0
    for i, chan in enumerate(corr_chans):
        noise_real=np.std(np.real(all_xcorrs[i][0,dN+500:]))
        print("chan is", chan, "using max at ", COARSE_CENTER, "noise ", noise_real)
        tot_noise += noise_real**2
        xcorr_arr[i,:] = outils.get_interp_xcorr(all_xcorrs[i][0,COARSE_CENTER-dN:COARSE_CENTER+dN],chan,sample_no,coarse_sample_no)
    t2=time.time()
    print("all upsampling done", t2-t1,(t2-t1)/len(corr_chans))
    final_xcorr = np.sum(xcorr_arr,axis=0)
    tot_noise = np.sqrt(tot_noise)
    mm=np.argmax(final_xcorr.real)
    print("final xcorr max at", mm)
    print("total noise", tot_noise)
    new_snr = (final_xcorr-final_xcorr.real[mm])/tot_noise
    one=np.where(new_snr > -1)[0]
    one_sig.append(one)
    three=np.where(new_snr > -3)[0]
    three_sig.append(three)
    # print("num peaks 1 and 3 sig", len(one), len(three))

# print("1 sigma", one_sig)
sets = [set(lst) for lst in one_sig]

common = set.intersection(*sets)
print(common)
print(len(common))
