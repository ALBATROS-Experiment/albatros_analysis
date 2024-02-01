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
size = 20000
# idx1+=size*30 <- had gotten an error here but that was because of my useless assertion. last specnum need not always be present. might be in a missing region.
# idx2+=size*30
# nchunks = int((t2 - t1) / (size * dt))
block_len = 20 #226 for 9, 167 for 9, 139 for 10, 4 for 15
nchunks=block_len * 2
print(
    f"Start channel: {1839:d} at index {chanstart:d}. Num channels: {nchans:d}. Block length: {size:d}. nchunks: {nchunks:d}"
)

a1_coords = [51.4646065, -68.2352594, 341.052] #north antenna
a2_coords = [51.46418956, -68.23487849, 338.32526665] #south antenna
satnorads = [40087, 41187]
# satnorads = [28654]
niter = int(t2 - t1) + 2  # run it for an extra second to avoid edge effects
delays = {}
og_delays = {}
times = np.arange(0,niter) # time in seconds
for i, num in enumerate(satnorads):
    print(i, num)
    og_delays[num] = outils.get_sat_delay(
        a1_coords, a2_coords, "orbcomm_28July21.txt", t1, niter, num
    )
    delays[num] = np.interp(np.arange(nchunks) *size * dt, times, og_delays[num])
print(f"loaded TLEs")
print(delays)
sat2chan = {41187: [1840, 1844], 40087: [1846, 1847]}
chan2sat = {1840:41187, 1844:41187, 1846:40087, 1847:40087}
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
osamp=40
dN_interp=100
sample_no=np.arange(0,2*dN_interp*4096*osamp)
coarse_sample_no=np.arange(0,2*dN_interp)*4096*osamp
print("osamp is", osamp)
block_xcorr = np.zeros((block_len, dN_interp*4096*osamp),dtype='complex128')
tot_noise = 0
jj=-1
for ii, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
    jj+=1
    print(f"Processing block {ii} and filling block {jj}...")
    # print(chunk2['specnums'], len(chunk2['specnums']))
    # assert(chunk2['specnums'][-1]-chunk2['specnums'][0]+1==size)
    p0_a1 = np.zeros((size, nchans), dtype="complex64")  # could parallelize
    p0_a2 = np.zeros((size, nchans), dtype="complex64")
    perc_missing_a1 = (1 - len(chunk1["specnums"]) / size) * 100
    perc_missing_a2 = (1 - len(chunk2["specnums"]) / size) * 100
    print(f"perc missing ant1 and ant2: {perc_missing_a1:.3f}%, {perc_missing_a2:.3f}%")
    if perc_missing_a1 > 10 or perc_missing_a2 > 10:
        print("TOO MUCH MISSING, CONTINUING")
        continue
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
        for chan in sat2chan[satnum]:
            print(f"Sat #{satnum} in chan #{chan}")
            chan_num = int(chan - hdr1["channels"][chanstart])
            corr_chans.append(chan)
            all_xcorrs.append(
                np.fft.fftshift(
                    outils.get_coarse_xcorr(
                        p0_a1[:, chan_num],
                        p0_a2[:, chan_num],
                    ),
                    axes=1,
                )[:, stamp]
            )
    # if len(corr_chans)==1:
    #     fig,ax = plt.subplots(1,1)
    #     ax = [ax]
    # else:
    #     fig, ax = plt.subplots(2, len(corr_chans) // 2)
    #     ax = ax.flatten()
    # fig.set_size_inches(10, 8)
    # plt.suptitle(f"chunk #{ii}")
    for j, xcorr in enumerate(all_xcorrs):
        m = np.argmax(np.abs(all_xcorrs[j][0, :]))
        print("max at", m)
        # ax[j].set_title(f"chan {corr_chans[j]}, max @ {m}")
        # ax[j].plot(np.abs(all_xcorrs[j][0, m - 100 : m + 100]))
    # now = datetime.datetime.now()
    # timestamp = now.strftime("%Y%m%d_%H%M%S")
    # plt.savefig(f"/scratch/s/sievers/mohanagr/coarse_xcorr_{timestamp}.jpg")
    # plt.savefig(f"./coarse_xcorr_{timestamp}.jpg")
    # if ii == 0 :
    #     COARSE_CENTER = m # fix the center so we can track the peaks
    print("starting upsampling...")
    
    N = all_xcorrs[0].shape[1]
    dN = dN_interp
    COARSE_CENTER = N//2
    xcorr_arr = np.zeros((len(all_xcorrs),2*dN*4096*osamp),dtype='complex128')
    final_xcorr = np.zeros((len(all_xcorrs),dN*4096*osamp),dtype='complex128') #gonna shift and take a smaller bit of total
    t1=time.time()
    for i, chan in enumerate(corr_chans):
        sat = chan2sat[chan]
        shift = -int(delays[sat][ii] * 250e6 * osamp) #get delay for chunk num ii
        # print("dely is", delays[sat][ii])
        # print("shift is ", shift, "for sat", sat)
        noise_real=np.std(np.real(all_xcorrs[i][0,COARSE_CENTER+500:]))
        print("chan is", chan, "using max at ", COARSE_CENTER, "noise ", noise_real)
        tot_noise += noise_real**2
        xcorr_arr[i,:] = outils.get_interp_xcorr(all_xcorrs[i][0,COARSE_CENTER-dN:COARSE_CENTER+dN],chan,sample_no,coarse_sample_no)
        center=dN*4096*osamp+shift
        delta = dN*4096*osamp//2
        final_xcorr[i,:] = xcorr_arr[i,center-delta:center+delta]
    t2=time.time()
    print("all upsampling done", t2-t1,(t2-t1)/len(corr_chans))
    block_xcorr[jj,:] = np.sum(final_xcorr,axis=0)
    if (ii+1)%block_len == 0:
        print("ONE BLOCK COMPLETE")
        tot_noise = np.sqrt(2*tot_noise/block_len**2)
        final_block_xcorr = np.mean(block_xcorr,axis=0)
        mm=np.argmax(final_block_xcorr.real)
        print("block xcorr max at", mm)
        print("total noise", tot_noise)
        new_snr = (final_block_xcorr-final_block_xcorr.real[mm])/tot_noise
        one=np.where(new_snr > -3)[0]
        one_sig.append(one)
        tot_noise=0
        jj=-1
        # block_xcorr[:] = 0

print("1 sigma", one_sig)
sets = [set(lst) for lst in one_sig]

common = set.intersection(*sets)
print(common,"corresponds to delays", np.asarray(list(common))-delta)
print(len(common))
