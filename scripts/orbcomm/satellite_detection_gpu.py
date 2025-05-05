import os

print("CUPY_CACHE_DIR =", os.environ.get("CUPY_CACHE_DIR"))

import sys
import time
#status as of Feb 14, 2024: after all speed updates, once again compared to jupyter output.
#                           sat delay values match, coarse xcorr values match, SNR matches
from os import path
sys.path.insert(0, "/home/s/sievers/thomasb/")

from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import json
import cProfile, pstats
import cupy as cp

T_SPECTRA = 4096 / 250e6
# T_ACCLEN = 393216 * T_SPECTRA
T_ACCLEN = 5
DEBUG=True

# a1_path = "/scratch/s/sievers/mohanagr/mars1_2024/baseband/"
# a2_path = "/scratch/s/sievers/mohanagr/mars2_2024/baseband/"
a1_path = "/project/s/sievers/albatros/mars/202307/baseband/stn_1_central"
a2_path = "/project/s/sievers/albatros/mars/202307/baseband/stn_2_east"

out_path = "/project/s/sievers/thomasb/"
# all the sats we track
# satlist = [28654,25338,33591,57166,59051,44387]
satlist = [57166,59051]
satmap = {}
assert min(satlist) > len(
    satlist
)  # to make sure there are no collisions, we'll never have an i that's also a satnum
for i, satnum in enumerate(satlist):
    satmap[i] = satnum
    satmap[satnum] = i
# print(satmap)

# for each file get the risen sats and divide them up into unique transits
a1_coords = [79+25.031/60, -90-46.041/60, 189]  # MARS 1
a2_coords = [79+25.033/60, -90-45.531/60, 176]  # MARS 2

sat_data = {}

# tstart = butils.get_tstamp_from_filename(file)
nrows=int(24*3600*1/5)
# tstart = 1721800002+1000*5
tstart = 1699711338
sat_data[tstart] = []
tle_path = outils.get_tle_file(tstart, "/project/s/sievers/mohanagr/OCOMM_TLES")
print("USING TLE PATH", tle_path)
arr = np.zeros((nrows, len(satlist)), dtype="int64")
rsats = outils.get_risen_sats(tle_path, a1_coords, tstart, dt=5, niter=nrows,good=satlist,altitude_cutoff=15)
num_sats_risen = [len(x) for x in rsats]
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10,4)
fig.suptitle(f"Risen sats for file {tstart}")
ax[0].plot(num_sats_risen)
ax[0].set_xlabel("time (in units of 5 sec)")
for i, row in enumerate(rsats):
    for satnum, satele,sataz in row:
        arr[i,satmap[satnum]] = 1
ax[1].set_ylabel("time in units of 5 sec")
ax[1].set_xlabel("Sat ID")
ax[1].imshow(arr,aspect='auto',interpolation="none")
plt.tight_layout()
fig.savefig(path.join(out_path,f"risen_sats_{tstart}_{str(time.time())}.jpg"))
pulses = outils.get_simul_pulses(arr)
print("Sat transits detected are:", pulses)

fig, ax = plt.subplots(np.ceil(len(pulses)/2).astype(int), 2)
fig.set_size_inches(10, np.ceil(len(pulses)/2)*4)
fig.suptitle(str(tstart))
ax=ax.flatten()
for pnum, [(pstart, pend), sats] in enumerate(pulses):
    print("running", pnum, pstart, pend, sats)
    pulse_data = {}
    pulse_data["start"] = pstart
    pulse_data["end"] = pend
    pulse_data["sats"] = {}
    numsats_in_pulse = len(sats)
    t1 = tstart + pstart * T_ACCLEN
    t2 = tstart + pend * T_ACCLEN  #probably do something about this. 
    # dont need files for the entire pulse. some 50 sec chunk within the pulse will work
    # t2 = t1 + 50
    print("t1 t2 for the pulse are:", t1, t2)
    try:
        files_a1, idx1 = butils.get_init_info(t1, t2, a1_path)
        files_a2, idx2 = butils.get_init_info(t1, t2, a2_path)
    except Exception as e:
        print(e)
        print(f"skipping pulse {pstart}:{pend} in {tstart} as some file discontinuity was encountered.")
        continue
    # print(files_a1,files_a2)
    channels = np.asarray(bdc.get_header(files_a1[0])["channels"],dtype='int64')
    chanstart = np.where(channels == 1834)[0][0]
    chanend = np.where(channels == 1852)[0][0]
    nchans = chanend - chanstart
    size = 1000000
    # #dont impose any chunk num, continue iterating as long as a chunk with small enough missing fraction is found.
    # #have passed enough files to begin with. should not run out of files.
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
    p0_a1 = cp.zeros((size, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
    p0_a2 = cp.zeros((size, nchans), dtype="complex64")
    p0_a2_delayed = cp.zeros((size, nchans), dtype="complex64")
    niter = int(t2 - t1) + 1  # run it for an extra second to avoid edge effects
    print("niter for delay is", niter, "t1 is", t1)
    delays = np.zeros((size, len(sats)))
    # get geo delay for each satellite from Skyfield
    for i, satID in enumerate(sats):
        d = outils.get_sat_delay(
            a1_coords,
            a2_coords,
            tle_path,
            t1,
            niter,
            satmap[satID],
        )
        delays[:, i] = np.interp(
            np.arange(0, size) * T_SPECTRA, np.arange(0, niter), d
        )
        # print(f"delay for {satmap[satID]}", delays[0:10,i], delays[-10:,i])
    # get baseband chunk for the duration of required transit. Take the first chunk `size` long that satisfies missing packet requirement
    # print(delays[0:10,1], delays[-10:,1])
    # print(delays[0:10,0], delays[-10:,0])
    delays = cp.asarray(delays)
    a1_start = ant1.spec_num_start
    a2_start = ant2.spec_num_start
    for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
        perc_missing_a1 = (1 - len(chunk1["specnums"]) / size) * 100
        perc_missing_a2 = (1 - len(chunk2["specnums"]) / size) * 100
        print("missing a1", perc_missing_a1, "missing a2", perc_missing_a2)
        if perc_missing_a1 > 10 or perc_missing_a2 > 10:
            a1_start = ant1.spec_num_start
            a2_start = ant2.spec_num_start
            continue
        # print(chunk1["pol0"])
        # print(chunk2["pol0"])
        # print("nchans is", nchans)
        # print("first row chunk1", chunk1['pol0'][0,:])
        bdc.make_continuous_gpu(chunk1['pol0'],chunk1['specnums']-a1_start,np.arange(nchans),size,nchans=nchans, out=p0_a1)
        bdc.make_continuous_gpu(chunk2['pol0'],chunk2['specnums']-a2_start,np.arange(nchans),size,nchans=nchans, out=p0_a2)
        # print("first row p0_a1", p0_a1[0,:])
        break
    #If we detect a sat, note down what file we started from and what the specnum at the start was. 
    info_tstamps_a1 = str(butils.get_tstamp_from_filename(files_a1[0]))+":"+str(ant1.spec_num_start-size)
    info_tstamps_a2 = str(butils.get_tstamp_from_filename(files_a2[0]))+":"+str(ant2.spec_num_start-size)
    specnum_offset = ant1.spec_num_start - ant2.spec_num_start #this is the initial delay between specnums when the antennas booted up
    print(info_tstamps_a1)
    print(info_tstamps_a2)
    print("initial specnum offset 1 - 2", specnum_offset)
    cx = []  # store coarse xcorr for each satellite
    N = 2 * size
    dN = min(100000, int(0.3 * N))
    print("2*N and 2*dN", N, dN)
    temp_satmap = []  # will need to map the row number to satID later
    cx.append(outils_g.coarse_xcorr(p0_a1, p0_a2, dN))  # no correction
    if DEBUG:
        fig2, ax2 = plt.subplots(np.ceil(cx[0].shape[0]/3).astype(int), 3)
        fig2.set_size_inches(12, np.ceil(cx[0].shape[0]/3)*3)
        ax2=ax2.flatten()
        fig2.suptitle(f"for pulse {pstart}:{pend}")
        for i in range(cx[0].shape[0]):
            mm=cp.argmax(cp.abs(cx[0][i,:]))
            ax2[i].set_title(f"chan {1834+i} max: {mm}")
            ax2[i].plot(cp.asnumpy(cp.abs(cx[0][i,:])))
            # ax2[i].set_xlim(mm-1000,mm+1000)
        plt.tight_layout()
        fig2.savefig(path.join(out_path,f"debug_cxcorr_{tstart}_{pstart}_{pend}.jpg"))
        print(path.join(out_path,f"debug_cxcorr_{tstart}_{pstart}_{pend}.jpg"))
        # sys.exit(0)
    temp_satmap.append("default")  # zeroth row is always "no phase"
    # get beamformed visibilities for each satellite
    freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)

    for i, satID in enumerate(sats):
        print("processing satellite:", satmap[satID])
        temp_satmap.append(satmap[satID])
        # phase_delay = 2 * np.pi * delays[:, i : i + 1] @ freq
        # print("phase delay shape", phase_delay.shape)
        outils_g.apply_delay(p0_a2, delays[:,i], freqs, out=p0_a2_delayed)
        cx.append(
                outils_g.coarse_xcorr(
                    p0_a1, p0_a2_delayed, dN
                )
        )
    snr_arr = np.zeros(
        (len(sats) + 1, nchans), dtype="float64"
    )  # rows = sats, cols = channels
    detected_sats = np.zeros(nchans, dtype="int")
    detected_peaks = np.zeros(nchans, dtype="int")
    # save the SNR for each channel for each satellite
    # cx[0] is the default "dont do anything" xcorr
    ax[pnum].set_title(f"Pulse {pstart} to {pend}.")
    for i in range(len(sats) + 1):
        snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx[i]),axis=1))
        ax[pnum].plot(snr_arr[i, :], label=f"{temp_satmap[i]}")
    ax[pnum].set_xlabel("channels")
    ax[pnum].set_ylabel("SNR")
    ax[pnum].legend()
    # for each channel, update the detected satellite for that channel
    print(snr_arr)
    for chan in range(nchans):
        sortidx = np.argsort(snr_arr[:, chan])
        if (
            sortidx[-1] == 0
        ):  # no sat was detected, idx 0 is the default "no phase" value
            continue
        if (snr_arr[sortidx[-1], chan] - snr_arr[sortidx[-2], chan]) / np.sqrt(
            2
        ) > 5:  # if SNR 1 = a1/sigma, SNR 2 = a2/sigma.
            # I want SNR on a1-a2 i.e. is the difference significant.
            # print(
            #     "top two snr for chan",
            #     chan,
            #     snr_arr[sortidx[-1], chan],
            #     snr_arr[sortidx[-2], chan],
            # )
            cxnum = sortidx[-1] # which cxcorr has the detection
            if DEBUG: # these plots are for the channels where something was detected
                fig2, ax2 = plt.subplots(np.ceil(cx[cxnum].shape[0]/3).astype(int), 3)
                fig2.set_size_inches(12, np.ceil(cx[cxnum].shape[0]/3)*3)
                ax2=ax2.flatten()
                fig2.suptitle(f"for pulse {pstart}:{pend}")
                for i in range(cx[cxnum].shape[0]):
                    mm=cp.argmax(cp.abs(cx[cxnum][i,:]))
                    ax2[i].set_title(f"chan {1834+i} max: {mm}")
                    ax2[i].plot(cp.asnumpy(cp.abs(cx[cxnum][i,:])))
                    # ax2[i].set_xlim(mm-1000,mm+1000)
                plt.tight_layout()
                fig2.savefig(path.join(out_path,f"debug_cxdetect_{tstart}_{pstart}_{pend}.jpg"))
                print(path.join(out_path,f"debug_cxdetect_{tstart}_{pstart}_{pend}.jpg"))
            detected_sats[chan] = temp_satmap[sortidx[-1]]
            detected_peaks[chan] = cp.argmax(cp.abs(cx[sortidx[-1]][chan,:]))
            # detected_sats[chan] = satmap[sats[sortidx[-1]]]
    sat_peaks=[]
    if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
        peak_guesses = detected_peaks[detected_peaks>0]
        print("detected peak locations are", peak_guesses)
        best_guess_offset = np.max(detected_peaks)
        if (best_guess_offset-dN) > 0:
                #tau > 0; idxstart0 += detected_peaks[chan]-dN
                specnum_offset += (best_guess_offset-dN)
        else:
            #tau < 0; idxstart1 += abs(best_guess_offset-dN)
            specnum_offset -= np.abs(best_guess_offset - dN)
        print("specnum offset updated to:", specnum_offset )
    for i, satID in enumerate(sats):
        where_sat = np.where(detected_sats == satmap[satID])[0]
        for ids in where_sat:
            sat_peaks.append([int(ids)+1834, int(detected_peaks[ids])]) #append the channel and the peak location of that channel
        pulse_data["sats"][satmap[satID]] = sat_peaks # make sure it's serializable with json. numpy array wont work
    sat_data[tstart].append(pulse_data)
fig.savefig(
        path.join(out_path,f"debug_snr_{tstart}_{int(time.time())}.jpg")
    )
print(path.join(out_path,f"debug_snr_{tstart}_{int(time.time())}.jpg"))
json_output = path.join(out_path,f"debug_snr_{tstart}_{int(time.time())}.json")
with open(json_output, "w") as file:
    json.dump(sat_data, file, indent=4)
print(sat_data)
