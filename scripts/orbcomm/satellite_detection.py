import sys
import time
#status as of Feb 14, 2024: after all speed updates, once again compared to jupyter output.
#                           sat delay values match, coarse xcorr values match, SNR matches
sys.path.insert(0, "/home/s/sievers/mohanagr/")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import json
import cProfile, pstats
from os import path

T_SPECTRA = 4096 / 250e6
T_ACCLEN = 393216 * T_SPECTRA
DEBUG=False

# deployment_yyyymm = "202210"
deployment_yyyymm="202107"
ant1_snap = "snap3"
# ant2_snap = "snap4"
ant2_snap = "snap1"
base_path = path.join("/project/s/sievers/albatros/uapishka",deployment_yyyymm)
out_path = "/scratch/s/sievers/mohanagr/"

# get a list of all direct spectra files between two timestamps
print(base_path, path.join(base_path,"data_auto_cross", ant1_snap))
# ts1 = 1667000000 #newFee job
# ts1 = 1627700000

ts1 = 1627454647 # this is the default test input
ts2 = int(ts1 + 560*T_ACCLEN)
num_files_to_process = 1

# ts1=1667077364
# ts1 = 1667145976

# ts2 = ts1 + 50
ts2 = 1627735000
# start_file = butils.get_file_from_timestamp(
#     ts1, path.join(base_path,"data_auto_cross", ant1_snap), "d"
# )[0]
# start_file_tstamp = butils.get_tstamp_from_filename(start_file)
print(f"requested {ts1} to {ts2}")
direct_files = butils.time2fnames(
    ts1,
    ts2,
    path.join(base_path,"data_auto_cross", ant1_snap),
    "d"
)
# print(direct_files)

# all the sats we track
satlist = [
    40086,
    40087,
    40091,
    41179,
    41182,
    41183,
    41184,
    41185,
    41186,
    41187,
    41188,
    41189,
    25338,
    28654,
    33591,
    40069,
]
satmap = {}
assert min(satlist) > len(
    satlist
)  # to make sure there are no collisions, we'll never have an i that's also a satnum
for i, satnum in enumerate(satlist):
    satmap[i] = satnum
    satmap[satnum] = i
# print(satmap)

# for each file get the risen sats and divide them up into unique transits
a1_coords = [51.4646065, -68.2352594, 341.052]  # north antenna -> snap4 on 20221026
a2_coords = [51.46418956, -68.23487849, 338.32526665]  # south antenna -> snap3 on 20221026

sat_data = {}
profiler = cProfile.Profile()
for file_num, file in enumerate(direct_files):
    if file_num == num_files_to_process:
        break
    # tstart = butils.get_tstamp_from_filename(file)
    # nrows=560
    tstart = ts1
    nrows = 50
    sat_data[tstart] = []
    # tle_path = outils.get_tle_file(tstart, "/project/s/sievers/mohanagr/OCOMM_TLES")
    tle_path = "./orbcomm_28July21.txt"
    print("USING TLE PATH", tle_path)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, a1_coords, tstart, niter=nrows)
    num_sats_risen = [len(x) for x in rsats]
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10,4)
    fig.suptitle(f"Risen sats for file {tstart}")
    ax[0].plot(num_sats_risen)
    ax[0].set_xlabel("niter = nrows in file (x 6.44 sec)")
    for i, row in enumerate(rsats):
        for satnum, satele in row:
            arr[i][satmap[satnum]] = 1
    ax[1].set_ylabel("niter (x 6.44 sec)")
    ax[1].set_xlabel("Sat ID")
    ax[1].imshow(arr,aspect='auto',interpolation="none")
    plt.tight_layout()
    fig.savefig(path.join(out_path,f"risen_sats_{tstart}.jpg"))
    pulses = outils.get_simul_pulses(arr)
    print("Sat transits detected are:", pulses)
    fig, ax = plt.subplots(np.ceil(len(pulses)/2).astype(int), 2)
    fig.set_size_inches(10, np.ceil(len(pulses)/2)*4)
    fig.suptitle(str(tstart))
    ax=ax.flatten()
    for pnum, [(pstart, pend), sats] in enumerate(pulses):
        print(pstart, pend, sats)
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
            files_a1, idx1 = butils.get_init_info(
                t1, t2, path.join(base_path,"baseband", ant1_snap)
            )
            files_a2, idx2 = butils.get_init_info(
                t1, t2,path.join(base_path,"baseband", ant2_snap)
            )
        except Exception as e:
            print(e)
            print(f"skipping pulse {pstart}:{pend} in {tstart} as some file discontinuity was encountered.")
            continue
        # print(files_a1,files_a2)
        channels = bdc.get_header(files_a1[0])["channels"]
        chanstart = np.where(channels == 1834)[0][0]
        chanend = np.where(channels == 1852)[0][0]
        nchans = chanend - chanstart
        # #a1 = antenna 1 = SNAP3
        # #a2 = antenna 2 = SNAP1
        size = 3000000
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

        p0_a1 = np.zeros((size, nchans), dtype="complex128") #remember that BDC returns complex64. wanna do phase-centering in 128.
        p0_a2 = np.zeros((size, nchans), dtype="complex128")
        p0_a2_delayed = np.zeros((size, nchans), dtype="complex128")
        # freq = 250e6 * (1 - np.arange(1834, 1854) / 4096).reshape(
        #     -1, nchans
        # )  # get actual freq from aliasedprint("FREQ",freq/1e6," MHz")
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
            outils.make_continuous(
                p0_a1, chunk1["pol0"], chunk1["specnums"] - a1_start
            )
            outils.make_continuous(
                p0_a2, chunk2["pol0"], chunk2["specnums"] - a2_start
            )
            break
        # print(p0_a1, p0_a2)
        cx = []  # store coarse xcorr for each satellite
        N = 2 * size
        dN = min(100000, int(0.3 * N))
        print("2*N and 2*dN", N, dN)
        temp_satmap = []  # will need to map the row number to satID later
        cx0 = outils.get_coarse_xcorr_fast2(p0_a1, p0_a2, dN)
        print("RUNNING SPEED TEST")
        tottime=0
        profiler.enable()
        for i in range(20):
            t1=time.time()
            cx0 = outils.get_coarse_xcorr_fast2(p0_a1, p0_a2, dN)
            t2=time.time()
            tottime+=t2-t1
            print("current impl taking", t2-t1)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        # with open(path.join(out_path, "stats.txt"), 'w') as f:
        #     stats.stream = f
        #     stats.print_stats(15)
        print("average time taken", tottime/20)
        exit(1)
        cx.append(outils.get_coarse_xcorr_fast(p0_a1, p0_a2, dN))  # no correction

        if DEBUG:
            fig2, ax2 = plt.subplots(np.ceil(cx[0].shape[0]/3).astype(int), 3)
            fig2.set_size_inches(12, np.ceil(cx[0].shape[0]/3)*3)
            ax2=ax2.flatten()
            fig2.suptitle(f"for pulse {pstart}:{pend}")
            for i in range(cx[0].shape[0]):
                ax2[i].set_title(f"chan {1834+i} max: {np.argmax(np.abs(np.abs(cx[0][i,:])))}")
                ax2[i].plot(np.abs(cx[0][i,:]))
            plt.tight_layout()
            fig2.savefig(path.join(out_path,f"debug_cxcorr_{tstart}_{int(time.time())}.jpg"))

        temp_satmap.append("default")  # zeroth row is always "no phase"
        # get beamformed visibilities for each satellite
        freqs = 250e6 * (1 - np.arange(1834, 1852) / 4096)

        for i, satID in enumerate(sats):
            print("processing satellite:", satmap[satID])
            temp_satmap.append(satmap[satID])
            # phase_delay = 2 * np.pi * delays[:, i : i + 1] @ freq
            # print("phase delay shape", phase_delay.shape)
            outils.apply_delay(p0_a2, p0_a2_delayed, delays[:,i], freqs)
            cx.append(
                    outils.get_coarse_xcorr_fast(
                        p0_a1, p0_a2_delayed, dN
                    )
            )
        snr_arr = np.zeros(
            (len(sats) + 1, nchans), dtype="float64"
        )  # rows = sats, cols = channels
        detected_sats = np.zeros(nchans, dtype="int")
        # save the SNR for each channel for each satellite
        # cx[0] is the default "dont do anything" xcorr
        ax[pnum].set_title(f"Pulse {pstart} to {pend}.")
        for i in range(len(sats) + 1):
            snr_arr[i, :] = np.max(np.abs(cx[i]), axis=1) / stats.median_abs_deviation(
                np.abs(cx[i]), axis=1
            )
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
                detected_sats[chan] = temp_satmap[sortidx[-1]]
        for i, satID in enumerate(sats):
            where_sat = (
                np.where(detected_sats == satmap[satID])[0] + 1834
            )  # what channels is this sat in
            pulse_data["sats"][
                satmap[satID]
            ] = (
                where_sat.tolist()
            )  # make sure it's serializable with json. numpy array wont work
        sat_data[tstart].append(pulse_data)
    fig.savefig(
            path.join(out_path,f"debug_snr_{tstart}_{int(time.time())}.jpg")
        )
json_output = path.join(out_path,f"debug_snr_{tstart}_{int(time.time())}.json")
with open(json_output, "w") as file:
    json.dump(sat_data, file, indent=4)
print(sat_data)
