import sys
import time
import os

sys.path.insert(0, "/home/s/sievers/mohanagr/")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

T_SPECTRA = 4096 / 250e6
T_ACCLEN = 393216 * T_SPECTRA

# get a list of all direct spectra files between two timestamps

ts1 = 1627454647
# t2 = int(t1 + 560*T_SPECTRA)
ts2 = ts1 + 50
start_file = butils.get_file_from_timestamp(
    ts1, "/project/s/sievers/albatros/uapishka/202107/data_auto_cross/snap3", "d"
)[0]
start_file_tstamp = butils.get_tstamp_from_filename(start_file)
direct_files = butils.time2fnames(
    start_file_tstamp,
    ts2,
    "/project/s/sievers/albatros/uapishka/202107/data_auto_cross/snap3",
)
print(direct_files)

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
a1_coords = [51.4646065, -68.2352594, 341.052]  # north antenna
a2_coords = [51.46418956, -68.23487849, 338.32526665]  # south antenna

for file in direct_files:
    tstart = butils.get_tstamp_from_filename(file)
    nrows=560
    # tstart = ts1
    # nrows = 50
    tle_path = "/home/s/sievers/mohanagr/albatros_analysis/scripts/orbcomm/orbcomm_28July21.txt"
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, a1_coords, tstart, niter=nrows)
    print(rsats)
    for i, row in enumerate(rsats):
        for satnum, satele in row:
            arr[i][satmap[satnum]] = 1
    print(arr)
    pulses = outils.get_simul_pulses(arr)
    print(pulses)

    for (pstart, pend), sats in pulses:
        print(pstart, pend, sats)
        numsats_in_pulse = len(sats)
        t1 = tstart + pstart * T_ACCLEN
        t2 = tstart + pend * T_ACCLEN
        # t2 = t1 + 50
        print("t1 t2 for the pulse are:", t1, t2)
        files_a1, idx1 = butils.get_init_info(
            t1, t2, "/project/s/sievers/albatros/uapishka/202107/baseband/snap3/"
        )
        files_a2, idx2 = butils.get_init_info(
            t1, t2, "/project/s/sievers/albatros/uapishka/202107/baseband/snap1/"
        )
        print(files_a1)
        print(bdc.get_header(files_a1[0]))
        channels = bdc.get_header(files_a1[0])["channels"]
        chanstart = np.where(channels == 1834)[0][0]
        chanend = np.where(channels == 1854)[0][0]
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

        p0_a1 = np.zeros((size, nchans), dtype="complex64")
        p0_a2 = np.zeros((size, nchans), dtype="complex64")
        freq = 250e6 * (1 - np.arange(1834, 1854) / 4096).reshape(
            -1, nchans
        )  # get actual freq from aliasedprint("FREQ",freq/1e6," MHz")
        niter = int(t2 - t1) + 1  # run it for an extra second to avoid edge effects
        print("niter is", niter)
        delays = np.zeros((size, len(sats)))
        for i, satID in enumerate(sats):
            d = outils.get_sat_delay(
                a1_coords,
                a2_coords,
                "/home/s/sievers/mohanagr/albatros_analysis/scripts/orbcomm/orbcomm_28July21.txt",
                t1,
                niter,
                satmap[satID],
            )
            delays[:, i] = np.interp(
                np.arange(0, size) * T_SPECTRA, np.arange(0, niter), d
            )
            print(f"delay for {satmap[satID]}", delays[0:10,i], delays[-10:,i])
        for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
            perc_missing_a1 = (1 - len(chunk1["specnums"]) / size) * 100
            perc_missing_a2 = (1 - len(chunk2["specnums"]) / size) * 100
            print("missing a1", perc_missing_a1, "missing a2", perc_missing_a2)
            if perc_missing_a1 > 5 or perc_missing_a2 > 5:
                continue
            outils.make_continuous(
                p0_a1, chunk1["pol0"], chunk1["specnums"] - chunk1["specnums"][0]
            )
            outils.make_continuous(
                p0_a2, chunk2["pol0"], chunk2["specnums"] - chunk2["specnums"][0]
            )
            break
        cx = []
        N = 2 * size
        dN = min(100000, int(0.3 * N))
        stamp = slice(N // 2 - dN, N // 2 + dN)
        print(N, dN)
        temp_satmap = []  # will need to map the row number to satID later
        cx.append(
            np.fft.fftshift(outils.get_coarse_xcorr_fast(p0_a1, p0_a2), axes=1)[:, stamp]
        )  # no correction
        temp_satmap.append("default")  # zeroth row is always "no phase"
        for i, satID in enumerate(sats):
            print("processing satellite:", satmap[satID])
            temp_satmap.append(satmap[satID])
            phase_delay = 2 * np.pi * delays[:, i : i + 1] @ freq
            print("phase delay shape", phase_delay.shape)
            cx.append(
                np.fft.fftshift(
                    outils.get_coarse_xcorr_fast(p0_a1, p0_a2 * np.exp(1j * phase_delay)),
                    axes=1,
                )[:, stamp]
            )
            # print(cx[i+1][5, 1:5])
        snr_arr = np.zeros((len(sats) + 1, nchans), dtype="float64")
        detected_sats = [np.nan] * nchans
        # save the SNR for each channel for each satellite
        # cx[0] is the default "dont do anything" xcorr

        fig, ax = plt.subplots(1, 1)
        for i in range(len(sats) + 1):
            snr_arr[i, :] = np.max(np.abs(cx[i]), axis=1) / stats.median_abs_deviation(
                np.abs(cx[i]), axis=1
            )
            ax.plot(snr_arr[i, 5:15], label=f"{temp_satmap[i]}")
        plt.legend()
        plt.savefig("/scratch/s/sievers/mohanagr/debug_snr_channel.jpg")
        # for each channel, update the detected satellite for that channel
        print(snr_arr)
        for chan in range(nchans):
            sortidx = np.argsort(snr_arr[:, chan])
            print(sortidx)
            if (
                sortidx[-1] == 0
            ):  # no sat was detected, idx 0 is the default "no phase" value
                continue
            print("not continuing")
            if (snr_arr[sortidx[-1],chan] - snr_arr[sortidx[-2], chan]) / np.sqrt(
                2
            ) > 5:  # if SNR 1 = a1/sigma, SNR 2 = a2/sigma.
                # I want SNR on a1-a2 i.e. is the difference significant.
                print("top two snr for chan", chan, snr_arr[sortidx[-1],chan], snr_arr[sortidx[-2], chan])
                detected_sats[chan] = temp_satmap[sortidx[-1]]
        print(detected_sats)
