import os
import sys
import time
from os import path
sys.path.insert(0, "/home/s/sievers/thomasb/")
out_path = "/project/s/sievers/thomasb/debug"
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

#----Basic Setup----
T_SPECTRA = 4096 / 250e6
T_SCAN = 5
altitude_cutoff = 15 
array_time =  3*3600
DEBUG=True

#Unpack Config File 
with open("config.json", "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    for i, (ant, details) in enumerate(config["antennas"].items()):
        coords.append(details['coordinates'])
        dir_parents.append(details["path"])
    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    c_acclen = config["correlation"]["coarse_accumulation_length"]
    acclen = config["correlation"]["accumulation_length"]

#----Blah----
a1_coords = coords[0] 
a1_path = dir_parents[0]
satlist = [28654,25338,33591,57166,59051,44387]
satmap = {}
assert min(satlist) > len(satlist)  
for i, satnum in enumerate(satlist):
    satmap[i] = satnum
    satmap[satnum] = i

tstart = init_t 
sat_data = {} 
sat_data[tstart] = [] 
nrows = int((array_time)/T_SCAN)
arr = np.zeros((nrows, len(satlist)), dtype="int64") 
tle_path = outils.get_tle_file(tstart, "/project/s/sievers/mohanagr/OCOMM_TLES")
rsats = outils.get_risen_sats(tle_path, a1_coords, tstart, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
num_sats_risen = [len(x) for x in rsats]

for i, row in enumerate(rsats):
    for satnum, satele, sataz in row:
        arr[i,satmap[satnum]] = 1

pulses = outils.get_simul_pulses(arr)
npulses = len(pulses)
print("Sat transits detected are:", pulses, '\n')
print("Number of Pulses:", npulses, '\n')

for antnum in range(1,len(dir_parents)):
    print(f"--------------- ANTENNA {antnum}-----------------")
    a2_path = dir_parents[antnum]
    a2_coords = coords[antnum]
    snrplot, axS = plt.subplots(np.ceil(npulses/2).astype(int), 2)
    snrplot.set_size_inches(10, np.ceil(npulses/2)*4)
    snrplot.suptitle(str(tstart))
    axS=axS.flatten()

    #----Iterate over each Pulse----
    for pnum, [(pstart, pend), sats_present] in enumerate(pulses):
        print(f"--------Pulse{pnum}--------")
        numsats_in_pulse = len(sats_present)
        t1 = tstart + pstart * T_SCAN
        t2 = tstart + pend * T_SCAN

        try:
            files_a1, idx1 = butils.get_init_info(t1, t2, a1_path)
            files_a2, idx2 = butils.get_init_info(t1, t2, a2_path)
        except Exception as e:
            print(e)
            print(f"skipping pulse {pstart} to {pend} in {tstart} as some file discontinuity was encountered.")
            continue
        
        channels = np.asarray(bdc.get_header(files_a1[0])["channels"],dtype='int64')
        chanstart = np.where(channels == 1834)[0][0]
        chanend = np.where(channels == 1852)[0][0]
        nchans = chanend - chanstart

        ant1 = bdc.BasebandFileIterator(
            files_a1,
            0,
            idx1,
            c_acclen,
            None,
            chanstart=chanstart,
            chanend=chanend,
            type="float",
        )
        ant2 = bdc.BasebandFileIterator(
            files_a2,
            0,
            idx2,
            c_acclen,
            None,
            chanstart=chanstart,
            chanend=chanend,
            type="float",
        )

        p0_a1 = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
        p0_a2 = cp.zeros((c_acclen, nchans), dtype="complex64")
        p0_a2_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
        niter = int(t2 - t1) + 1  # run it for an extra second to avoid edge effects



        #----Obtain Geometric Delay----
        delays = np.zeros((c_acclen, len(sats_present)))
        for i, satidx in enumerate(sats_present):
            d = outils.get_sat_delay(
                a1_coords,
                a2_coords,
                tle_path,
                t1,
                niter,
                satmap[satidx],
            )
            delays[:, i] = np.interp(
                np.arange(0, c_acclen) * T_SPECTRA, np.arange(0, niter), d
            )
        delays = cp.asarray(delays)

        #----Verify Chunk Data Percentage----
        a1_start = ant1.spec_num_start
        a2_start = ant2.spec_num_start
        for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
            perc_missing_a1 = (1 - len(chunk1["specnums"]) / c_acclen) * 100
            perc_missing_a2 = (1 - len(chunk2["specnums"]) / c_acclen) * 100
            if perc_missing_a1 > 10 or perc_missing_a2 > 10:
                a1_start = ant1.spec_num_start
                a2_start = ant2.spec_num_start
                continue
            bdc.make_continuous_gpu(chunk1['pol0'],chunk1['specnums']-a1_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_a1)
            bdc.make_continuous_gpu(chunk2['pol0'],chunk2['specnums']-a2_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_a2)
            break


        #----Record Initial Spectrum Number Offset----

        info_tstamps_a1 = str(butils.get_tstamp_from_filename(files_a1[0]))+":"+str(ant1.spec_num_start-c_acclen)
        info_tstamps_a2 = str(butils.get_tstamp_from_filename(files_a2[0]))+":"+str(ant2.spec_num_start-c_acclen)
        specnum_offset = ant1.spec_num_start - ant2.spec_num_start #this is the initial delay between specnums when the antennas booted up
        print("initial specnum offset 1 - 2", specnum_offset)


        #----Set up Temporary Satmap----
        temp_satmap = [] 
        temp_satmap.append("Uncorrected")  # zeroth row is always "no phase" (thomas: changed from 'default' to 'Uncorrected')


        #----Coarse xcorr, WITHOUT corrections----
        cx = []  # store coarse xcorr for each satellite
        N = 2 * c_acclen
        dN = min(100000, int(0.3 * N))
        cx.append(outils_g.coarse_xcorr(p0_a1, p0_a2, dN))  # no correction

        # debug option to visually see the coarse xcorr peaks
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
            #fig2.savefig(path.join(out_path,f"debug_cxcorr_{tstart}_{pstart}_{pend}.jpg"))
            #print(path.join(out_path,f"debug_cxcorr_{tstart}_{pstart}_{pend}.jpg"))


        #----Coarse xcorr, WITH correction----
        freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)
        for i, satidx in enumerate(sats_present):
            print("Processing Satellite with ID:", satmap[satidx], "aka wrote its xcorr to cx")
            temp_satmap.append(satmap[satidx])
            # phase_delay = 2 * np.pi * delays[:, i : i + 1] @ freq
            # print("phase delay shape", phase_delay.shape)
            outils_g.apply_delay(p0_a2, delays[:,i], freqs, out=p0_a2_delayed)
            cx.append(
                    outils_g.coarse_xcorr(
                        p0_a1, p0_a2_delayed, dN
                    )
            )


        #----Get SNR----
        snr_arr = np.zeros((len(sats_present) + 1, nchans), dtype="float64")  
        axS[pnum].set_title(f"Pulse {pstart} to {pend}.")
        for i in range(len(sats_present) + 1):
            snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx[i]),axis=1))
            axS[pnum].plot(snr_arr[i, :], label=f"{temp_satmap[i]}")
        axS[pnum].set_xlabel("Channels")
        axS[pnum].set_ylabel("SNR")
        axS[pnum].legend()
        #print("SNR Array:", snr_arr)


        #---------Detect and Identify Peaks---------------------------------------
        #this here is where the work is going to be done.

        detected_sats = np.zeros(nchans, dtype="int")
        detected_peaks = np.zeros(nchans, dtype="int")
        print("NCHANS:", nchans)

        for chan in range(nchans):
            sortidx = np.argsort(snr_arr[:, chan])
            if (sortidx[-1] == 0):  # no sat was detected, idx 0 is the default "no phase" value
                print(f"No Peak in Channel {chan}")
                continue

            #here is the condition for being a peak. I do not understand it.

            if (snr_arr[sortidx[-1], chan] - snr_arr[sortidx[-2], chan]) / np.sqrt(2) > 5:  
                # if SNR 1 = a1/sigma, SNR 2 = a2/sigma.
                # I want SNR on a1-a2 i.e. is the difference significant.
                # print(
                #     "top two snr for chan",
                #     chan,
                #     snr_arr[sortidx[-1], chan],
                #     snr_arr[sortidx[-2], chan],
                # )
                print(f"Processing Peak in Channel {chan}")
                cx_idx = sortidx[-1] # which cxcorr has the detection
                satID = temp_satmap[cx_idx]
                print("SatID of detected peak:", satID)
            

                # these plots are for the channels where something was detected
                if DEBUG: 
                    fig2, ax2 = plt.subplots(np.ceil(cx[cx_idx].shape[0]/3).astype(int), 3)
                    fig2.set_size_inches(12, np.ceil(cx[cx_idx].shape[0]/3)*3)
                    ax2=ax2.flatten()
                    fig2.suptitle(f"for pulse {pstart}:{pend}, sat {satID}")
                    for i in range(cx[cx_idx].shape[0]):
                        mm=cp.argmax(cp.abs(cx[cx_idx][i,:]))
                        ax2[i].set_title(f"chan {1834+i} max: {mm}")
                        ax2[i].plot(cp.asnumpy(cp.abs(cx[cx_idx][i,:])))
                        # ax2[i].set_xlim(mm-1000,mm+1000)
                    plt.tight_layout()
                    fig2.savefig(path.join(out_path,f"dg_cxdet_{tstart}_{pstart}_{pend}_sat{satID}.jpg"))
                    print("Saving Plots as:", path.join(out_path,f"dg_cxdet_{tstart}_{pstart}_{pend}_sat{satID}.jpg"))
                
                detected_sats[chan] = temp_satmap[sortidx[-1]]
                detected_peaks[chan] = cp.argmax(cp.abs(cx[sortidx[-1]][chan,:]))

            else:
                print("Neither is covered")

        print("Detected Sats:", detected_sats)
        print("Detected Peaks:", detected_peaks)
        



        #----Update Spectrum Number Offset----
        sat_peaks=[]
        if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
            peak_guesses = detected_peaks[detected_peaks>0]
            print("detected peak locations are", peak_guesses)
            best_guess_offset = np.max(detected_peaks)  #use the maximum peak to determine best guess offset for each pulse
            if (best_guess_offset-dN) > 0:
                    #tau > 0; idxstart0 += detected_peaks[chan]-dN
                    specnum_offset += (best_guess_offset-dN)
            else:
                #tau < 0; idxstart1 += abs(best_guess_offset-dN)
                specnum_offset -= np.abs(best_guess_offset - dN)
            print("specnum offset updated to:", specnum_offset)
        else:
            print("No detected peaks for this pulse")

        
        #----Store Pulse Data----

        #create dictionary to store pulse information
        pulse_data = {}
        pulse_data["start"] = pstart
        pulse_data["end"] = pend
        pulse_data["sats_present"] = {}
        numsats_in_pulse = len(sats_present)
    
        for i, satID in enumerate(sats_present):
            where_sat = np.where(detected_sats == satmap[satID])[0]
            for ids in where_sat:
                sat_peaks.append([int(ids)+1834, int(detected_peaks[ids])]) #append the channel and the peak location of that channel
            pulse_data["sats_present"][satmap[satID]] = sat_peaks # make sure it's serializable with json. numpy array wont work

        #Finally, apprend pulse data to satellite_data
        sat_data[tstart].append(pulse_data)
    


    #----Save SNR Debug for each Antenna----

    snrplot.subplots_adjust(hspace=0.6)
    snrplot.savefig(
            path.join(out_path,f"debug_snr_ant{antnum}_{tstart}_{int(time.time())}.jpg")  #update antenna number?? +1?
        )
    print(path.join(out_path,f"debug_snr_ant{antnum}_{tstart}_{int(time.time())}.jpg"))


#----Save Pulse Data to Json for each Antenna
#question: how do we want to configure the json to read off the antenna information?
#          because as of now, no antenna information encoded directly. 
#          could add an extra dictionary element which gives the antenna, may be a move.

json_output = path.join(out_path,f"pulsedata_{tstart}_{int(time.time())}.json")
with open(json_output, "w") as file:
   json.dump(sat_data, file, indent=4)
print(sat_data)




