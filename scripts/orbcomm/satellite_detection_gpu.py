import os
import sys
import time
from os import path
sys.path.insert(0, "/home/s/sievers/thomasb/")
out_path = "/project/s/sievers/thomasb/"
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import json
import cProfile, pstats
import cupy as cp


#----Basic Setup----

T_SPECTRA = 4096 / 250e6
T_SCAN = 5 #seconds between each pulse scan -- look for sat rise/set every 5 sec.
altitude_cutoff = 15  #cutoff when looking for satellites
array_time =  3*3600  #how long we look for pulses after the start time (in seconds)
DEBUG=False
ONLY_RUSSIANS = True

#----Unpack Config File Data----

with open("config.json", "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    # unpack information from the json file
    # Call get_starting_index for all antennas except reference
    print('\n', "Antenna Details:")
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ant, details)
        coords.append(details['coordinates'])
        dir_parents.append(details["path"])
    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    c_acclen = config["correlation"]["coarse_accumulation_length"]
    acclen = config["correlation"]["accumulation_length"]

print("Antenna Coordinates:", coords, '\n')
print("Coarse Accumulation Length", c_acclen, '\n')
print("Accumulation Length:", acclen, '\n')


#----Define Reference Antenna----

a1_coords = coords[0] 
a1_path = dir_parents[0]


#----Make Satellite List----
# (currently) hard coded list of satellites we track
# generalize in the future

satlist = [28654,25338,33591,57166,59051,44387]
# satlist = [57166,59051]



#----Make Satellite Map----
#this is used throughout the code to identify the overall satellite ID (e.g. 33591) to its index in the satlist (e.g. 2)

satmap = {}
assert min(satlist) > len(
    satlist
)  # to make sure there are no collisions, we'll never have an i that's also a satnum
for i, satnum in enumerate(satlist):
    satmap[i] = satnum
    satmap[satnum] = i
# print(satmap)



#----Obtain and Plot the Risen Satellites----
# arr: temporary array where we store which satellite has a pulse active, using 1 and 0 (on/off)
# rsats: list of lists of coordinates and ID of satellites visible (aka risen), entry each dt (usually 5 secs)
# num_sats_risen: integer list with number of visible satellites at that point, entry each dt
# sat_data: dictionary where we will eventually store all pulse data

#questions:  how long do the TLE files go on for/ how long are they reliable?
#            why do we choose 3 hours from the start specifically?
#            does this mean that we only care about the 3 hours after the start time to find our spectrum number offsets?

tstart = init_t #define start time (observer frame)
sat_data = {} 
sat_data[tstart] = [] 
nrows = int((array_time)/T_SCAN)
arr = np.zeros((nrows, len(satlist)), dtype="int64") #array 
tle_path = outils.get_tle_file(tstart, "/project/s/sievers/mohanagr/OCOMM_TLES")
print("Using TLE path:", tle_path, '\n')
rsats = outils.get_risen_sats(tle_path, a1_coords, tstart, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
num_sats_risen = [len(x) for x in rsats]

#plot risen sats
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10,4)
fig.suptitle(f"Risen sats for file {tstart}")
ax[0].plot(num_sats_risen)
ax[0].set_xlabel("time (in units of 5 sec)")
for i, row in enumerate(rsats):
    for satnum, satele, sataz in row:
        arr[i,satmap[satnum]] = 1
ax[1].set_ylabel("time in units of 5 sec")
ax[1].set_xlabel("Sat ID") #Sat ID with respect to the satmap dictionary index
ax[1].imshow(arr,aspect='auto',interpolation="none")
plt.tight_layout()
fig.savefig(path.join(out_path,f"risen_sats_{tstart}_{str(time.time())}.jpg"))
print(arr)



#----Get Pulses, AKA Satellite Transits----
# pulses is in the form of [[start, end], [sats_present]]. sats_present is identified in terms of its index in satlist, by the way
# just a practical rearranging of the previously obtained data.

pulses = outils.get_simul_pulses(arr)
npulses = len(pulses)
print("Sat transits detected are:", pulses, '\n')
print("Number of Pulses:", npulses, '\n')




#----Iterate over each Antenna----

for antnum in range(1,len(dir_parents)):
    print(f"--------------- ANTENNA {antnum}-----------------")

    #setup paths and coordinates for each antenna at the start of the loop
    a2_path = dir_parents[antnum]
    a2_coords = coords[antnum]

    #define SNR plots for later (here defined for one antenna, all pulses)
    #snrplot, axS = plt.subplots(npulses, 1, figsize=(6, 2 * npulses), sharex=True)

    snrplot, axS = plt.subplots(np.ceil(npulses/2).astype(int), 2)
    snrplot.set_size_inches(10, np.ceil(npulses/2)*4)
    snrplot.suptitle(str(tstart))
    axS=axS.flatten()

    #----Iterate over each Pulse----

    for pnum, [(pstart, pend), sats_present] in enumerate(pulses):
        print(f"------Pulse Number {pnum}-------")
        print("Pulse Start Idx:", pstart)
        print("Pulse End Idx:", pend)
        print("Satelite Idxs Present:", sats_present, '\n')
        numsats_in_pulse = len(sats_present)

        #define our pulse length
        t1 = tstart + pstart * T_SCAN
        t2 = tstart + pend * T_SCAN
        print("In ctime we measure Pulse t1:", t1, "Pulse t2:", t2)
        print("Length of Pulse is:", t2-t1, '\n')

        #this is buggy and sets off the exception below. fix so that max 50 seconds of pulse read.
        #if (pend-pstart)*T_SCAN > 50:
        #    t2 = t1 + 50
        #else:
        #    t2 = tstart + pend * T_SCAN
        #print("Full time difference:", tstart + pend * T_SCAN - t1)
        #print("Maxed time difference:", t2-t1)

        # Make sure no problem in files
        try:
            files_a1, idx1 = butils.get_init_info(t1, t2, a1_path)
            files_a2, idx2 = butils.get_init_info(t1, t2, a2_path)
        except Exception as e:
            print(e)
            print(f"skipping pulse {pstart} to {pend} in {tstart} as some file discontinuity was encountered.")
            continue
        # print(files_a1,files_a2)

        # Set up the number of channels we look through
        channels = np.asarray(bdc.get_header(files_a1[0])["channels"],dtype='int64')
        chanstart = np.where(channels == 1834)[0][0]
        chanend = np.where(channels == 1852)[0][0]
        nchans = chanend - chanstart


        # recall we defined acclen above with the config file unpacking
        # dont impose any chunk num, continue iterating as long as a chunk with small enough missing fraction is found.
        # have passed enough files to begin with. should not run out of files.
        print("Setting Antenna as BFI Objects", '\n')

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

        #set up our polarization arrays 
        # question:  why do we only do this for one chunk from the unpacked data?

        p0_a1 = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
        p0_a2 = cp.zeros((c_acclen, nchans), dtype="complex64")
        p0_a2_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
        niter = int(t2 - t1) + 1  # run it for an extra second to avoid edge effects
        print("niter for delay is", niter, "t1 is", t1)



        #----Obtain Geometric Delay----
        # get geo delay for each satellite present in the pulse from Skyfield (often just one or two)

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
            # print(f"delay for {satmap[satID]}", delays[0:10,i], delays[-10:,i])
        # get baseband chunk for the duration of required transit. Take the first chunk `acclen` long that satisfies missing packet requirement
        # print(delays[0:10,1], delays[-10:,1])
        # print(delays[0:10,0], delays[-10:,0])
        delays = cp.asarray(delays)



        #----Verify Chunk Data Percentage----
        #iterates so that we take the first chunk which is above tolenance fill (I THINK?)

        a1_start = ant1.spec_num_start
        a2_start = ant2.spec_num_start
        for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
            perc_missing_a1 = (1 - len(chunk1["specnums"]) / c_acclen) * 100
            perc_missing_a2 = (1 - len(chunk2["specnums"]) / c_acclen) * 100
            print("missing a1", perc_missing_a1, "missing a2", perc_missing_a2)
            if perc_missing_a1 > 10 or perc_missing_a2 > 10:
                a1_start = ant1.spec_num_start
                a2_start = ant2.spec_num_start
                continue
            # print(chunk1["pol0"])
            # print(chunk2["pol0"])
            # print("nchans is", nchans)
            # print("first row chunk1", chunk1['pol0'][0,:])
            bdc.make_continuous_gpu(chunk1['pol0'],chunk1['specnums']-a1_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_a1)
            bdc.make_continuous_gpu(chunk2['pol0'],chunk2['specnums']-a2_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_a2)
            # print("first row p0_a1", p0_a1[0,:])
            break


        #----Record Initial Spectrum Number Offset----
        #If we detect a sat, note down what file we started from and what the specnum at the start was. 

        info_tstamps_a1 = str(butils.get_tstamp_from_filename(files_a1[0]))+":"+str(ant1.spec_num_start-c_acclen)
        info_tstamps_a2 = str(butils.get_tstamp_from_filename(files_a2[0]))+":"+str(ant2.spec_num_start-c_acclen)
        specnum_offset = ant1.spec_num_start - ant2.spec_num_start #this is the initial delay between specnums when the antennas booted up
        print(info_tstamps_a1)
        print(info_tstamps_a2)
        print("initial specnum offset 1 - 2", specnum_offset)


        #----Set up Temporary Satmap----
        # cx will have multiple entries, first one is no phase correction, next ones are for specific satellite geo offsets
        # temp_satmap has satellite ID in the correct index (e.g. ['Uncorrected', 33591] for single sat present)
        # i.e. maps row number of cx to satID. Works off of index of 'sats_present'

        temp_satmap = [] 
        temp_satmap.append("Uncorrected")  # zeroth row is always "no phase" (thomas: changed from 'default' to 'Uncorrected')



        #----Coarse xcorr, WITHOUT corrections----

        cx = []  # store coarse xcorr for each satellite
        N = 2 * c_acclen
        dN = min(100000, int(0.3 * N))
        print("2*N and 2*dN", N, dN)
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
            fig2.savefig(path.join(out_path,f"dg_cxcorr_{tstart}_{pstart}_{pend}.jpg"))
            print(path.join(out_path,f"dg_cxcorr_{tstart}_{pstart}_{pend}.jpg"))
            # sys.exit(0)

        

        #----Coarse xcorr, WITH correction----
        # aka beamformed visibilities
        # thomas: changed "satID" to "satidx" since in 'sats_present' we have the satelitte indices from satlist present,
        #         then convert to their satID using satmap. More intuitive to me.
    
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
        if DEBUG:
            fig2, ax2 = plt.subplots(np.ceil(cx[1].shape[0]/3).astype(int), 3)
            fig2.set_size_inches(12, np.ceil(cx[1].shape[0]/3)*3)
            ax2=ax2.flatten()
            fig2.suptitle(f"for pulse {pstart}:{pend}")
            for i in range(cx[1].shape[0]):
                mm=cp.argmax(cp.abs(cx[1][i,:]))
                ax2[i].set_title(f"chan {1834+i} max: {mm}")
                ax2[i].plot(cp.asnumpy(cp.abs(cx[1][i,:])))
                # ax2[i].set_xlim(mm-1000,mm+1000)
            plt.tight_layout()
            fig2.savefig(path.join(out_path,f"dg_CORR_cxcorr_{tstart}_{pstart}_{pend}.jpg"))
            print(path.join(out_path,f"dg_CORR_cxcorr_{tstart}_{pstart}_{pend}.jpg"))
            # sys.exit(0)

        

        #----Get SNR----
        # we want an array of the SNR for each channel for each satellite (plus uncorrected)

        snr_arr = np.zeros((len(sats_present) + 1, nchans), dtype="float64")  
        # set up plot for all pulses, finished at end of big pulse loop
        # beware: need a 2D array of plots here. For each antenna (row) have a column of pnum (column) plots


        axS[pnum].set_title(f"Pulse {pstart} to {pend}.")
        for i in range(len(sats_present) + 1):
            snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx[i]),axis=1))
            axS[pnum].plot(snr_arr[i, :], label=f"{temp_satmap[i]}")
        axS[pnum].set_xlabel("Channels")
        axS[pnum].set_ylabel("SNR")
        axS[pnum].legend()
        #print("SNR Array:", snr_arr)



        #----Detect Peaks----
        # rows = sats_present, cols = channels
        # questions: does the first if statement mean uncorrected is larger than corrected?
        #            in the second if statement, we could be comparing between satellites no? difference between the two SNRs?

        detected_sats = np.zeros(nchans, dtype="int")
        detected_peaks = np.zeros(nchans, dtype="int")

        print('\nStarting to Process SNRs \n')

        for chan in range(nchans):
            sortidx = np.argsort(snr_arr[:, chan])
            #if the index of maximum SNR is the 'uncorrected' value, then no sat detected.
            if (sortidx[-1] == 0):  
                continue
            #below is the minimum condition of SNR for a detection. Most basic requirement
            if (snr_arr[sortidx[-1], chan] - snr_arr[sortidx[-2], chan]) / np.sqrt(2) > 5: 
                # if SNR 1 = a1/sigma, SNR 2 = a2/sigma.
                # I want SNR on a1-a2 i.e. is the difference significant.
                # print(
                #     "top two snr for chan",
                #     chan,
                #     snr_arr[sortidx[-1], chan],
                #     snr_arr[sortidx[-2], chan],
                # )
                cx_idx = sortidx[-1] # which cxcorr has the detection
                print(f"\nDetected Peak in cx index {cx_idx} in channel {chan}")
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
                    print(path.join(out_path,f"dg_cxdet_{tstart}_{pstart}_{pend}_sat{satID}.jpg"))

                #this is a requirement that can be turned on to identify reliable peaks only
                if ONLY_RUSSIANS:
                    data = cp.abs(cx[sortidx[-1]][chan,:])
                    peak_location = cp.argmax(data)
                    peak_data = data[peak_location - 200:peak_location + 200]
                    data_cpu = cp.asnumpy(data)
                    peaks_total = find_peaks(data_cpu, height=0.001)
                    heights = peaks_total[1]['peak_heights']
                    height_indices = np.argsort(heights)

                    tallest = heights[height_indices[-1]]
                    reps, total = 4, 0
                    for i in range(reps):
                        total += (tallest - heights[height_indices[-(i+2)]])
                    ratio = (total/(tallest * reps))
                    print("\nRATIO:", ratio)

                    if ratio > 0.6:
                        print("CCCP")
                    if ratio < 0.1:
                        print("AMERICAN")
                    else:
                        print("undecided?")

                # if we actually detect the peak in the data, we add it to these arrays
                # note that present and detected are different: present is from prediction, detected is from data.
                detected_sats[chan] = temp_satmap[sortidx[-1]]
                detected_peaks[chan] = cp.argmax(cp.abs(cx[sortidx[-1]][chan,:]))
                # detected_sats[chan] = satmap[sats_present[sortidx[-1]]]

        print("\nDetected Sats:", detected_sats)
        print("Detected Peaks:", detected_peaks)
        



        #----Update Spectrum Number Offset----
        #question:  if we detect multiple conflicting offsets for different pulses, do they just overwrite each other?

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



