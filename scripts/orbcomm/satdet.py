import os
import sys
import time
from os import path
sys.path.insert(0, "/home/thomasb/")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
import cupy as cp
import argparse
import json
from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import cProfile, pstats


'''
general notes:

- is the TLE file reliable the whole way through a full day?
- change freqs for full functionality with different spectra types and such
- careful with satmap vs satID vs satidx and such



explanations of what's going on:

important convention:  a satellite PASS is when we know from TLE it is overhead. a satellite PULSE is when we can confirm we've seen it in our data.
(to me, it's useful to make sure we separate our seen data and our predicted knowledge of the sats)

arr:             temporary array where we store which satellite has a pass active, using 1 and 0 (on/off)
rsats:           list of lists of coordinates and ID of satellites visible (aka risen), entry each T_SCAN seconds (default 5s)
num_sats_risen:  integer list with number of visible satellites at that point, entry each dt
sat_data:        dictionary where we will eventually store all pulse data


passes:          list of all recorded satellite passes, in the form of [[start, end], [sats_present]]
sats_present:    list of the satellites present in a specific pass. The sat in question is identified by its index in satlist

(for one pass)
cx:              list of coarse cross-correlations. Done for each present satellite, as well as one uncorrected (non-beamformed)
temp_satmap:     has satellite ID in the correct index (e.g. ['Uncorrected', 33591] for single sat present). lets you move from indices to their actual satID

snr_arr:         an array of the SNR for each channel for each satellite (plus uncorrected aka non-beamformed), shape (numsats_in_pass + 1, nchans)

'''



#will come in handy later
def get_mode(data):
    nonzero_data = [x for x in data if x != 0]
    if len(nonzero_data) < 3:
        print("not a lot of reliable offsets, mode may be unstable")
        return "empty"
    else:
        return int(stats.mode(nonzero_data)[0])



#--------------terminal functionality-----------


if __name__ == "__main__":
    
    #required argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Config file containing all required data.",
    )


    #optional arguments
    parser.add_argument(
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for debug and pulses"
    )

    parser.add_argument(
        "-e", "--extra_data", action="store_true", help="Adds extra data to json dump, such as coarse offset and reliability ratio"
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="debug option that spits out a ton of plots to see stuff"
    )

    parser.add_argument(
        "--T_spectra", type=float, default= 4096 / 250e6, help="lets you change spectrum period from default"
    )

    parser.add_argument(
        "--T_scan", type = int, default=5, help="lets you change period of satellite scans from default of 5 secs"
    )

    parser.add_argument(
        "--alt_cutoff", type = int, default=15, help="lets you change cutoff altitude from default of 15 degrees"
    )

    parser.add_argument(
        "-gr", "--generalized_offsets", default = True, help = "For each pulse adds the actual offset between the two files. This is done for each pulse to avoid rollover errors between days."
    )

    args = parser.parse_args()


    #--------------------set variables------------------
    T_SPECTRA = args.T_spectra
    T_SCAN = args.T_scan #seconds between each satellite risen scan -- look for sat rise/set every 5 sec.
    altitude_cutoff = args.alt_cutoff  #cutoff when looking for satellites
    out_path = args.output_path


    #--------------------extract config data------------------

    with open(args.config_file, "r") as f:
        config = json.load(f)
        dir_parents = []
        coords = []

        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            print(ant, details)
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])

        global_start_t = config["correlation"]["start_timestamp"] #global as it is in unix time, reference frame
        global_end_t = config["correlation"]["end_timestamp"]
        c_acclen = config["correlation"]["coarse_acclen"]
        v_acclen = config["correlation"]["vis_acclen"]

    print("\nAntenna Coordinates:", coords)
    print("Coarse Accumulation Length", c_acclen)
    print("Visibility Accumulation Length:", v_acclen, '\n')


    #-----------------Setup-------------
    # ra refers to Reference Antenna, nra refers to Non-Reference Antenna
    array_time =  global_end_t - global_start_t
    ra_coords = coords[0] 
    ra_path = dir_parents[0]

    satlist = [28654,25338,33591,57166,59051,44387]
    # satlist = [57166,59051]

    # used throughout the code to map satellite ID (e.g. 33591) to its index in the satlist (e.g. 2), without collisions
    satmap = {}
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i


    #------------------get and plot risen sats-----------------
    nrows = int((array_time)/T_SCAN)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    print("Using TLE path:", tle_path, '\n')
    rsats = outils.get_risen_sats(tle_path, ra_coords, global_start_t, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
    num_sats_risen = [len(x) for x in rsats]

    #plot risen sats
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10,4)
    fig.suptitle(f"Risen sats for starting time {global_start_t}")
    ax[0].plot(num_sats_risen)
    ax[0].set_xlabel("time (in units of 5 sec)")
    for i, row in enumerate(rsats):
        for sat_ID, satele, sataz in row:
            arr[i,satmap[sat_ID]] = 1
    ax[1].set_ylabel("time in units of 5 sec")
    ax[1].set_xlabel("Sat Index (from satlist)") #Sat Index with respect to the satlist dictionary indexing, corresponds to an actual satellite ID
    ax[1].imshow(arr,aspect='auto',interpolation="none")
    plt.tight_layout()
    fig.savefig(path.join(out_path,f"risen_sats_{global_start_t}_{str(time.time())}.jpg"))
    print(arr)


    #-----------------get passes-------------
    passes = outils.get_simul_pulses(arr) #beware the function is named pulses
    print(passes)
    print(type(passes))
    npasses = len(passes)
    print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')

    sat_data = {} 
    sat_data[global_start_t] = {}  #here you can modify how you want to store the data, if you want


    #------------------Iterate over each Antenna------------------

    for antnum in range(1,len(dir_parents)):
        print(f"--------------- ANTENNA {antnum}-----------------")

        sat_data[global_start_t][f"antenna {antnum}"] = []
        nra_path = dir_parents[antnum] 
        nra_coords = coords[antnum]
        snrplot, axS = plt.subplots(np.ceil(npasses/2).astype(int), 2)
        snrplot.set_size_inches(10, np.ceil(npasses/2)*4)
        snrplot.suptitle(str(global_start_t))
        axS=axS.flatten()

        #--------Iterate over each Pass--------

        for pnum, [(pstart, pend), sats_present] in enumerate(passes):

            print(f"------Pass Number {pnum}-------")
            print("Pass Start Idx:", pstart)
            print("Pass End Idx:", pend)
            print("Satelite Idxs Present:", sats_present, '\n')
            numsats_in_pass = len(sats_present)

            #define our pass length
            t1 = global_start_t + pstart * T_SCAN
            t2 = global_start_t + pend * T_SCAN  #might need to shorten the pass duration for practicality
            print("Pass Duration:", t2-t1, '\n')

            # Make sure no problem in files
            try:
                files_ra, idx_ra = butils.get_init_info(t1, t2, ra_path)
                files_nra, idx_nra = butils.get_init_info(t1, t2, nra_path)
            except Exception as e:
                print(e)
                print(f"skipping pass {pstart} to {pend} in {global_start_t} as some file discontinuity was encountered.")
                continue

            print("Setting Antenna as BFI Objects", '\n')

            # Set up the number of channels we look through
            channels = np.asarray(bdc.get_header(files_ra[0])["channels"],dtype='int64')
            chanstart = np.where(channels == 1834)[0][0]
            chanend = np.where(channels == 1852)[0][0]
            nchans = chanend - chanstart

            # dont impose any chunk num, continue iterating as long as a chunk with small enough missing fraction is found.
            # have passed enough files to begin with. should not run out of files.
            
            ra = bdc.BasebandFileIterator(
                files_ra,
                0,
                idx_ra,
                c_acclen,
                None,
                chanstart=chanstart,
                chanend=chanend,
                type="float",
            )
            nra = bdc.BasebandFileIterator(
                files_nra,
                0,
                idx_nra,
                c_acclen,
                None,
                chanstart=chanstart,
                chanend=chanend,
                type="float",
            )

            #--------set up pol arrays---------

            p0_ra = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
            p0_nra = cp.zeros((c_acclen, nchans), dtype="complex64")
            p0_nra_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
            niter = int(t2 - t1) + 1  # run it for an extra second to avoid edge effects

            #--------get geo delay (done for each sat in pass)-------
            delays = np.zeros((c_acclen, numsats_in_pass))
            for i, satidx in enumerate(sats_present):
                d = outils.get_sat_delay(
                    ra_coords,
                    nra_coords,
                    tle_path,
                    t1,
                    niter,
                    satmap[satidx],
                )
                delays[:, i] = np.interp(
                    np.arange(0, c_acclen) * T_SPECTRA, np.arange(0, niter), d
                )
            delays = cp.asarray(delays)

            #---------check chunk data percentage--------
            #iterates so that we take the first chunk which is above tolenance fill
            ra_start = ra.spec_num_start
            nra_start = nra.spec_num_start
            for i, (chunk_ra, chunk_nra) in enumerate(zip(ra, nra)):
                perc_missing_ra = (1 - len(chunk_ra["specnums"]) / c_acclen) * 100
                perc_missing_nra = (1 - len(chunk_nra["specnums"]) / c_acclen) * 100
                print("missing a1", perc_missing_ra, "missing a2", perc_missing_nra)
                if perc_missing_ra > 10 or perc_missing_nra > 10:
                    ra_start = ra.spec_num_start
                    nra_start = nra.spec_num_start
                    continue
                
                bdc.make_continuous_gpu(chunk_ra['pol0'],chunk_ra['specnums']-ra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_ra)
                bdc.make_continuous_gpu(chunk_nra['pol0'],chunk_nra['specnums']-nra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_nra)
                break


            #----Record Initial Spectrum Number Offset----
            #If we detect a sat, note down what file we started from and what the specnum at the start was. 

            info_tstamps_ra = str(butils.get_tstamp_from_filename(files_ra[0]))+":"+str(ra.spec_num_start-c_acclen)
            info_tstamps_nra = str(butils.get_tstamp_from_filename(files_nra[0]))+":"+str(nra.spec_num_start-c_acclen)
            specnum_offset = ra.spec_num_start - nra.spec_num_start #this is the initial delay between specnums when the antennas booted up

            #----Set up Temporary Satmap----
            temp_satmap = [] 
            temp_satmap.append("Uncorrected")  # zeroth row is always "no phase"

            #----coarse xcorr, WITHOUT corrections----

            cx = []  # store coarse xcorr for each satellite
            N = 2 * c_acclen
            dN = min(100000, int(0.3 * N))
            print("2*N and 2*dN", N, dN)
            cx.append(outils_g.coarse_xcorr(p0_ra, p0_nra, dN))  # no correction

            # debug option to visually see the coarse xcorr peaks
            if args.debug:
                fig2, ax2 = plt.subplots(np.ceil(cx[0].shape[0]/3).astype(int), 3)
                fig2.set_size_inches(12, np.ceil(cx[0].shape[0]/3)*3)
                ax2=ax2.flatten()
                fig2.suptitle(f"for pass {pstart}:{pend}")
                for i in range(cx[0].shape[0]):
                    mm=cp.argmax(cp.abs(cx[0][i,:]))
                    ax2[i].set_title(f"chan {1834+i} max: {mm}")
                    ax2[i].plot(cp.asnumpy(cp.abs(cx[0][i,:])))
                    # ax2[i].set_xlim(mm-1000,mm+1000)
                plt.tight_layout()
                fig2.savefig(path.join(out_path,f"UNCORR_pass{pstart}_ant{antnum}.jpg"))
                print(path.join(out_path,f"dg_cxcorr_{global_start_t}_{pstart}_{pend}.jpg"))
                # sys.exit(0)


            #----Coarse xcorr, WITH correction----
            # aka beamformed visibilities
        
            freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)   #hard-coded. need to change for full spectra-variable functionality
            for i, satidx in enumerate(sats_present):
                print("\nProcessing Satellite with ID:", satmap[satidx])
                temp_satmap.append(satmap[satidx])
                # phase_delay = 2 * np.pi * delays[:, i : i + 1] @ freq
                # print("phase delay shape", phase_delay.shape)
                outils_g.apply_delay(p0_nra, delays[:,i], freqs, out=p0_nra_delayed)
                cx.append(
                        outils_g.coarse_xcorr(
                            p0_ra, p0_nra_delayed, dN
                        )
                )
            if args.debug:
                fig2, ax2 = plt.subplots(np.ceil(cx[1].shape[0]/3).astype(int), 3)
                fig2.set_size_inches(12, np.ceil(cx[1].shape[0]/3)*3)
                ax2=ax2.flatten()
                fig2.suptitle(f"for pass {pstart}:{pend}")
                for i in range(cx[1].shape[0]):
                    mm=cp.argmax(cp.abs(cx[1][i,:]))
                    ax2[i].set_title(f"chan {1834+i} max: {mm}")
                    ax2[i].plot(cp.asnumpy(cp.abs(cx[1][i,:])))
                plt.tight_layout()
                fig2.savefig(path.join(out_path,f"CORR_pass{pstart}_ant{antnum}.jpg"))
                

            #------------Get SNR----------
            # want an array of the SNR for each channel for each satellite (plus uncorrected)
            snr_arr = np.zeros((numsats_in_pass + 1, nchans), dtype="float64")  


            # beware: need a 2D array of plots here. For each antenna (row) have a column of pnum (column) plots
            # they're all put into one big figure at the end
            axS[pnum].set_title(f"Pulse {pstart} to {pend}.")
            for i in range(numsats_in_pass + 1):
                snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx[i]),axis=1))
                axS[pnum].plot(snr_arr[i, :], label=f"{temp_satmap[i]}")
            axS[pnum].set_xlabel("Channels")
            axS[pnum].set_ylabel("SNR")
            axS[pnum].legend()


            #----Detect Peaks----
            # rows = sats_present, cols = channels
            detected_sats = np.zeros(nchans, dtype="int")
            detected_peaks = np.zeros(nchans, dtype="int")

            print('Processing SNRs \n')

            pulse_ratios = {}

            for chan in range(nchans):
                sortidx = np.argsort(snr_arr[:, chan])
                #if the index of maximum SNR is the 'uncorrected' value, then no sat detected.
                if (sortidx[-1] == 0):  
                    continue
                #below is the minimum condition of SNR for a detection. Most basic requirement
                if (snr_arr[sortidx[-1], chan] - snr_arr[sortidx[-2], chan]) / np.sqrt(2) > 5: 
                    # if SNR 1 = ra/sigma, SNR nra = a2/sigma.
                    # I want SNR on ra-nra i.e. is the difference significant.
                    
                    cx_idx = sortidx[-1] # which cxcorr has the detection
                    print(f"\nDetected Peak in cx index {cx_idx} in channel {chan}")
                    satID = temp_satmap[cx_idx]
                    print("SatID of detected peak:", satID)

                    # these plots are for the channels where something was detected
                    if args.debug: 
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
                        fig2.savefig(path.join(out_path,f"DETECT_ant{antnum}_pulse{pstart}_sat{satID}.jpg"))

                    data_gpu = cp.abs(cx[sortidx[-1]][chan,:])
                    data_cpu = cp.asnumpy(data_gpu)

                    peak_location = cp.argmax(data_gpu)
                    peak_data = data_gpu[peak_location - 200:peak_location + 200]
                    peaks_total = find_peaks(data_cpu, height=0.001)

                    heights = peaks_total[1]['peak_heights']
                    height_indices = np.argsort(heights)
                    tallest = heights[height_indices[-1]]
                    reps, total = 4, 0
                    for i in range(reps):
                        total += (tallest - heights[height_indices[-(i+2)]])
                    reliability_ratio = (total/(tallest * reps))
                    print("\nRATIO:", reliability_ratio)

                    pulse_label = ""
                    if reliability_ratio < 0.15:
                        pulse_label = "UNRELIABLE"
                    elif reliability_ratio > 0.8:
                        pulse_label = "RELIABLE"
                    else:
                        pulse_label = "UNCLEAR"
                    print(pulse_label)

                    pulse_ratios[chan] = (reliability_ratio, pulse_label)
                
                    #detected = graduates from pass to pulse. also picks what channels detection happens
                    detected_sats[chan] = temp_satmap[sortidx[-1]]
                    detected_peaks[chan] = cp.argmax(cp.abs(cx[sortidx[-1]][chan,:]))
            

            #----Update Spectrum Number Offset----

            if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
                peak_guesses = detected_peaks[detected_peaks>0]
                print("detected peak locations are", peak_guesses)
                best_guess_offset = np.max(detected_peaks)  #use the maximum peak to determine best guess offset for each pulse
                print(best_guess_offset)
                if (best_guess_offset-dN) > 0:
                        #tau > 0; idxstart0 += detected_peaks[chan]-dN
                        specnum_offset += (best_guess_offset-dN)
                else:
                    #tau < 0; idxstart1 += abs(best_guess_offset-dN)
                    specnum_offset -=  np.abs(best_guess_offset - dN)
                print("specnum offset updated to:", specnum_offset)
            else:
                print("No detected peaks for this pulse")

            
            #-----------------Store Pulse Data--------------
            #create dictionary to store pulse information (but only if there was a detection)
            #reset these variables to make sure we don't add more than the current pulse's info
            sat_peaks = []
            pulse_data = {}

            if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
                pulse_data["start"] = pstart
                pulse_data["end"] = pend
                pulse_data["sats_present"] = {}
                pulse_data["individual_offset"] = int(specnum_offset)
            
                for i, sat_ID in enumerate(sats_present):
                    where_sat = np.where(detected_sats == satmap[sat_ID])[0]
            
                    for chanidx in where_sat:
                        if args.extra_data:
                            sat_peaks.append([int(chanidx)+1834, int(detected_peaks[chanidx]), pulse_ratios[chanidx]]) #append the channel and the peak location of that channel
                        else:
                            sat_peaks.append([int(chanidx)+1834, pulse_ratios[chanidx][1]])

                    pulse_data["sats_present"][satmap[sat_ID]] = sat_peaks # make sure it's serializable with json. numpy array wont work

                sat_data[global_start_t][f"antenna {antnum}"].append(pulse_data)
        

        #----Save SNR Debug for each Antenna----

        snrplot.subplots_adjust(hspace=0.6)
        snrplot.savefig(
                path.join(out_path,f"SNR_ant{antnum}_datastart{global_start_t}.jpg")  #update antenna number?? +1?
            )
    


        #--------Applying generalized offsets--------
        if args.generalized_offsets:
            print("------GETTING GENERALIZED OFFSETS-----")
            #first we extract all the offset information
            all_SO = []
            rel_SO = []
            for details in sat_data[global_start_t][f"antenna {antnum}"]:
                if len(details["sats_present"]) > 1:  #for now only worry about one-sat pulses
                    continue
                ind_offset = details["individual_offset"] #individual offset

                REL = True
                #verify that none of the channels have an unreliable offset. If it passes, add it to reliable list.
                satIDs = list(details['sats_present'].keys())
                for satID in satIDs:
                    satinfo = details['sats_present'][satIDs[0]]
                    for detection in satinfo:
                        print(detection)
                        if detection[1] != 'RELIABLE':
                            REL = False
                
                all_SO.append(ind_offset)
                if REL:
                    rel_SO.append(ind_offset)
                elif not REL:
                    rel_SO.append(0)  #done to keep indices aligned

            print("ALL Specnum Offsets", all_SO)
            print("\nRELIABLE Specnum Offsets", rel_SO)

            
            # Okay here's the context and explanation.
            # This whole case thing is to dummy-proof the generalized offset generation.
            # Sometimes, you might get timestamps that fall inside multiple different days,
            # meaning the computer did a spectrum number reset and some off the offests will be drastically different.
            # So, this was made to separate out different days which correspond to different offsets.
            # They are attributed to the different pulses. 
            # So, if this is overkill for your case, and you just want ONE offset, and you just threw in timestamps that fall in one day,
            # this most likely does not apply.
            

            diff_all_SO = np.diff(all_SO)  #this is an array!

            print("\nDiff array", diff_all_SO)

            #quick and dirty fix is setting a hard limit. this has worked just fine for the Nov 2023 data, but results may differ
            #better method is certainly some kind of outlier tracking, to do later.

            tolerance = 50000
            split_index = 0
            break_count = 0

            for (i, delta)  in enumerate(diff_all_SO):
                if np.abs(delta) > tolerance:
                    break_count += 1
                    split_index = i

            print("split index", split_index)

            #make sure this doesn't break
            if break_count > 1:
                print("you are either spanning more than two days, your tolerance is too low, or your offsets are waaay too volatile. try again!")
                sys.exit()


            #case 1: nothing special, all just one day.
            if break_count == 0:
                M = get_mode(rel_SO)
                #case 1.1: there are enough reliable offsets:
                if M != "empty":
                    SO = M 

                #case 1.2: there are not enough reliable offsets:
                else:
                    SO = int(stats.mode(all_SO)[0])

                for details in sat_data[global_start_t][f"antenna {antnum}"]:
                    details["generalized_offset"] = SO


            #case 2: there is a split somewhere, multiple days.
            elif break_count == 1:
                all_SO1, all_SO2 = all_SO[:split_index+1], all_SO[split_index+1:]
                print('all offsets before split', all_SO1)
                print('all offsets after split', all_SO2)
                rel_SO1, rel_SO2 = rel_SO[:split_index+1], rel_SO[split_index+1:]
                print('all reliable offsets before split', rel_SO1)
                print('all reliable offsets after split', rel_SO2)
                M1, M2 = get_mode(rel_SO1), get_mode(rel_SO2)
                
                #same subcases again 
                if M1 != "empty":
                    SO1 = M1
                else:
                    SO1 = int(stats.mode(all_SO1)[0])
                print("SO1", SO1)

                if M2 != "empty":
                    SO2 = M2
                else:
                    SO2 = int(stats.mode(all_SO2)[0])
                print("SO2", SO2)

                multiple_sat_counter = 0 

                for i, details in enumerate(sat_data[global_start_t][f"antenna {antnum}"]):
                        print("number of sats", len(details["sats_present"]))

                        # I don't use pulses with multiple sats to find offsets, so 
                        if len(details["sats_present"]) == 1:
                            if i <= split_index + multiple_sat_counter:
                                details["generalized_offset"] = SO1
                            else:
                                details["generalized_offset"] = SO2
                                
                        # I don't use pulses with multiple sats to find offsets, so need to account for them in the json
                        elif len(details["sats_present"]) > 1:
                            multiple_sat_counter += 1
                            current_offset = details["individual_offset"]
                            #if there's an edge case, just assign to the closer offset
                            if abs(current_offset - SO1) > abs(current_offset - SO2):
                                details["generalized_offset"] = SO2
                            else:
                                details["generalized_offset"] = SO1

                        print(multiple_sat_counter)

                #idea: add a check if there are not a lot of data points in all_SO to see if there is ONE value in reliable
                #just try to fix small sample size problems.



    #----Save Pulse Data to Json for each Antenna
    #question: how do we want to configure the json to read off the antenna information?
    #          because as of now, no antenna information encoded directly. 
    #          could add an extra dictionary element which gives the antenna, may be a move.

    json_output = path.join(out_path,f"pulsedata_{global_start_t}_{global_end_t}.json")
    with open(json_output, "w") as file:
        json.dump(sat_data, file, indent=4)
        print(sat_data)

