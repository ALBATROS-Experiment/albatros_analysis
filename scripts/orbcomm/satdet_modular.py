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
import sat_utils as su

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",
    )

    parser.add_argument(
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for debug and pulses"
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
        ant_names = []

        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            print(ant, details)
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])

        global_start_t = config["correlation"]["start_timestamp"] #global as it is in unix time, reference frame
        global_end_t = config["correlation"]["end_timestamp"]
        c_acclen = config["correlation"]["coarse_acclen"]
        v_acclen = config["correlation"]["vis_acclen"]

    print("\nAntenna Coordinates:", coords)
    print("Coarse Accumulation Length", c_acclen)
    print("Visibility Accumulation Length:", v_acclen, '\n')


    #SETUP (Note that ra = Reference Ant, nra = Non-Reference Ant)
    array_time =  global_end_t - global_start_t
    ra_coords, ra_path = coords[0], dir_parents[0]
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")

    satlist = [28654,25338,33591,57166,59051,44387]
    satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i

    #RISEN SATS
    nrows = int((array_time)/T_SCAN)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, ra_coords, global_start_t, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
    num_sats_risen = [len(x) for x in rsats]

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

    #PASSES
    passes = outils.get_simul_pulses(arr) #beware the function is named pulses
    print(passes)
    print(type(passes))
    npasses = len(passes)
    print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')

    #START LOOPING
    sat_data = {} 
    sat_data[global_start_t] = {}  
    for antnum in range(1,len(dir_parents)):
        print(f"--------------- {ant_names[antnum]}-----------------")

        sat_data[global_start_t][f"{ant_names[antnum]}"] = {}
        baseline_pulse_data = []
        nra_path, nra_coords = dir_parents[antnum], coords[antnum]

        snrplot, axS = plt.subplots(np.ceil(npasses/2).astype(int), 2)
        snrplot.set_size_inches(10, np.ceil(npasses/2)*4)
        snrplot.suptitle(str(global_start_t))
        axS=axS.flatten()

        #--------Iterate over each Pass--------

        for pnum, [(pstart, pend), sats_present] in enumerate(passes):
            print(f"---------------starting pulse {pnum}---------")
            pstart, pend = pstart*T_SCAN, pend*T_SCAN       #go from indices to times
            t1, t2 = global_start_t + pstart, global_start_t + pend  #get in unix time
            tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES") #use most up-to-date tle file
            print("Pass Duration:", t2-t1, '\n')

            # Make sure no problem in files
            try:
                files_ra, idx_ra = butils.get_init_info(t1, t2, ra_path)
                files_nra, idx_nra = butils.get_init_info(t1, t2, nra_path)
            except Exception as e:
                print(e)
                print(f"WARNING: skipping pass {pstart} to {pend}. MISTAKE IN FILE CHECKER!!")
                continue

            print("Setting Antenna as BFI Objects", '\n')

            # Set up the number of channels we look through
            channels = np.asarray(bdc.get_header(files_ra[0])["channels"],dtype='int64')
            chanstart = np.where(channels == 1834)[0][0]
            chanend = np.where(channels == 1852)[0][0]
            nchans = chanend - chanstart

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

            #PICK THE CHUNK, PUT IN DATA
            p0_ra = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
            p0_nra = cp.zeros((c_acclen, nchans), dtype="complex64")
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

            specnum_offset = ra.spec_num_start - nra.spec_num_start #this is the initial delay between specnums when the antennas booted up
            temp_satmap = [] 
            temp_satmap.append("Uncorrected")
            for i, satidx in enumerate(sats_present):
                temp_satmap.append(satmap[satidx])

            #GET CXCORR DATA
            N = 2*c_acclen
            dN = min(100000, int(0.3 * N))
            cx = su.get_cxcorr_many_sats(p0_ra,
                                         p0_nra, 
                                         tle_path, 
                                         [t1,t2], 
                                         sats_present, #maybe change so I can just throw in temp_satmap?
                                         satmap,
                                         [ra_coords, nra_coords],
                                         N,
                                         dN)

            #GET SNR, ADD PULSE SNR TO PLOT
            snr_arr = np.zeros((len(sats_present) + 1, nchans), dtype="float64")  #for each chan for each sat (plus uncorrected)

            axS[pnum].set_title(f"Pulse {pstart} to {pend}.")  #beware: array of plots that is pnum long. 
            for i in range(len(sats_present) + 1):
                snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx[i]),axis=1))
                axS[pnum].plot(snr_arr[i, :], label=f"{temp_satmap[i]}")
            axS[pnum].set_xlabel("Channels")
            axS[pnum].set_ylabel("SNR")
            axS[pnum].legend()

            #DETECTIONS: PASS TO PULSE
            detected_sats, detected_peaks, rel_ratios = su.get_detections(cx, snr_arr, temp_satmap)
            print('detected sats:', detected_sats)
            print('detected peaks:', detected_peaks)
            print('rel ratios:', rel_ratios)

            #SPECNUMOFFSET FROM CXCORR MAX
            if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
                peak_guesses = detected_peaks[detected_peaks>0]
                print("detected peak locations are", peak_guesses)
                best_guess_offset = np.max(detected_peaks)  #use the maximum peak to determine best guess offset for each pulse
                print(best_guess_offset)
                if (best_guess_offset-dN) > 0:
                        specnum_offset += (best_guess_offset-dN)
                else:
                    specnum_offset -=  np.abs(best_guess_offset - dN)
                print("specnum offset updated to:", specnum_offset)
            else:
                print("No detected peaks for this pulse")

            
            #STORE PULSE DATA
            pulse_data = {}
            if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
                pulse_data["start"] = pstart
                pulse_data["end"] = pend
                pulse_data["sats_present"] = {}
                pulse_data["individual_offset"] = int(specnum_offset)
            
                for i, sat_ID in enumerate(sats_present):
                    where_sat = np.where(detected_sats == satmap[sat_ID])[0]
                    if len(where_sat) == 0:
                        continue
                    sat_peaks = []
                    for chanidx in where_sat:
                        sat_peaks.append([int(chanidx)+1834, int(detected_peaks[chanidx]), int(rel_ratios[chanidx])]) #append the channel and the peak location of that channel
                    pulse_data["sats_present"][satmap[sat_ID]] = sat_peaks # make sure it's serializable with json. numpy array wont work
                baseline_pulse_data.append(pulse_data)

        #SAVE SNR DEBUGPLOTS
        snrplot.subplots_adjust(hspace=0.6)
        snrplot.savefig(
                path.join(out_path,f"SNR_ant{antnum}_datastart{global_start_t}.jpg")  #update antenna number?? +1?
            )
        print(baseline_pulse_data)

        #UPDATE OFFSETS
        con_off = su.get_consensus_offset(baseline_pulse_data)
        for pulse_dict in baseline_pulse_data:
            ind_off = pulse_dict["individual_offset"]
            pulse_dict["diff_to_consensus"] = con_off - ind_off

        #SAVE TO SAT DATA
        sat_data[global_start_t][f"{ant_names[antnum]}"]["consensus_offset"] = con_off
        sat_data[global_start_t][f"{ant_names[antnum]}"]["pulse_data"] = baseline_pulse_data
        print('added to sat_data, starting new antenna')

    #SAVE TO JSON
    json_output = path.join(out_path,f"test_pulsedata_{global_start_t}_{time.time()}.json")
    with open(json_output, "w") as file:
        json.dump(sat_data, file, indent=4)
        print(sat_data)

