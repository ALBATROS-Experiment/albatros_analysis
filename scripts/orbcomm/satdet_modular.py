import os
import sys
import time
import gc
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
import sat_utils as su
import sat_utils_gpu as sug
import figures as fgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for debug and pulses"
    )
    args = parser.parse_args()

    #VARBS, INCLUDING HARD-CODED
    T_SPECTRA = 4096/250e6
    T_SCAN = 5 #seconds between each satellite risen scan -- look for sat rise/set every 5 sec.
    altitude_cutoff = 5  #cutoff when looking for satellites
    satlist = [28654,25338,33591,57166,59051,44387]
    out_path = args.output_path
    characteristic_time = int(time.time())

    #OPEN CONFIG
    with open(args.config_file, "r") as f:
        config = json.load(f)
        dir_parents, coords, ant_names = [], [], []

        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            print(ant, details)
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])

        global_start_t = config["correlation"]["start_timestamp"] #global as it is in unix time, reference frame
        global_end_t = config["correlation"]["end_timestamp"]
        c_acclen = config["correlation"]["coarse_acclen"]
    print("\nAntenna Coordinates:", coords)
    print("Coarse Accumulation Length", c_acclen)

    #SETUP
    array_time =  global_end_t - global_start_t
    ra_coords, ra_path = coords[0], dir_parents[0]  #(ra = Reference Ant, nra = Non-Reference Ant)
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    debug_output = os.path.join(out_path, f'satdet_debugplots_{global_start_t}_cxlen_{c_acclen/10e6}_seclen_{array_time}_{characteristic_time}')
    os.makedirs(debug_output, exist_ok=True)

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
    for i, row in enumerate(rsats):
        for sat_ID, satele, sataz in row:
            arr[i,satmap[sat_ID]] = 1

    fig = fgs.make_risen_sats_plot(arr, global_start_t, num_sats_risen, T_SCAN=T_SCAN)
    fig.savefig(path.join(out_path,f"risen_sats_{global_start_t}_{characteristic_time}.jpg"), dpi=300)
    fig.clf()
    plt.close(fig)
    del fig
    print(arr)

    #PASSES
    passes = outils.get_simul_pulses(arr) #beware the function is named pulses
    print(passes)
    print(type(passes))
    npasses = len(passes)
    print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    #ITERATE OVER ANTS
    sat_data = {}    #when saving everything to one json
    sat_data[global_start_t] = {}  
    temp_files = []  #paths of temporary per-antenna json files
    for antnum in range(1,len(dir_parents)):
        print(f"--------------- {ant_names[antnum]}-----------------")

        baseline_pulse_data = []
        nra_path, nra_coords = dir_parents[antnum], coords[antnum]

        debug_ant_path = os.path.join(debug_output, f'{ant_names[antnum]}')
        os.makedirs(debug_ant_path, exist_ok=True)

        #ITERATE OVER PASS
        for pnum, [(pstart, pend), sats_present] in enumerate(passes):
            print(f"---------------starting pulse {pnum}---------")
            print(f"we're at {ant_names[antnum]} right now")
            pstart, pend = pstart*T_SCAN, pend*T_SCAN  #go from T_SCAN indices to times in s

            su.print_memory_usage(note = "start of pulse")
            
            #take a chunk halfway through the pulse:
            #pstart_chunk = pstart + int((pend-pstart)/4)
            
            chunk_interval = 30  # seconds
            chunk_length = int(c_acclen * T_SPECTRA)
            pstarts = list(range(pstart, int(pend - chunk_length), chunk_interval))
            detected = False
            for pstart_chunk in pstarts:

                t1, t2 = global_start_t + pstart_chunk, global_start_t + pend  #get in unix time
                tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES") #use most up-to-date tle file
                print('Pass Start:', pstart + global_start_t)
                print("Pass Duration:", t2-t1, '\n')
                print("Chunk Start:", pstart_chunk)

                # Make sure no problem in files
                try:
                    files_ra, idx_ra = butils.get_init_info(t1, t2, ra_path)
                    files_nra, idx_nra = butils.get_init_info(t1, t2, nra_path)
                except Exception as e:
                    print(e)
                    print(f"WARNING: skipping pass {pstart} to {pend}. MISTAKE IN FILE CHECKER!!")
                    continue

                print('idxs ref ant:', idx_ra)
                print('idxs nonref ant:', idx_nra)

                print("Setting Antenna as BFI Objects", '\n')

                # Set up the number of channels we look through
                channels = np.asarray(bdc.get_header(files_ra[0])["channels"],dtype='int64')
                chanstart = np.where(channels == 1834)[0][0]
                chanend = np.where(channels == 1852)[0][0]
                nchans = chanend - chanstart

                p0_ra, p0_nra, specnum_offset = sug.get_chunk_data([files_ra, files_nra], 
                                                                [idx_ra, idx_nra], 
                                                                chanstart, 
                                                                chanend,
                                                                c_acclen=c_acclen)

                temp_satmap = [] 
                temp_satmap.append("Uncorrected")
                for i, satidx in enumerate(sats_present):
                    temp_satmap.append(satmap[satidx])


                #GET CXCORR DATA
                N = 2*c_acclen
                dN = 100000
                cx_gpu = sug.get_cxcorr_many_sats(p0_ra,
                                                p0_nra, 
                                                tle_path, 
                                                [t1,t2], 
                                                sats_present, #maybe change so I can just throw in temp_satmap?
                                                satmap,
                                                [ra_coords, nra_coords],
                                                dN,
                                                c_acclen = c_acclen)

                #GET SNR
                snr_arr = np.zeros((len(sats_present) + 1, nchans), dtype="float64")  #for each chan for each sat (plus uncorrected)
                for i in range(len(sats_present) + 1):
                    print('CX GPU SHAPE', cx_gpu[i].shape)
                    print('nchans', len(cx_gpu[i]))
                    snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx_gpu[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx_gpu[i]),axis=1))

                #GET DETECTIONS
                cx = []
                for cxcorr in cx_gpu:
                    cx.append(cxcorr.get())

                detected_sats, detected_peaks, detected_snrs, rel_ratios = su.get_detections(cx, snr_arr, temp_satmap)
                    

                #now have all required data.
                print('overall snr array:', snr_arr)
                print('detected sats:', detected_sats)
                print('detected peaks:', detected_peaks)
                print('detection snrs:', detected_snrs)
                print('rel ratios:', rel_ratios)


                if not np.all(detected_sats == 0):
                    print("DETECTION! BREAKING LOOP")
                    pstart = pstart_chunk
                    break

                if np.all(detected_sats == 0):
                    print(f'NO DETECTION, TRY {chunk_interval}s LATER')


            if np.all(detected_sats == 0):
                print('NO DETECTIONS FOR WHOLE PASS')

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

            #SAVE DEBUG FIGURES
            debug_pulse_path = os.path.join(debug_ant_path, f'pulse_{pnum}_start_{pstart}')
            os.makedirs(debug_pulse_path, exist_ok=True)
            for idx, sat in enumerate(temp_satmap):
                cxfig = fgs.make_cxcorr_plot(cx[idx])
                cxfig.savefig(os.path.join(debug_pulse_path, f'cxcorr_{sat}.jpg'))
                cxfig.clf()
                plt.close(cxfig)
                del cxfig
            snrfig = fgs.make_snr_plot(snr_arr, temp_satmap)
            snrfig.savefig(os.path.join(debug_pulse_path, f'SNRs_{pnum}_{pstart}.jpg'))
            snrfig.clf()
            plt.close(snrfig)
            del snrfig
            
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
                        sat_peaks.append([int(chanidx)+1834, int(detected_peaks[chanidx]), int(detected_snrs[chanidx]), int(rel_ratios[chanidx])]) #append the channel and the peak location of that channel
                    pulse_data["sats_present"][satmap[sat_ID]] = sat_peaks # make sure it's serializable with json. numpy array wont work
                baseline_pulse_data.append(pulse_data)

            su.print_memory_usage(note = 'before memory freeing')
            del p0_ra
            del p0_nra
            del cx
            del cx_gpu
            gc.collect()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            su.print_memory_usage(note = 'after memory freeing')

        print(baseline_pulse_data)

        #UPDATE OFFSETS
        con_off = su.get_consensus_offset(baseline_pulse_data)
        for pulse_dict in baseline_pulse_data:
            ind_off = pulse_dict["individual_offset"]
            pulse_dict["diff_to_consensus"] = con_off - ind_off

        #SAVE TO SAT DATA
        ant_data = {}
        ant_data["consensus_offset"] = con_off
        ant_data["pulse_data"] = baseline_pulse_data

        temp_path = os.path.join(out_path, f"temp_{ant_names[antnum]}_{global_start_t}.json")
        with open(temp_path, "w") as f:
            json.dump(ant_data, f, indent=4)
        temp_files.append((ant_names[antnum], temp_path))
        print('saved ant_data to temporary json')

        #free up memory
        del baseline_pulse_data
        del ant_data
        gc.collect()

    #SAVING ALL BLINES TO ONE JSON
    for ant_name, temp_path in temp_files:
        with open(temp_path, "r") as f:
            sat_data[global_start_t][ant_name] = json.load(f)

    final_json = path.join(out_path,f"satdet_data_{global_start_t}_cxlen_{c_acclen/10e6}_seclen_{array_time}_{characteristic_time}.json")
    with open(final_json, "w") as f:
        json.dump(sat_data, f, indent=4)
    
    print(f'saved final json to {out_path}')

    for _, temp_path in temp_files:
        os.remove(temp_path)
    print('deleted temporary jsons')


