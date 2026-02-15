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
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for plots and pulses"
    )
    args = parser.parse_args()

    #GET HARD-CODED PARAMS---------------------------------------------------------------------
    T_SPECTRA = 4096/250e6
    T_SCAN = 5 #seconds between each satellite risen scan -- look for sat rise/set every 5 sec.
    altitude_cutoff = 5  #cutoff when looking for satellites
    satlist = [28654,25338,33591,57166,59051,44387]
    out_path = args.output_path
    characteristic_time = int(time.time())

    #OPEN CONFIG-------------------------------------------------------------------------------
    with open(args.config_file, "r") as f:
        config = json.load(f)
        dir_parents, coords, ant_names = [], [], []

        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            print(ant, details)
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])
        batch_start_ts = config["correlation"]["start_timestamp"]
        batch_end_ts = config["correlation"]["end_timestamp"]
        c_acclen = config["correlation"]["coarse_acclen"]

    print("\nAntenna Coordinates:", coords)
    print("Coarse acclen", c_acclen)

    #SETUP---------------------------------------------------------------------------------------
    array_time =  batch_end_ts - batch_start_ts
    coords_ref, path_ref = coords[0], dir_parents[0]  #(ref = Reference Ant, nref = Non-Reference Ant)
    tle_path = outils.get_tle_file(batch_start_ts, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    plot_output = os.path.join(out_path, f'satdet_plots_{batch_start_ts}_{int(c_acclen/1e6)}M_len_{array_time}_{characteristic_time}')
    os.makedirs(plot_output, exist_ok=True)

    satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i

    #GET RISEN SATS------------------------------------------------------------------------------------
    nrows = int((array_time)/T_SCAN)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, coords_ref, batch_start_ts, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
    num_sats_risen = [len(x) for x in rsats]
    for i, row in enumerate(rsats):
        for sat_ID, satele, sataz in row:
            arr[i,satmap[sat_ID]] = 1

    fig = fgs.make_risen_sats_plot(arr, batch_start_ts, num_sats_risen, satlist, T_SCAN=T_SCAN)
    fig.savefig(path.join(out_path,f"risen_sats_{batch_start_ts}_{characteristic_time}.jpg"), dpi=300)
    fig.clf()
    plt.close(fig)
    del fig
    print(arr)

    #PASSES----------------------------------------------------------------------------------------------
    passes = outils.get_simul_pulses(arr) #beware the function is named pulses
    print(passes)
    print(type(passes))
    npasses = len(passes)
    print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    #ITERATE OVER ANTENNA----------------------------------------------------------------------------------
    sat_data = {}    #when saving everything to one json
    sat_data[batch_start_ts] = {}  
    temp_files = []  #paths of temporary per-antenna json files
    for antnum in range(1,len(dir_parents)):
        print(f"--------------- {ant_names[antnum]}-----------------")

        baseline_data = []
        path_nref, coords_nref = dir_parents[antnum], coords[antnum]

        ant_plot_path = os.path.join(plot_output, f'{ant_names[antnum]}')
        os.makedirs(ant_plot_path, exist_ok=True)

        #ITERATE OVER PASSES----------------------------------------------------------------------------------
        for pnum, [(pstart, pend), sats_present] in enumerate(passes):
            print(f"---------------{ant_names[antnum]}, Pulse {pnum}---------")
            pstart, pend = pstart*T_SCAN, pend*T_SCAN  #go from T_SCAN indices to times in s
            print('pstart', pstart)
            print('pend', pend)
            print("pass length:", pend-pstart)
            su.print_memory_usage(note = "start of pulse")
        
            chunk_length_secs = c_acclen * T_SPECTRA
            chunk_times = np.arange(pstart, pend, chunk_length_secs)
            print('chunk times', chunk_times)
            nchunks = len(chunk_times)-1

            #chunk_interval = 30  # seconds
            #pstarts = list(range(pstart, int(pend - chunk_length), chunk_interval))
            #detected = False

            temp_satmap = [] 
            temp_satmap.append("Uncorrected")
            for i, satidx in enumerate(sats_present):
                temp_satmap.append(satmap[satidx])

            #get things in unix time
            pass_start_unix, pass_end_unix = batch_start_ts + pstart, batch_start_ts + pend
            tle_path = outils.get_tle_file(pass_start_unix, "/project/rrg-sievers/mohanagr/OCOMM_TLES") #use most up-to-date tle file
            print('unix pstart:', pass_start_unix)


            #ANTENNA OBJECT STUFF----------------------------------------------------------------------------
            # Make sure no problem in files
            try:
                files_ref, idx_ref = butils.get_init_info(pass_start_unix, pass_end_unix, path_ref)
                files_nref, idx_nref = butils.get_init_info(pass_start_unix, pass_end_unix, path_nref)
            except Exception as e:
                print(e)
                print(f"WARNING: skipping pass {pstart} to {pend}. MISTAKE IN FILE CHECKER!!")
                continue

            print('idxs ref ant:', idx_ref)
            print('idxs nonref ant:', idx_nref)

            # Set up the number of channels we look through
            print("Setting Antenna as BFI Objects", '\n')
            channels = np.asarray(bdc.get_header(files_ref[0])["channels"],dtype='int64')
            chanstart = np.where(channels == 1834)[0][0]
            chanend = np.where(channels == 1852)[0][0]
            nchans = chanend - chanstart


            nchans = chanend - chanstart
            ref = bdc.BasebandFileIterator(
                files_ref,
                0,
                idx_ref,
                c_acclen,
                nchunks,
                chanstart=chanstart,
                chanend=chanend,
                type="float",
            )
            nref = bdc.BasebandFileIterator(
                files_nref,
                0,
                idx_nref,
                c_acclen,
                nchunks,
                chanstart=chanstart,
                chanend=chanend,
                type="float",
            )

            print('ref: acclen', ref.acclen, 'nchunks', ref.nchunks)
            print('nref: acclen', nref.nchunks, 'nchunks', nref.nchunks)
            specnum_offset = ref.spec_num_start - nref.spec_num_start
            pulse_times, pulse_snrs, pulse_specnumoffsets = [], [], []
            detection = False

            for chunkidx, (chunk_ref, chunk_nref) in enumerate(zip(ref, nref)):

                print(f'\n----- Chunk {chunkidx}/{nchunks-1} ------')
                chunk_start_unix = chunk_times[chunkidx] + batch_start_ts
                chunk_end_unix = chunk_times[chunkidx+1] + batch_start_ts
                print('chunk start', chunk_start_unix, 'aka', chunk_times[chunkidx])
                print('chunk end', chunk_end_unix, 'aka', chunk_times[chunkidx+1])

                data_ref = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
                data_nref = cp.zeros((c_acclen, nchans), dtype="complex64")
                ref_specnum_start = ref.spec_num_start
                nref_specnum_start = nref.spec_num_start
                

                print('CHUNK IDX', chunkidx)
                print('NCHUNKS', nchunks)
                if chunkidx == nchunks:
                    print('offbyone error!!')
                    sys.exit()

                perc_missing_ref = (1 - len(chunk_ref["specnums"]) / c_acclen) * 100
                perc_missing_nref = (1 - len(chunk_nref["specnums"]) / c_acclen) * 100
                if perc_missing_ref > 10 or perc_missing_nref > 10:
                    print(f'big problem! plenty of data missing {perc_missing_ref} and {perc_missing_nref}. abort!')
                    sys.exit()

                bdc.make_continuous_gpu(chunk_ref['pol0'],
                                        chunk_ref['specnums']-ref_specnum_start,
                                        np.arange(nchans),
                                        c_acclen,
                                        nchans=nchans, 
                                        out=data_ref)
                
                bdc.make_continuous_gpu(chunk_nref['pol0'],
                                        chunk_nref['specnums']-nref_specnum_start,
                                        np.arange(nchans),
                                        c_acclen,
                                        nchans=nchans, 
                                        out=data_nref)
                
                print('ref: data shape', data_ref.shape, 'percentage missing', perc_missing_ref)
                print('ref: data shape', data_nref.shape, 'percentage missing', perc_missing_nref)

                #GET CXCORR DATA
                cx = []
                dN = 100000
                freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)
                data_nref_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")
                niter = int(chunk_end_unix - chunk_start_unix) + 1

                delays = np.zeros((c_acclen, len(sats_present)))
                for i, satidx in enumerate(sats_present):
                    d = outils.get_sat_delay(  #get delay for whole pulse even though we only interpolate over one chunk.
                        coords_ref, coords_nref, tle_path, chunk_start_unix, niter, satmap[satidx]
                        )
                    delays[:, i] = np.interp(
                        np.arange(0, c_acclen) * T_SPECTRA, np.arange(0, niter), d
                    )
                delays = cp.asarray(delays)
                cx.append(outils_g.coarse_xcorr(data_ref, data_nref, dN))

                for i, satidx in enumerate(sats_present):
                    print("getting cxcorr of:", satmap[satidx])
                    outils_g.apply_delay(data_nref, delays[:,i], freqs, out=data_nref_delayed)
                    cx.append(outils_g.coarse_xcorr(data_ref, data_nref_delayed, dN))

                # cx_gpu = sug.get_cxcorr_many_sats(data_ref,
                #                                 data_nref, 
                #                                 tle_path, 
                #                                 [chunk_start_unix, chunk_end_unix], 
                #                                 sats_present, #maybe change so I can just throw in temp_satmap?
                #                                 satmap,
                #                                 [coords_ref, coords_nref],
                #                                 dN,
                #                                 c_acclen = c_acclen)


                #GET SNR
                snr_arr = np.zeros((len(sats_present) + 1, nchans), dtype="float64")  #for each chan for each sat (plus uncorrected)
                for i in range(len(sats_present) + 1):
                    print('CX GPU SHAPE', cx[i].shape)
                    print('nchans', len(cx[i]))
                    snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx[i]),axis=1))

                #GET DETECTIONS
                cx_cpu = []
                for cxcorr in cx:
                    cx_cpu.append(cxcorr.get())
                detected_sats, detected_peaks, detected_snrs, rel_ratios = su.get_detections(cx_cpu, snr_arr, temp_satmap)
                    

                #now have all required data.
                print('overall snr array:', snr_arr)
                print('detected sats:', detected_sats)
                print('detected peaks:', detected_peaks)
                print('detection snrs:', detected_snrs)
                print('rel ratios:', rel_ratios)

                if np.all(detected_sats == 0) and not detection:
                    #if this chunk sees nothing and we havent made a detection yet, throw away
                    print(f'NO DETECTION YET, THROWING AWAY')
                elif np.all(detected_sats == 0) and detection:
                    #the pulse sees something, but this chunk doesn't. save anyways because might be a gap
                    print('NOT THIS CHUNK, BUT WE SEE STUFF SO SAVING')
                    pulse_snrs.append((0, 0, 0))
                    pulse_times.append([float(chunk_start_unix), float(chunk_end_unix)])
                    pulse_specnumoffsets.append(0)
                else:
                    #this chunk sees something, which is good
                    print("DETECTION!")
                    detection = True
                    det_idx = int(np.argmax(detected_snrs))
                    pulse_times.append([float(chunk_start_unix), float(chunk_end_unix)])
                    pulse_snrs.append((int(np.max(detected_snrs)), det_idx, int(detected_sats[det_idx])))
                    best_guess_offset = np.max(detected_peaks) 
                    if (best_guess_offset-dN) > 0:
                        pulse_specnumoffsets.append(int(specnum_offset + (best_guess_offset-dN)))
                    else:
                        pulse_specnumoffsets.append(int(specnum_offset -  np.abs(best_guess_offset - dN)))
        
            if not detection:
                #case 3: no detections at all, so don't write anything
                print('NO DETECTIONS FOR WHOLE PULSE. SKIPPING')
                continue
            else:
                #cut non-detections from the back
                print(pulse_snrs)
                while pulse_snrs[-1][0] == 0:
                    print('cutting the back:')
                    print(pulse_times[-1])
                    pulse_times = pulse_times[:-1]
                    print(pulse_specnumoffsets[-1])
                    pulse_specnumoffsets = pulse_specnumoffsets[:-1]
                    print(pulse_snrs[-1])
                    pulse_snrs = pulse_snrs[:-1]
                pulse_times_merged = [pulse_times[0][0], pulse_times[-1][1]]
            
            print('merged times', pulse_times_merged)
            print('pulse snrs', pulse_snrs)
            print('specnumoffsets', pulse_specnumoffsets)

            #SPECNUMOFFSET FROM CXCORR MAX
            # if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
            #     peak_guesses = detected_peaks[detected_peaks>0]
            #     print("detected peak locations are", peak_guesses)
            #     best_guess_offset = np.max(detected_peaks)  #use the maximum peak to determine best guess offset for each pulse
            #     print(best_guess_offset)
            #     if (best_guess_offset-dN) > 0:
            #             specnum_offset += (best_guess_offset-dN)
            #     else:
            #         specnum_offset -=  np.abs(best_guess_offset - dN)
            #     print("specnum offset updated to:", specnum_offset)
            # else:
            #     print("No detected peaks for this pulse")


            #SAVE DEBUG FIGURES
            pulse_plot_path = os.path.join(ant_plot_path, f'pulse_{pnum}_start_{pstart}')
            os.makedirs(pulse_plot_path, exist_ok=True)
            for idx, sat in enumerate(temp_satmap):
                cxfig = fgs.make_cxcorr_plot(cx[idx])
                cxfig.savefig(os.path.join(pulse_plot_path, f'cxcorr_{sat}.jpg'))
                cxfig.clf()
                plt.close(cxfig)
                del cxfig
            snrfig = fgs.make_snr_plot(snr_arr, temp_satmap)
            snrfig.savefig(os.path.join(pulse_plot_path, f'SNRs_{pnum}_{pstart}.jpg'))
            snrfig.clf()
            plt.close(snrfig)
            del snrfig
            
            #STORE PULSE DATA
            pulse_data = {}
            # if len(np.where(detected_peaks>0)[0]) > 0: #if we have channels with detections
            #     pulse_data["start"] = pstart
            #     pulse_data["end"] = pend
            #     pulse_data["sats_present"] = {}
            #     pulse_data["individual_offset"] = int(specnum_offset)
            
            #     for i, sat_ID in enumerate(sats_present):
            #         where_sat = np.where(detected_sats == satmap[sat_ID])[0]
            #         if len(where_sat) == 0:
            #             continue
            #         sat_peaks = []
            #         for chanidx in where_sat:
            #             #append the channel and the peak location of that channel
            #             sat_peaks.append([int(chanidx)+1834, int(detected_peaks[chanidx]), int(detected_snrs[chanidx]), int(rel_ratios[chanidx])])
            #         pulse_data["sats_present"][satmap[sat_ID]] = sat_peaks # make sure it's serializable with json. numpy array wont work
            #     baseline_pulse_data.append(pulse_data)

            pulse_data["times"] = pulse_times_merged
            pulse_data["specnumoffsets"] = pulse_specnumoffsets
            pulse_data["SNR, Chan, Sat"] = pulse_snrs
            print(pulse_data)
            baseline_data.append(pulse_data)

            su.print_memory_usage(note = 'before memory freeing')
            del data_ref
            del data_nref
            del cx
            del cx_cpu
            gc.collect()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            su.print_memory_usage(note = 'after memory freeing')

        print(baseline_data)

        #UPDATE OFFSETS
        # con_off = su.get_consensus_offset(baseline_pulse_data)
        # for pulse_dict in baseline_pulse_data:
        #     ind_off = pulse_dict["individual_offset"]
        #     pulse_dict["diff_to_consensus"] = con_off - ind_off

        #SAVE TO SAT DATA
        # ant_data = {}
        # ant_data["consensus_offset"] = con_off
        #ant_data["pulse_data"] = baseline_data

        temp_path = os.path.join(out_path, f"temp_antidx{antnum}_{batch_start_ts}.json")
        with open(temp_path, "w") as f:
            json.dump(baseline_data, f, indent=4)
        temp_files.append((ant_names[antnum], temp_path))
        print('saved ant_data to temporary json')

        #free up memory
        del baseline_data
        gc.collect()

    #SAVING ALL BLINES TO ONE JSON
    for ant_name, temp_path in temp_files:
        with open(temp_path, "r") as f:
            sat_data[batch_start_ts][ant_name] = json.load(f)

    final_json = path.join(out_path,f"satdet_data_{batch_start_ts}_{int(c_acclen/1e6)}M_len_{array_time}_{characteristic_time}.json")
    with open(final_json, "w") as f:
        json.dump(sat_data, f, indent=4)
    
    print(f'saved final json to {out_path}')

    for _, temp_path in temp_files:
        os.remove(temp_path)
    print('deleted temporary jsons')