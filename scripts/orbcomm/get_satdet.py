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
from albatros_analysis.scripts.xcorr import helper as hxc
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
        "config_file", type=str, help="Config file containing all required data.")
    parser.add_argument(
        '-r', '--ref_antenna', type=str, default='MARS2', help='determines the reference antenna'
    )
    #parser.add_argument(
    #    "-a",'--antenna',type=int,nargs='+',default=-1,help='all antenna indices you want to include, measured from config file')
    parser.add_argument(
        '-m', "--meteors_only", action='store_false', help='makes it so we only look at russian satellites')
    args = parser.parse_args()

    #GET HARD-CODED PARAMS---------------------------------------------------------------------
    T_SPECTRA = 4096/250e6
    T_SCAN = 5 #seconds between each satellite risen scan -- look for sat rise/set every 5 sec.
    altitude_cutoff = 5  #cutoff when looking for satellites
    if args.meteors_only:
        satlist = [57166,59051]
    else:
        satlist = [28654,25338,33591,57166,59051,44387]

    #OPEN CONFIG-------------------------------------------------------------------------------
    with open(args.config_file, "r") as f:
        config = json.load(f)
        dir_parents, coords, ant_names = [], [], []

        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            #if args.antenna not in (-1, [-1]):
            #    if i not in args.antenna:
            #        continue
            print(ant, details)
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])
        batch_start_ts = config["correlation"]["start_timestamp"]
        batch_end_ts = config["correlation"]["end_timestamp"]
        c_acclen = config["correlation"]["coarse_acclen"]

    print("\nAntenna Names:", ant_names)
    print("\nAntenna Coordinates:", coords)
    print("Coarse acclen", c_acclen)

    #SETUP---------------------------------------------------------------------------------------
    array_time =  batch_end_ts - batch_start_ts
    #reference setup
    ref_antnum = ant_names.index(args.ref_antenna)
    coords_ref, path_ref = coords[ref_antnum], dir_parents[ref_antnum]  #(ref = Reference Ant, nref = Non-Reference Ant)
    tle_path = outils.get_tle_file(batch_start_ts, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    path_batch = os.path.join('/scratch/thomasb', f'batch_{batch_start_ts}')
    os.makedirs(path_batch, exist_ok=True)
    path_satdet = os.path.join(path_batch, 'satdet')
    os.makedirs(path_satdet, exist_ok = True)
    path_debugplots = os.path.join(path_satdet, 'debugplots')
    os.makedirs(path_debugplots, exist_ok=True)

    satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i
    print('Satmap', satmap)

    #GET RISEN SATS------------------------------------------------------------------------------------
    nrows = int((array_time)/T_SCAN)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, coords_ref, batch_start_ts, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
    num_sats_risen = [len(x) for x in rsats]
    for i, row in enumerate(rsats):
        for sat_ID, satele, sataz in row:
            arr[i,satmap[sat_ID]] = 1

    #plot risen sats
    fig_risen_sats, ax = plt.subplots(1, 2)
    fig_risen_sats.set_size_inches(10,4)
    fig_risen_sats.suptitle(f"Risen sats batch {batch_start_ts}")
    ax[0].plot(num_sats_risen)
    ax[0].set_xlabel(f"Time ({T_SCAN} s)")
    ax[1].set_ylabel(f"Time ({T_SCAN} s)")
    ax[1].set_xlabel("Sat ID")
    ax[1].imshow(arr,aspect='auto',interpolation="none")
    ax[1].set_xticks(range(len(satlist)))
    ax[1].set_xticklabels(satlist)
    plt.tight_layout()
    fig_risen_sats.savefig(path.join(path_debugplots,'risen_sats.png'))
    plt.close(fig_risen_sats)

    #PASSES----------------------------------------------------------------------------------------------
    passes = outils.get_simul_pulses(arr) #beware the function is named pulses
    npasses = len(passes)
    #print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    #ITERATE OVER ANTENNA----------------------------------------------------------------------------------
    sat_data = {}    #when saving everything to one json
    sat_data['summary'] = {}
    temp_files = []  #paths of temporary per-antenna json files
    #get the files where the ref antenna has specnum overflows
    overflow_files_ref = hxc.get_overflow_files(batch_start_ts, batch_end_ts, dir_parents[ref_antnum])

    for antnum in range(len(dir_parents)):
        print(f"\n--------------- {ant_names[antnum]}-----------------")
        #skip the reference antenna
        if antnum == ref_antnum:
            print("You don't play on the pres")
            continue

        
        #make paths for the debug plots. add antenna to temp_files even if already computed so it's saved later
        temp_path = os.path.join(path_satdet, f"temp_antidx{antnum}_{batch_start_ts}.json")
        temp_files.append((ant_names[antnum], temp_path))
        #if this data has already been computed, and there is a temp file in the satdet directory, skip this ant
        if os.path.isfile(os.path.join(path_satdet, f"temp_antidx{antnum}_{batch_start_ts}.json")):
            print("Already have this data computed, continue!")
            continue
        
        baseline_data = []
        path_nref, coords_nref = dir_parents[antnum], coords[antnum]

        path_ant = os.path.join(path_debugplots, f'{ant_names[antnum]}')
        os.makedirs(path_ant, exist_ok=True)
        #get files where non-ref antenna has specnum overflows
        #check how to solve for spotty ant 1
        if antnum == 0:
            overflow_files_nref = np.array([])
        else:
            overflow_files_nref = hxc.get_overflow_files(batch_start_ts, batch_end_ts, dir_parents[antnum])

        #ITERATE OVER PASSES----------------------------------------------------------------------------------
        for pnum, [(pstart, pend), sats_present] in enumerate(passes):

            print(f"\n---------------{ant_names[antnum]}, Pulse {pnum}---------")
            pstart, pend = pstart*T_SCAN, pend*T_SCAN  #go from T_SCAN indices to times in s
            print('pstart', pstart)
            print('pend', pend)
            print("pass length:", pend-pstart)
            #get things in unix time
            pass_start_unix, pass_end_unix = batch_start_ts + pstart, batch_start_ts + pend
            tle_path = outils.get_tle_file(pass_start_unix, "/project/rrg-sievers/mohanagr/OCOMM_TLES") #use most up-to-date tle file
            print('unix pstart:', pass_start_unix)
            #check data presence. assume ref ant always has impeccable data coverage.
            missing = butils.check_data_holes(pass_start_unix, pass_end_unix, dir_parents[antnum])
            if missing:
                print('missing data for pulse, we skip')
                continue
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
            print('nref: acclen', nref.acclen, 'nchunks', nref.nchunks)

            overflow_ct_ref = np.sum(overflow_files_ref<pass_start_unix)
            overflow_ct_nref = np.sum(overflow_files_nref<pass_start_unix)
            print('Overflows ref, nref:', overflow_ct_ref, overflow_ct_nref)
            start_specnum_ref = ref.spec_num_start + overflow_ct_ref*(2**32)
            start_specnum_nref = nref.spec_num_start + overflow_ct_nref*(2**32)
            specnum_offset = start_specnum_ref - start_specnum_nref
            print('Initial Specnum Offset:', specnum_offset)

            pulse_times, pulse_snrs, pulse_specnumoffsets = [], [], []
            detection = False
            chunk_data = {}
            ref_specnum_start = ref.spec_num_start
            nref_specnum_start = nref.spec_num_start
            for chunkidx, (chunk_ref, chunk_nref) in enumerate(zip(ref, nref)):
                print(f'\n----- Chunk {chunkidx}/{nchunks-1} ------')
                chunk_start_unix = chunk_times[chunkidx] + batch_start_ts
                chunk_end_unix = chunk_times[chunkidx+1] + batch_start_ts
                print('chunk start', chunk_start_unix, 'aka', chunk_times[chunkidx])
                print('chunk end', chunk_end_unix, 'aka', chunk_times[chunkidx+1])

                data_ref = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
                data_nref = cp.zeros((c_acclen, nchans), dtype="complex64")

                print('CHUNK IDX', chunkidx)
                print('NCHUNKS', nchunks)
                if chunkidx == nchunks:
                    print('offbyone error!!')
                    sys.exit()

                perc_missing_ref = (1 - len(chunk_ref["specnums"]) / c_acclen) * 100
                perc_missing_nref = (1 - len(chunk_nref["specnums"]) / c_acclen) * 100
                if perc_missing_ref > 10 or perc_missing_nref > 10:
                    print(f'big problem! plenty of data missing {perc_missing_ref} and {perc_missing_nref}. abort!')
                    if chunkidx == 0: #logic here is that maybe after a reboot the data is corrupted or something: will let chunk 1 free
                        continue
                    sys.exit()  #otherwise we will need to check what's going on. may be a sign of something ystemically wrong in data

                ref_spec_idxs = chunk_ref['specnums']-(ref_specnum_start+c_acclen*chunkidx)
                nref_spec_idxs = chunk_nref['specnums']-(nref_specnum_start+c_acclen*chunkidx)
                print('ref start idx', chunk_ref['specnums'][0])
                print('nref start spec', chunk_nref['specnums'][0])

                bdc.make_continuous_gpu(chunk_ref['pol0'],
                                        ref_spec_idxs,
                                        np.arange(nchans),
                                        c_acclen,
                                        nchans=nchans, 
                                        out=data_ref)
                
                bdc.make_continuous_gpu(chunk_nref['pol0'],
                                        nref_spec_idxs,
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

                    chunk_data[chunkidx] = {}
                    for idx, sat in enumerate(temp_satmap):
                        chunk_data[chunkidx][sat] = cp.asnumpy(cx[idx][det_idx])
                    chunk_data[chunkidx]['snrs'] = snr_arr

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
                
            #MAKE SINGLE DEBUGPLOT
            figt1 = time.time()
            ndetchunks = len(chunk_data)
            fig, ax = plt.subplots(ndetchunks,4,figsize=(20, ndetchunks * 4),squeeze=False)

            for chk, (_, d) in enumerate(chunk_data.items()):
                snr_ax = ax[chk, 3]
                for s, satid in enumerate(temp_satmap):
                    d2 = np.abs(d[satid])
                    corr_ax = ax[chk, s]
                    corr_ax.plot(d2)
                    corr_ax.set_title(f"satID={satid} off={np.argmax(d2)} chk={chk}")
                    snr_ax.plot(d['snrs'][s, :],label=str(satid))
                snr_ax.set_xlabel("Channels")
                snr_ax.set_ylabel("SNR")
                snr_ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(path_ant, f"pulse_{pnum}_start_{pstart}.png"))
            plt.close(fig)
            print("time for plot generation", time.time() - figt1)

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

        #SAVE TO SAT DATA
        # ant_data = {}
        # ant_data["consensus_offset"] = con_off
        #ant_data["pulse_data"] = baseline_data

        with open(temp_path, "w") as f:
            json.dump(baseline_data, f, indent=4)
        
        print('saved ant_data to temporary json')

        #free up memory
        del baseline_data
        gc.collect()

    #SAVING ALL BLINES TO ONE JSON
    for ant_name, temp_path in temp_files:
        with open(temp_path, "r") as f:
            sat_data[ant_name] = json.load(f)

    final_json = path.join(path_satdet,f"satdet_{int(c_acclen/1e6)}M_ref{ant_names[ref_antnum]}.json")
    with open(final_json, "w") as f:
        json.dump(sat_data, f, indent=4)
    
    print(f'saved final json to {path_satdet}')

    sys.exit()
    for _, temp_path in temp_files:
        os.remove(temp_path)
    print('deleted temporary jsons')