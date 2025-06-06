import os
import sys
from os import path
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np 
import numba as nb
import time
from scipy import linalg
from scipy import stats
from matplotlib import pyplot as plt
from datetime import datetime as dt
from src.correlations import baseband_data_classes as bdc
from src.utils import baseband_utils as butils
from src.utils import orbcomm_utils as outils
import json
import argparse
from scipy.optimize import least_squares
import coord_helper as ch
import h5py


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Config file containing all required data.",
    )

    parser.add_argument(
        "baseline", type=int, help="Which baseline you want to calibrate with respect to the reference antenna"
    )

    parser.add_argument(
        "-o", "--output_path", type=str, default="/project/s/sievers/thomasb", help="Output directory for debug and pulses"
    )

    parser.add_argument(
        "-dg", "--debug", action="store_true", help="debug option that spits out a ton of plots to see stuff"
    )

    parser.add_argument(
        "-pp", "--pre_plots_only", action='store_true', help="makes the script exit before fitting, so that you only get the pre-fit plots."
    )

    parser.add_argument(
        "-ft", "--fit_type", type= str, default = 'trf', help="lets you decide which type of fit you want. 'trf' gives Trust Region Reflective, 'lm' gives Levenberg-Marquardt, 'both' gives you both."
    )

    parser.add_argument(
        "-cf", "--coordinate_fuzz", action="store_true", help="fits for several plausible initial coordinate guesses to verify fit validity"
    )

    parser.add_argument(
        "-sd", "--saved_data", action='store_true', help='call the flag if you are running from saved visibilities'
    )

    parser.add_argument(
        "-ig", "--improved_guess", action='store_true', help='uses first guess altitude to get a better guess for the overall fit'
    )

    args = parser.parse_args()


out_path = args.output_path
baseline_idx = args.baseline


#----------SETUP FROM CONFIG FILE-------------

with open(f"{args.config_file}", "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    # unpack information from the json file
    # Call get_starting_index for all antennas except reference
    print('\n', "Antenna Details:")
    for i, (ant, details) in enumerate(config["antennas"].items()):
        if (i == 0) or (i ==baseline_idx):
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
    global_start_time = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    c_acclen = config["correlation"]["coarse_acclen"]
    v_acclen = config["correlation"]["vis_acclen"]
    visibility_window = config["correlation"]["visibility_window"]
    T_SPECTRA = config["correlation"]["point_PFB"] / config["correlation"]["sample_rate"]

print("Antenna Paths:", dir_parents, '\n')
print("Antenna Coordinates:", coords, '\n')
print("Visibility Accumulation Length", v_acclen, '\n')
print("Coarse Accumulation Length:", c_acclen, '\n')


C_T_ACCLEN = c_acclen* T_SPECTRA
V_T_ACCLEN = v_acclen* T_SPECTRA

c_nchunks = int((visibility_window)/C_T_ACCLEN)
v_nchunks = int((visibility_window)/V_T_ACCLEN)

tle_path = outils.get_tle_file(global_start_time, "/project/s/sievers/mohanagr/OCOMM_TLES")

ref_coords = coords[0]
ref_path = dir_parents[0]

a2_coords = coords[1]
a2_path = dir_parents[1]

context = [global_start_time, visibility_window, [T_SPECTRA, v_acclen, v_nchunks], ref_coords, tle_path]




#---------------GET OBSERVED DATA-----------------

#objective here is to obtain both the 'info' and the 'observed_data' lists
#this is either correlated directly in this code, or imported from saved data


observed_data = []
time_total = time.time()

if args.saved_data == False:

    with open(f"pulsedata_{global_start_time}_1748962214.json", "r") as f:
        pulsedata = json.load(f)
        
        info = []
        offsets = []
        for pulse_idx, details in enumerate(pulsedata[f"{global_start_time}"]["antenna 1"]):
            #print(f"\n--------pulse idx {pulse_idx}---------")
            start_time = details["start"] * 5
            end_time = details["end"] * 5
            rel = False

            #For now, disregard multiple sat passes. For v2.
            if len(details["sats_present"]) > 1:
                continue

            for satID, values in details["sats_present"].items():
                
                for i in range(len(values)):
                    pulse_info = []
                    #print(values[i])
                    chan = values[i][0]
                    corr_offset = values[i][1]
                    if values[i][2][0] > 0.9:
                        rel = True
                    pulse_info.append([start_time, end_time])
                    pulse_info.append(int(satID))
                    pulse_info.append(chan)
                    pulse_info.append(corr_offset)

                    info.append(pulse_info)

            if rel==True:
                offsets.append(details["timestream_offset"])
                
    print(info)
    print(offsets)
    specnumoffset = int(stats.mode(offsets)[0])
    print(specnumoffset)
    print(len(info))


    for pulse_idx in range(len(info)):

        print(f"---------STARTING PULSE {pulse_idx}---------")

        #--------times-----
        relative_start_time = info[pulse_idx][0][0]
        pulse_duration_chunks = int(  (info[pulse_idx][0][1] - info[pulse_idx][0][0]) / (T_SPECTRA * v_acclen)  )
        t_start = global_start_time + relative_start_time
        t_end = t_start + visibility_window

        #----get initialized information----
        files_ref, idx_ref = butils.get_init_info(t_start, t_end, ref_path)
        files_a2, idx2 = butils.get_init_info(t_start, t_end, a2_path)

        #------get corrected offsets-----
        idx_correction = info[pulse_idx][3] - 100000
        if idx_correction>0:
            idx_ref = idx_ref + idx_correction
        else:
            idx2 = idx2 + np.abs(idx_correction)
        #print("Corrected Starting Indices:", idx1_v, idx2_v)

        #-------set up channels-------
        channels = bdc.get_header(files_ref[0])["channels"].astype('int64')
        chanstart = np.where(channels == 1834)[0][0] 
        chanend = np.where(channels == 1852)[0][0]
        nchans=chanend-chanstart

        chan_bigidx = info[pulse_idx][2] #index in total list , e.g. 1836 (for predicted phases)
        chanmap = channels[chanstart:chanend].astype(int)  #list of all the channels between 1834 and 1852
        chan_smallidx = np.where(chanmap == chan_bigidx)[0][0] #index in small list, e.g. 2 (for data selection)

        #--------open object----------
        ant_ref = bdc.BasebandFileIterator(files_ref, 0, idx_ref, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')
        ant2 = bdc.BasebandFileIterator(files_a2, 0, idx2, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')

        #--------get visibilities-----
        m_ref = ant_ref.spec_num_start
        m2 = m_ref + specnumoffset
        print(specnumoffset)
        print(m_ref - ant2.spec_num_start)

        visibility_phased = np.zeros((v_nchunks,len(ant_ref.channel_idxs)), dtype='complex64')
        time_pulse=time.time()
        #print(f"--------- Processing Pulse Idx {pulse_idx} ---------")
        for i, (chunk1,chunk2) in enumerate(zip(ant_ref,ant2)):
                xcorr = ch.avg_xcorr_4bit_2ant_float(
                    chunk1['pol0'], 
                    chunk2['pol0'],
                    chunk1['specnums'],
                    chunk2['specnums'],
                    m_ref+i*v_acclen,
                    m2+i*v_acclen)
                visibility_phased[i,:] = np.sum(xcorr,axis=0)/v_acclen
                #print("CHUNK", i, " has ", xcorr.shape[0], " rows")
        print(f"DONE PULSE {pulse_idx}. TIME:", time.time()-time_pulse)
        visibility_phased = np.ma.masked_invalid(visibility_phased)
        vis_phase = np.angle(visibility_phased)
        obs = np.unwrap(vis_phase[0:pulse_duration_chunks, chan_smallidx])
        observed_data.append(obs)

        if pulse_idx == 2:
            break

    

    with h5py.File(f'vis_all_{global_start_time}_.h5', 'w') as f:
        for pulse_idx, observed in enumerate(observed_data):
            satID = info[pulse_idx][1]
            chan = info[pulse_idx][2]
            start_time = info[pulse_idx][0][0]
            end_time = info[pulse_idx][0][1]
            
            pulse_array = f.create_dataset(f'{start_time}_{chan}', data=observed)
            pulse_array.attrs['pulse_info'] = [start_time, end_time, satID, chan]
    


if args.saved_data:
    
    observed_data = []
    info = []

    with h5py.File('visibilities.h5', 'r') as f:
        for pulse in f[f'/{global_start_time}']:
            # Access a dataset
            print(pulse)
            p = f[f'/{global_start_time}/{pulse}'].attrs['pulse_info']
            pulse_info = [[int(p[0]), int(p[1])], int(p[2]), int(p[3]), int(p[4])]
            info.append(pulse_info)
            data = f[f'/{global_start_time}/{pulse}'][:]
            observed_data.append(data)
    print(info)

    #possible to-do here is to order them by start time. but might not really matter.
    


print("Done extracting observed visibilities. Time taken:", time.time() - time_total)

if args.debug:
    fig, ax = plt.subplots(int(np.ceil(len(observed_data)/2)), 2)
    fig.set_size_inches(8, 8)
    ax = ax.flatten()
    fig.suptitle(f"Before Fitting")
    for pulse_idx in range(len(observed_data)):
        print(pulse_idx)
        predicted_data = ch.phase_pred(a2_coords, pulse_idx, info, context)
        ax[pulse_idx].set_title(f"Pulse Idx {pulse_idx}")
        ax[pulse_idx].plot(observed_data[pulse_idx])
        ax[pulse_idx].plot(predicted_data)
    plt.tight_layout()
    fig.savefig(path.join(out_path,f"pre_fit_calib_plots_{global_start_time}.jpg"))
    print(path.join(out_path,f"prefit_plot_coordfit_{global_start_time}.jpg"))


if args.pre_plots_only:
    sys.exit()

    
#----------------FITTING---------------
#clean up when I have time, there is a much better way to structure this, I know. There's just quite a few options I want to build into this.

print("Initial Coords:", a2_coords)

#depending on what fit type is called, make certain fits
if (args.fit_type == 'trf') or (args.fit_type == 'both'):
    trf_fit1 = ch.fitting_all(observed_data, a2_coords, ch.phase_pred, info, context, method='trf')[0]
    print("First TRF:", trf_fit1)

    if args.improved_guess:
        improved_guess = []
        improved_guess = [a2_coords[0], a2_coords[1], trf_fit1[2]]
        trf_fit2 = ch.fitting_all(observed_data, improved_guess, ch.phase_pred, info, context, method='trf')[0]
        print("Seconds TRF:", trf_fit2)

    #sanity check, fuzz guess coordinates around inital guess coordinate.
    if args.coordinate_fuzz:
        fuzzed_coords = ch.make_fuzzed_coords(a2_coords, meters=100)
        fitted_trf_rand = []
        for coords in fuzzed_coords:
            print(coords)
            single_fit = ch.fitting_all(observed_data, coords, ch.phase_pred, info, context, method='trf')[0]
            fitted_trf_rand.append(single_fit)


#same thing but for LM
if (args.fit_type == 'lm') or (args.fit_type == 'both'):
    lm_fit1 = ch.fitting_all(observed_data, a2_coords, ch.phase_pred, info, context, method='lm')[0]
    print("First LM:", lm_fit1)

    if args.improved_guess:
        improved_guess = []
        improved_guess = [a2_coords[0], a2_coords[1], lm_fit1[2]]
        lm_fit2 = ch.fitting_all(observed_data, improved_guess, ch.phase_pred, info, context, method='lm')[0]
        print("Second LM:", lm_fit2)

    if args.coordinate_fuzz:
        fuzzed_coords = ch.make_fuzzed_coords(a2_coords, meters=100)
        lm_fuzzed_list = []
        for coords in fuzzed_coords:
            single_fit = ch.fitting_all(observed_data, coords, ch.phase_pred, info, context, method='lm')[0]
            lm_fuzzed_list.append(single_fit)



#-------Debug plots--------

if args.debug:
    fit_types = []
    if (args.fit_type == 'trf') or (args.fit_type == 'both'):
        fit_types.append(('trf', trf_fit1))
    if (args.fit_type== 'lm') or (args.fit_type == 'both'):
        fit_types.append(('lm', lm_fit2))

    for fit_type_tuple in fit_types:

        #combined residuals
        fig, ax = plt.subplots(1,2, sharey=True)
        fig.set_size_inches(12, 4)
        ax = ax.flatten()
        fig.suptitle("Residuals Comparison")  #add start time into title!!
        ax[0].plot(ch.residuals_all(a2_coords, observed_data, ch.phase_pred, info, context))
        ax[0].set_title("Pre-Fit")
        ax[1].plot(ch.residuals_all(fit_type_tuple[1], observed_data, ch.phase_pred, info, context))
        ax[1].set_title("Post-Fit")
        plt.tight_layout()
        fig.savefig(os.path.join(out_path,f"residuals_combined_{fit_type_tuple[0]}_coordfit_{global_start_time}.jpg"))
        print("saved residual plot to:", os.path.join(out_path,f"residuals_combined_{fit_type_tuple[0]}_coordfit_{global_start_time}.jpg"))

        #individual residuals
        fig, ax = plt.subplots(len(observed_data),2, sharey=True)
        fig.set_size_inches(8, 12)
        ax = ax.flatten()
        fig.suptitle(f"Residuals Comparison {global_start_time}")
        for i in range(len(observed_data)):  

            ax[2*i].plot(ch.residuals_individual(a2_coords, observed_data, ch.phase_pred, i, info, context))
            ax[2*i].set_title(f"Pulse {i} Pre-Fit")

            ax[2*i +1].plot(ch.residuals_individual(fit_type_tuple[1], observed_data, ch.phase_pred, i, info, context))
            ax[2*i +1].set_title(f"Pulse {i} Post-Fit")

        fig.tight_layout()
        fig.savefig(os.path.join(out_path,f"residuals_individual_{fit_type_tuple[0]}_coordfit_{global_start_time}.jpg"))
        print("saved residual plot to:", os.path.join(out_path,f"residuals_individual_{fit_type_tuple[0]}_coordfit_{global_start_time}.jpg"))


        fig, ax = plt.subplots(int(np.ceil(len(observed_data)/2)), 2)
        fig.set_size_inches(8, 8)
        ax = ax.flatten()
        fig.suptitle(f"Post-Fit")
        for pulse_idx in range(len(observed_data)):
            print(pulse_idx)
            predicted_data = ch.phase_pred(fit_type_tuple[1], pulse_idx, info, context)
            ax[pulse_idx].set_title(f"Pulse Idx {pulse_idx}")
            ax[pulse_idx].plot(observed_data[pulse_idx])
            ax[pulse_idx].plot(predicted_data)
        plt.tight_layout()
        fig.savefig(os.path.join(out_path,f"post_fit_calib_plots_{fit_type_tuple[0]}_{global_start_time}.jpg"))
        print(os.path.join(out_path,f"post_fit_calib_plots_{fit_type_tuple[0]}_{global_start_time}.jpg"))


print("OG :", a2_coords)
if (args.fit_type == 'trf') or (args.fit_type == 'both'):
    print("trf:", trf_fit1)
    if args.improved_guess:
        print("trf second:", trf_fit2)

if (args.fit_type == 'lm') or (args.fit_type == 'both'):
    print("lm: ", lm_fit1)
    if args.improved_guess:
        print("lm second:", lm_fit2)


if args.coordinate_fuzz:
    print("---fuzzed guesses")
    for coord in range(len(fitted_trf_rand)):
        print(coord)


