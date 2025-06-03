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




# filtering process is manual, get list filtered_pulses

#times are in seconds, after global_start_time
# [ [pulse start, pulse end], satID, chan, offset]

# 17218
#info = [[[2110, 2630], 59051, 1837, 103156], 
       # [[6920, 7230], 25338, 1841, 79705], 
        #[[8135, 8680], 59051, 1837, 79731], 
       # [[13600, 13905],33591, 1850, 79705] ]

info =  [[[715, 1110], 28654, 1836, 86568], 
         [[4950, 5270], 59051, 1837, 109993], 
         [[7000, 7145], 28654, 1836, 109993], 
         [[10975, 11440], 59051, 1837, 86568], 
         [[17005, 17530], 59051, 1836, 86568], 
         [[22070, 22425], 25338, 1841, 86568],
         [[23025, 23570], 59051, 1837, 48959], 
         [[27945, 28075], 33591, 1850, 109994]] 


#17219
#info = [[[6920, 7230], 25338, 1841, 79705], 
      #  [[8135, 8680], 59051, 1837, 79731], 
      #  [[12925, 13380], 25338, 1841, 79705], 
       # [[13600, 13905], 33591, 1850, 79705], 
       # [[14150, 14695], 59051, 1837, 79731], 
        #[[17330, 17605], 28654, 1836, 79705]]


#---------------GET OBSERVED PHASES-----------------
#this is either correlated directly or done via saved data


observed_data = []
time_total = time.time()

if args.saved_data == False:
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
            idx_ref_v = idx_ref + idx_correction
            idx2_v = idx2
        else:
            idx2_v = idx2 + np.abs(idx_correction)
            idx_ref_v = idx_ref
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
        ant_ref = bdc.BasebandFileIterator(files_ref, 0, idx_ref_v, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')
        ant2 = bdc.BasebandFileIterator(files_a2, 0, idx2_v, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')

        #--------get visibilities-----
        m_ref = ant_ref.spec_num_start
        m2 = ant2.spec_num_start

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
    



print("Done with everything. Time taken:", time.time() - time_total)

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

#calculate both fit types
coords_trf_OG = ch.fitting_all(observed_data, a2_coords, ch.phase_pred, info, context, method='trf')[0]
coords_lm_OG =  ch.fitting_all(observed_data, a2_coords, ch.phase_pred, info, context, method='lm')[0]

#option to create some fuzz around your guess as a sanity check
if args.coordinate_fuzz:
    fuzzed_coords = ch.make_fuzzed_coords(a2_coords, meters=100)
    fitted_trf_rand = []
    for coords in fuzzed_coords:
        print(coords)
        single_fit = ch.fitting_all(observed_data, coords, ch.phase_pred, info, context, method='trf')[0]
        fitted_trf_rand.append(single_fit)

#-------Debug plots--------

if args.debug:
    fit_types = []
    if (args.fit_type == 'trf') or (args.fit_type == 'both'):
        fit_types.append(('trf', coords_trf_OG))
    if (args.fit_type== 'lm') or (args.fit_type == 'both'):
        fit_types.append(('lm', coords_lm_OG))

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
    print("trf:", coords_trf_OG)
if (args.fit_type == 'lm') or (args.fit_type == 'both'):
    print("lm: ", coords_lm_OG)


if args.coordinate_fuzz:
    print("---fuzzed guesses")
    for coord in range(fitted_trf_rand):
        print(coord)


