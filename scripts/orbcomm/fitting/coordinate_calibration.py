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


#define some basic functions here, like xcorr and etc
#not sure where they are from but can just import these bad boys from their source

@nb.njit()
def get_common_rows(specnum0,specnum1,idxstart0,idxstart1):
    nrows0,nrows1=specnum0.shape[0],specnum1.shape[0]
    maxrows=min(nrows0,nrows1)
    rownums0=np.empty(maxrows,dtype='int64')
    rownums0[:]=-1
    rownums1=rownums0.copy()
    rowidx=rownums0.copy()
    i=0;j=0;row_count=0;
    while i<nrows0 and j<nrows1:
        if (specnum0[i]-idxstart0)==(specnum1[j]-idxstart1):
            rownums0[row_count]=i
            rownums1[row_count]=j
            rowidx[row_count]=specnum0[i]-idxstart0
            i+=1
            j+=1
            row_count+=1
        elif (specnum0[i]-idxstart0)>(specnum1[j]-idxstart1):
            j+=1
        else:
            i+=1
    return row_count,rownums0,rownums1,rowidx

@nb.njit(parallel=True)
def avg_xcorr_4bit_2ant_float(pol0,pol1,specnum0,specnum1,idxstart0,idxstart1,delay=None,freqs=None):
    row_count,rownums0,rownums1,rowidx=get_common_rows(specnum0,specnum1,idxstart0,idxstart1)
    ncols=pol0.shape[1]
#     print("ncols",ncols)
    assert pol0.shape[1]==pol1.shape[1]
    xcorr=np.zeros((row_count,ncols),dtype='complex64') # in the dev_gen_phases branch
    if delay is not None:
        for i in nb.prange(row_count):
            for j in range(ncols):
                xcorr[i,j] = pol0[rownums0[i],j]*np.conj(pol1[rownums1[i],j]*np.exp(2j*np.pi*delay[rowidx[i]]*freqs[j]))
    else:
        for i in nb.prange(row_count):
            xcorr[i,:] = pol0[rownums0[i],:]*np.conj(pol1[rownums1[i],:])
    return xcorr

def get_coarse_xcorr(f1, f2, Npfb=4096):
    if len(f1.shape) == 1:
        f1 = f1.reshape(-1, 1)
    if len(f2.shape) == 1:
        f2 = f2.reshape(-1, 1)
    chans = f1.shape[1]
    Nsmall = f1.shape[0]
    wt = np.zeros(2 * Nsmall)
    wt[:Nsmall] = 1
    n_avg = np.fft.irfft(np.fft.rfft(wt) * np.conj(np.fft.rfft(wt)))
#     print(n_avg)
#     n_avg[Nsmall] = np.nan
#     print(n_avg[Nsmall-10:Nsmall+10])
    n_avg = np.tile(n_avg, chans).reshape(chans, 2*Nsmall)
#     print(n_avg.shape)
    bigf1 = np.vstack([f1, np.zeros(f1.shape, dtype=f1.dtype)])
    bigf2 = np.vstack([f2, np.zeros(f2.shape, dtype=f2.dtype)])
    bigf1 = bigf1.T.copy()
    bigf2 = bigf2.T.copy()
    bigf1f = np.fft.fft(bigf1,axis=1)
    bigf2f = np.fft.fft(bigf2,axis=1)
    xx = bigf1f * np.conj(bigf2f)
    xcorr = np.fft.ifft(xx,axis=1)
    xcorr = xcorr / n_avg
    xcorr[:,Nsmall] = np.nan
    return xcorr

def phase_pred(fit_coords, info, pulse_idx):
    
    ref_coords = coords[0]

    relative_start_time = info[pulse_idx][0][0]
    pulse_duration_sec = info[pulse_idx][0][1] - info[pulse_idx][0][0]
    pulse_duration_chunks = int( pulse_duration_sec / (T_SPECTRA * v_acclen) )


    time_start = global_start_time + relative_start_time
    sat_ID = info[pulse_idx][1]

    pulse_channel_idx = info[pulse_idx][2]
    pulse_freq = outils.chan2freq(pulse_channel_idx, alias=True)

    # 'd' has one entry per second
    d = outils.get_sat_delay(ref_coords, fit_coords, tle_path, time_start, visibility_window+1, sat_ID)
    # 'delay' has one entry per chunk (~0.5s) 
    delay = np.interp(np.arange(0, v_nchunks) * v_acclen * T_SPECTRA, np.arange(0, int(visibility_window)+1), d)
    #thus 'pred' has one entry for each chunk
    pred = (-delay[:pulse_duration_chunks]+ delay[0]) *  2*np.pi * pulse_freq

    return pred



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Config file containing all required data.",
    )

    #the following can also all be added into the config file. Just here as a reminder of stuff to add, and for use before everything gets done. 


    parser.add_argument(
        "-o", "--output_path", type=str, default="/project/s/sievers/thomasb", help="Output directory for debug and pulses"
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="debug option that spits out a ton of plots to see stuff"
    )

    parser.add_argument(
        "--T_spectra", type=float, default= 4096 / 250e6
    )

    parser.add_argument(
        "-v", "--visibility_window", type=int, default= 1000, help="max time we analyse the pulse. in seconds, default 1000"
    )

    parser.add_argument(
        "-p", "--pre_plots_only", action='store_true', help="makes the script exit before fitting, so that you only get the pre-fit plots."
    )

    parser.add_argument(
        "-f", "--fit_type", type='float', default = 'trf', help="lets you decide which type of fit you want. 'trf' gives Trust Region Reflective, 'lm' gives Levenberg-Marquardt, 'both' gives you both."
    )

    args = parser.parse_args()


#------Set Argument Variables-----
T_SPECTRA = args.T_spectra
out_path = args.output_path
visibility_window = args.visibility_window

#----------SETUP FROM CONFIG FILE-------------

with open(f"{args.config_file}", "r") as f:
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
    global_start_time = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    c_acclen = config["correlation"]["coarse_acclen"]
    v_acclen = config["correlation"]["vis_acclen"]

print("Antenna Paths:", dir_parents, '\n')
print("Antenna Coordinates:", coords, '\n')
print("Visibility Accumulation Length", v_acclen, '\n')
print("Coarse Accumulation Length:", c_acclen, '\n')


C_T_ACCLEN = c_acclen* T_SPECTRA
V_T_ACCLEN = v_acclen* T_SPECTRA

c_nchunks = int((visibility_window)/C_T_ACCLEN)
v_nchunks = int((visibility_window)/V_T_ACCLEN)

tle_path = outils.get_tle_file(global_start_time, "/project/s/sievers/mohanagr/OCOMM_TLES")



#right here is where I am going to want to iteralize for each antenna I think.
#------------START ANTENNA INTERATIONS-------------------------------------

a1_coords = coords[0]
a1_path = dir_parents[0]

a2_coords = coords[1]
a2_path = dir_parents[1]

# filtering process, get list filtered_pulses
#times are in seconds, after global_start_time
# [ [pulse start, pulse end], satID, chan, offset]

info = [[[2110, 2630], 59051, 1837, 103156], 
        [[6920, 7230], 25338, 1841, 79705], 
        [[8135, 8680], 59051, 1837, 79731], 
        [[13600, 13905],33591, 1850, 79705] ]


#---------------GET OBSERVED PHASES-----------------
observed_data = []
time_total = time.time()
for pulse_idx in range(len(info)):

    print(f"---------STARTING PULSE {pulse_idx}---------")

    #--------times-----
    relative_start_time = info[pulse_idx][0][0]
    pulse_duration_chunks = int(  (info[pulse_idx][0][1] - info[pulse_idx][0][0]) / (T_SPECTRA * v_acclen)  )
    t_start = global_start_time + relative_start_time
    t_end = t_start + visibility_window

    #----get initialized information----
    files_a1, idx1 = butils.get_init_info(t_start, t_end, a1_path)
    files_a2, idx2 = butils.get_init_info(t_start, t_end, a2_path)

    #------get corrected offsets-----
    idx_correction = info[pulse_idx][3] - 100000
    if idx_correction>0:
        idx1_v = idx1 + idx_correction
        idx2_v = idx2
    else:
        idx2_v = idx2 + np.abs(idx_correction)
        idx1_v = idx1
    #print("Corrected Starting Indices:", idx1_v, idx2_v)

    #-------set up channels-------
    channels = bdc.get_header(files_a1[0])["channels"].astype('int64')
    chanstart = np.where(channels == 1834)[0][0] 
    chanend = np.where(channels == 1852)[0][0]
    nchans=chanend-chanstart

    chan_bigidx = info[pulse_idx][2] #index in total list , e.g. 1836 (for predicted phases)
    chanmap = channels[chanstart:chanend].astype(int)  #list of all the channels between 1834 and 1852
    chan_smallidx = np.where(chanmap == chan_bigidx)[0][0] #index in small list, e.g. 2 (for data selection)

    #--------open object----------
    ant1 = bdc.BasebandFileIterator(files_a1, 0, idx1_v, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')
    ant2 = bdc.BasebandFileIterator(files_a2, 0, idx2_v, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')

    #--------get visibilities-----
    m1=ant1.spec_num_start
    m2=ant2.spec_num_start

    visibility_phased = np.zeros((v_nchunks,len(ant1.channel_idxs)), dtype='complex64')
    time_pulse=time.time()
    #print(f"--------- Processing Pulse Idx {pulse_idx} ---------")
    for i, (chunk1,chunk2) in enumerate(zip(ant1,ant2)):
            xcorr = avg_xcorr_4bit_2ant_float(
                chunk1['pol0'], 
                chunk2['pol0'],
                chunk1['specnums'],
                chunk2['specnums'],
                m1+i*v_acclen,
                m2+i*v_acclen)
            visibility_phased[i,:] = np.sum(xcorr,axis=0)/v_acclen
            #print("CHUNK", i, " has ", xcorr.shape[0], " rows")
    print(f"DONE PULSE {pulse_idx}. TIME:", time.time()-time_pulse)
    visibility_phased = np.ma.masked_invalid(visibility_phased)
    vis_phase = np.angle(visibility_phased)
    obs = np.unwrap(vis_phase[0:pulse_duration_chunks, chan_smallidx])
    observed_data.append(obs)
    
print("Done with everything. Time taken:", time.time() - time_total)

if args.debug:
    fig, ax = plt.subplots(int(np.ceil(len(observed_data)/2)), 2)
    fig.set_size_inches(8, 8)
    ax = ax.flatten()
    fig.suptitle(f"Before Fitting")
    for pulse_idx in range(len(observed_data)):
        print(pulse_idx)
        predicted_data = phase_pred(a2_coords, info, pulse_idx)
        ax[pulse_idx].set_title(f"Pulse Idx {pulse_idx}")
        ax[pulse_idx].plot(observed_data[pulse_idx])
        ax[pulse_idx].plot(predicted_data)
    plt.tight_layout()
    fig.savefig(path.join(out_path,f"pre_fit_calib_plots_{global_start_time}.jpg"))
    print(path.join(out_path,f"prefit_plot_coordfit_{global_start_time}.jpg"))


if args.pre_plots_only:
    sys.exit()

    
#----------------FIT IT---------------

def residuals(coords, observed_data, phase_pred, info):
    
    #gets all residuals in one long array
    residuals_all = []

    for pulse_idx, observed in enumerate(observed_data):
        
        predicted = phase_pred(coords, info, pulse_idx)  
        res = observed - predicted
        residuals_all.append(res.flatten())  # Flatten for least squares fitting

    return np.concatenate(residuals_all)

def residuals_individual(coords, observed_data, phase_pred, info, pulse_idx):
    predicted = phase_pred(coords, info, pulse_idx)
    res = observed_data[pulse_idx] - predicted
    
    return res


def fitting_trf(observed_data, initial_coordinates, phase_pred, info):
    result = least_squares(
        lambda coords: residuals(coords, observed_data, phase_pred, info),  # Pass a lambda that calls residuals
        initial_coordinates,
        method = 'trf'
    )
    optimized_coordinates = result.x
    return optimized_coordinates, result

def fitting_lm(observed_data, initial_coordinates, phase_pred, info):
    result = least_squares(
        lambda coords: residuals(coords, observed_data, phase_pred, info),  # Pass a lambda that calls residuals
        initial_coordinates,
        method = 'lm'
    )
    optimized_coordinates = result.x
    return optimized_coordinates, result


#calculate both anyways.
fitted_coords_trf = fitting_trf(observed_data, a2_coords, phase_pred, info)[0]
fitted_coords_lm = fitting_lm(observed_data, a2_coords, phase_pred, info)[0]


#-------Debug plots--------

if args.debug:
    fit_types = []
    if (argument == 'trf') or (argument == 'both'):
        fit_types.append(('tlm', fitted_coords_trf))
    if (argument == 'lm') or (argument == 'both'):
        fit_types.append(('lm', fitted_coords_lm))


    print(fit_types)
    for fit_type_tuple in fit_types:

        print(fit_type_tuple)
        print(fit_type_tuple[0])

        #combined residuals
        fig, ax = plt.subplots(1,2, sharey=True)
        fig.set_size_inches(12, 4)
        ax = ax.flatten()
        fig.suptitle("Residuals Comparison")  #add start time into title!!
        ax[0].plot(residuals_all(a2_coords, observed_data, phase_pred, info))
        ax[0].set_title("Pre-Fit")
        ax[1].plot(residuals_all(fit_type_tuple[1], observed_data, phase_pred, info))
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

            ax[2*i].plot(residuals_individual(a2_coords, observed_data, phase_pred, info, i))
            ax[2*i].set_title(f"Pulse {i} Pre-Fit")

            ax[2*i +1].plot(residuals_individual(fit_type_tuple[1], observed_data, phase_pred, info, i))
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
            predicted_data = phase_pred(fit_type_tuple[1], info, pulse_idx)
            ax[pulse_idx].set_title(f"Pulse Idx {pulse_idx}")
            ax[pulse_idx].plot(observed_data[pulse_idx])
            ax[pulse_idx].plot(predicted_data)
        plt.tight_layout()
        fig.savefig(os.path.join(out_path,f"post_fit_calib_plots_{fit_type_tuple[0]}_{global_start_time}.jpg"))
        print(os.path.join(out_path,f"post_fit_calib_plots_{fit_type_tuple[0]}_{global_start_time}.jpg"))


print("OG :", a2_coords)
print("trf:", fitted_coords_trf)
print("lm: ", fitted_coords_lm)

