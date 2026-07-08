import os
import sys
from os import path
sys.path.insert(0, "/home/thomasb")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
import cupy as cp
import json
import sat_utils as su
import sat_utils_gpu as sug
import figures as fgs
import argparse
import time
from scipy.optimize import minimize
from albatros_analysis.scripts.xcorr import helper as hp
from albatros_analysis.scripts.xcorr import helper_gpu as hpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",)
    args = parser.parse_args()


    #HARD-CODED STUFF
    fit_ant_name = 'Antenna 2'
    nfit_ant_name = 'Antenna 1'
    pulse_rel_start_t = 0
    buffer = 0
    satID = 25338
    channel = 1846
    
    chanlist = np.arange(1834, 1852)
    T_SPECTRA = 4096/250e6
    T_SCAN = 5 
    altitude_cutoff = 5
    satlist = [28654,25338,33591,57166,59051,44387]

    satmap = {}
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i

    chan_small_idx = np.where(chanlist == channel)[0]
    print('satmap', satmap)
    print('chan big idx', channel)
    print('chan small idx', chan_small_idx)
   
   
    #OPEN CONFIG
    dir_parents, coords, ant_names = [], [], []
    with open(args.config_file, "r") as f:
        config = json.load(f)
        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            print(ant, details)
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])
        global_start_t = config["correlation"]["start_timestamp"]
        global_end_t = config["correlation"]["end_timestamp"]
        c_acclen = config['correlation']['coarse_acclen']
    print("\nAntenna Coordinates:", coords)

    array_time =  global_end_t - global_start_t 
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")

    fit_ant_idx, nfit_ant_idx = ant_names.index(fit_ant_name), ant_names.index(nfit_ant_name)
    fit_ant_coords, nfit_ant_coords = coords[fit_ant_idx], coords[nfit_ant_idx]
    fit_ant_path, nfit_ant_path = dir_parents[fit_ant_idx], dir_parents[nfit_ant_idx]

    t1 = pulse_rel_start_t + global_start_t 
    t2 = pulse_rel_start_t + global_start_t + 200 #(just enough buffer. chunk at the start anyways)
   
    print('Fitting ant index, name', fit_ant_idx, fit_ant_name)
    print('Non-fitting and index, name', nfit_ant_idx, nfit_ant_name)

    #get files
    fit_ant_files, fit_ant_idx = butils.get_init_info(t1, t2, fit_ant_path)
    nfit_ant_files, nfit_ant_idx = butils.get_init_info(t1, t2, nfit_ant_path)

    #set channel info
    channels = np.asarray(bdc.get_header(fit_ant_files[0])["channels"],dtype='int64')
    chanstart = np.where(channels == 1834)[0][0]
    chanend = np.where(channels == 1852)[0][0]
    nchans = chanend - chanstart

    #get chunks
    #(convention is (ref, nref), or (fit, nfit))
    p0_fit, p0_nfit, specnumoffset = sug.get_chunk_data([fit_ant_files, nfit_ant_files], 
                                                        [fit_ant_idx, nfit_ant_idx], 
                                                        chanstart, 
                                                        chanend, 
                                                        c_acclen = c_acclen)

    #get initial SNR guess (unfitted coordinates)
    snr_guess = sug.get_snr_from_coords(fit_ant_coords,
                                         nfit_ant_coords,
                                         p0_fit,
                                         p0_nfit,
                                         [t1, t2],
                                         satmap,
                                         satID,
                                         chan_small_idx,
                                         c_acclen = c_acclen)
    
    #define some boundaries:
    guess_lat, guess_lon, guess_alt = fit_ant_coords

    bounds = [
    (guess_lat - 0.001, guess_lat + 0.001), 
    (guess_lon - 0.001, guess_lon + 0.001), 
    (guess_alt - 10,  guess_alt + 10)]

    #fit for coordinates of ant B (TRY DIFFERENT METHODS. NELDER MEAD DOESNT DO BOUNDS)
    fit =  minimize(lambda x: sug.get_snr_from_coords(x, 
                                                      nfit_ant_coords, 
                                                      p0_fit, 
                                                      p0_nfit,
                                                      [t1, t2], 
                                                      satmap, 
                                                      satID, 
                                                      chan_small_idx, 
                                                      T_SPECTRA=T_SPECTRA,
                                                      c_acclen=c_acclen),
                    fit_ant_coords,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'disp': True})

    coords_fitted = fit.x
    snr_fitted = fit.fun
    dist = su.get_bline_dist(coords_fitted, fit_ant_coords)

    print('initial coords', fit_ant_coords)
    print("fitted coords:", coords_fitted)
    print('distance between initial and fit:', dist)
    print("initial snr:", snr_guess)
    print('fitted snr', snr_fitted)

