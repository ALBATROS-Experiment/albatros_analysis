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
import time
import json
import sat_utils as su
import sat_utils_gpu as sug
import figures as fgs
import argparse
from scipy.optimize import minimize
from albatros_analysis.scripts.xcorr import helper as hp
from albatros_analysis.scripts.xcorr import helper_gpu as hpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",)
    args = parser.parse_args()

    print('---------------COMPARING SNRS FOR TWO COORDS-----------')


    #HARD-CODED STUFF
    fit_ant_name = 'Antenna 4'
    nfit_ant_name = 'Antenna 1'

    #two sets of coords. 1 is original, 2 is improved.
    fit_ant_coords_1 = [79.38846667, -91.01926667, 19.777]
    fit_ant_coords_2 = [79.38845797, -91.01936185, 19.777]

    #fit_ant_coords_1 = [79.41721666666666, -90.75885, 176]
    #fit_ant_coords_2 = [79.41721666666666, -90.75885, 176]

    nfit_ant_coords_1 = [79.41718333333333, -90.76735, 189]
    nfit_ant_coords_2 = [79.41717895, -90.76721818, 188.095]

    pulse_rel_start_t = 3900
    satID = 59051
    channel = 1837

    #pulse_rel_start_t = 1005
    #channel = 1846
    #satID = 25338
    
    chanlist = np.arange(1834, 1852)
    T_SPECTRA = 4096/250e6
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
        #print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            #print(ant, details)
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])
        global_start_t = config["correlation"]["start_timestamp"]
        global_end_t = config["correlation"]["end_timestamp"]
        c_acclen = config['correlation']['coarse_acclen']

    array_time =  global_end_t - global_start_t 
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")

    fit_ant_idx, nfit_ant_idx = ant_names.index(fit_ant_name), ant_names.index(nfit_ant_name)
    fit_ant_path, nfit_ant_path = dir_parents[fit_ant_idx], dir_parents[nfit_ant_idx]
    print('fit ant:', fit_ant_idx, fit_ant_path)
    print('nfit ant:', nfit_ant_idx, nfit_ant_path)

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

    #get SNR for bad coordinates
    snr_1 = sug.get_snr_from_coords(fit_ant_coords_1,
                                    nfit_ant_coords_2,
                                    p0_fit,
                                    p0_nfit,
                                    [t1, t2],
                                    satmap,
                                    satID,
                                    chan_small_idx,
                                    c_acclen = c_acclen,
                                    T_SPECTRA = T_SPECTRA)
    
    #get SNR for good coordinates
    snr_2 = sug.get_snr_from_coords(fit_ant_coords_2,
                                    nfit_ant_coords_2,
                                    p0_fit,
                                    p0_nfit,
                                    [t1, t2],
                                    satmap,
                                    satID,
                                    chan_small_idx,
                                    c_acclen = c_acclen,
                                    T_SPECTRA = T_SPECTRA)
    
    #fit using new nfit coord, guess is new fit coord
    bounds = (0.001, 0.001, 10)
    fit_mask = (True, True, False)

    print('-----------STARTING FIT-----------')

    snr_fit, coords_fit = sug.fit_coords_on_snr(fit_ant_coords_2,
                                                nfit_ant_coords_2,
                                                p0_fit,
                                                p0_nfit, 
                                                [t1, t2],
                                                satmap,
                                                satID,
                                                chan_small_idx,
                                                fit_mask = fit_mask, 
                                                bounds = bounds,
                                                T_SPECTRA = T_SPECTRA,
                                                c_acclen = c_acclen)

    snr_arr, lats, lons = sug.get_snr_vs_coords(fit_ant_coords_1,
                                                nfit_ant_coords_2,
                                                p0_fit,
                                                p0_nfit,
                                                0.00005,
                                                5,
                                                [t1, t2],
                                                satmap,
                                                satID,
                                                chan_small_idx,
                                                c_acclen = c_acclen,
                                                T_SPECTRA = T_SPECTRA)
    
    marker1 = (fit_ant_coords_1[0], fit_ant_coords_1[1])
    marker2 = (fit_ant_coords_2[0], fit_ant_coords_2[1])
    marker3 = (coords_fit[0], coords_fit[1])
    #marker4 = (nfit_ant_coords_2[0], nfit_ant_coords_2[1])
    marker_labels = ['old', 'new', 'fitted']

    fig = fgs.snr_vs_coords(snr_arr, 
                            lats, 
                            lons, 
                            title = f'sat {satID}, chan {channel} on {fit_ant_name}-{nfit_ant_name}',
                            out_path = f'/scratch/thomasb/snr_vs_coords_testing_{int(time.time())}.jpg',
                            marker_coords = [marker1, marker2, marker3],
                            marker_labels = marker_labels)
    
    print('SNR old coords', -snr_1)
    print('SNR new coords', -snr_2)
    print('SNR fitted coords', -snr_fit)
    print('\n')
    print('coords old', fit_ant_coords_1)
    print('coords new', fit_ant_coords_2)
    print('coords fit', coords_fit)