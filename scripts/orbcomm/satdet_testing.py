import os
import sys
import time
from os import path
sys.path.insert(0, "/home/thomasb")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
import cupy as cp
import json
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import sat_utils as su
import sat_utils_gpu as sug
import figures as fgs
import argparse
from albatros_analysis.scripts.xcorr import helper as hp
from albatros_analysis.scripts.xcorr import helper_gpu as hpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",)
    parser.add_argument(
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for debug and pulses")
    args = parser.parse_args()

    T_SPECTRA = 4096/250e6
    c_acclen = 10**6
    v_acclen = 5000
    T_SCAN = 5 
    altitude_cutoff = 15
    satlist = [28654,25338,33591,57166,59051,44387]
    out_path = args.output_path

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

        global_start_t = config["correlation"]["start_timestamp"]
        global_end_t = config["correlation"]["end_timestamp"]
    print("\nAntenna Coordinates:", coords)
    print("Coarse Accumulation Length", c_acclen)

    #SETUP
    array_time =  global_end_t - global_start_t
    print("array time:", array_time)
    ref_coords, ref_path = coords[0], dir_parents[0] 
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    cxcorr_testing_output = os.path.join(out_path, f'cxcorr_testing_{global_start_t}')
    os.makedirs(cxcorr_testing_output, exist_ok=True)

    satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i
    print(satmap)

    #RISEN SATS
    nrows = int((array_time)/T_SCAN)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, ref_coords, global_start_t, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
    num_sats_risen = [len(x) for x in rsats]
    for i, row in enumerate(rsats):
        for sat_ID, satele, sataz in row:
            arr[i,satmap[sat_ID]] = 1

    #PASSES
    p_5s = outils.get_simul_pulses(arr)
    passes = []
    for p in p_5s:
        passes.append([[p[0][0]*5, p[0][1]*5], p[1]])
    for p in passes:
        print(p)
    print(type(passes))
    npasses = len(passes)
    print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')


    print("STARTING SPECIFIC PULSE ANALYSIS\n--------------------")
    antenna_name = 'Antenna 5'
    specnumoffset = -232874
    pulse_rel_start_t = 25025
    buffer = 0

    nref_idx = ant_names.index(antenna_name)
    print('nref_idx', nref_idx)
    nref_path, nref_coords = dir_parents[nref_idx], coords[nref_idx]
    print('ref path', ref_path)
    print('nref path', nref_path)
    print('ref coords:', ref_coords)
    print('nref coords', nref_coords)

    temp_satmap = ['Uncorrected']
    times, sats_present = next((p for p in passes if p[0][0] == pulse_rel_start_t), None)
    print(times)
    print(sats_present)

    #adding artificially
    sats_present = [0,1]

    for sat in sats_present:
        temp_satmap.append(sat)
    print('temp_satmap', temp_satmap)

    rel_start_t, rel_end_t = times[0], times[1]
    t1, t2 = rel_start_t+global_start_t+buffer, rel_end_t+global_start_t
    print('rel pulse times', rel_start_t, rel_end_t)
    print('pulse times', t1, t2)

    tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    print('tle_path', tle_path)

    #get files using baseband utils
    files_ra, idx_ra = butils.get_init_info(t1, t2, ref_path)
    files_nra, idx_nra = butils.get_init_info(t1, t2, nref_path)

    print('idxs ref ant:', idx_ra)
    print('idxs nonref ant:', idx_nra)
    print('new idx nonref ant:', idx_nra)

    channels = np.asarray(bdc.get_header(files_ra[0])["channels"],dtype='int64')
    chanstart = np.where(channels == 1834)[0][0]
    chanend = np.where(channels == 1852)[0][0]
    nchans = chanend - chanstart

    p0_ref, p0_nref, specnum_offset = sug.get_chunk_data([files_ra, files_nra], 
                                                         [idx_ra, idx_nra], 
                                                         chanstart, 
                                                         chanend, 
                                                         c_acclen = c_acclen)

    print("STARTING CXCORR\n----------------")

    N = int(2* c_acclen)
    print('N value', N)
    dN = int(10**5)
    print('dN value', dN)

    pulse_output = os.path.join(cxcorr_testing_output, f'nrefant{nref_idx}_pulse_{pulse_rel_start_t}')
    os.makedirs(pulse_output, exist_ok=True)
    
    cx = sug.get_cxcorr_many_sats(p0_ref,
                                  p0_nref, 
                                  tle_path, 
                                  [t1, t2], 
                                  sats_present,
                                  satmap,
                                  [ref_coords, nref_coords],
                                  N,
                                  dN)

    print("STARTING VIS\n----------------")
    vis, chanlist = sug.get_vis_gpu(t1,
                                    t2,
                                    [ref_path, nref_path], 
                                    [0, specnumoffset],
                                    v_acclen = v_acclen)
    pol00, pol01, pol10, pol11 = vis[0,2,:,:], vis[0,3,:,:], vis[1,2,:,:], vis[1,3,:,:]
    p_vis, phase, chan_big_idx = su.get_fringes_phase(pol00, chanlist)
    chan_small_idx = np.where(chanlist == chan_big_idx)[0][0]
    print(chanlist)
    print('pol00 shape', pol00.shape)
    print('phase shape', phase.shape)
    print('channel index', chan_big_idx)

    #make and save figures
    sat = 25338

    visfig = fgs.makeplot_fringes_phase([ref_coords, nref_coords], 
                                        [t1,t2],
                                        chan_big_idx, 
                                        chanlist, 
                                        p_vis, 
                                        phase,
                                        satmap,
                                        sats_present,
                                        v_acclen)
    
    for sat in sats_present:
        cxfig = fgs.zoomed_cxcorr_plot(cx[sat], chan_small_idx)
        cxfig.savefig(os.path.join(pulse_output, f'peak_cxcorr_{satmap[sat]}.jpg'))

    visfig.savefig(os.path.join(pulse_output, f'plot_vis.jpg'))
    

    for idx, sat in enumerate(temp_satmap):
        cxfig = fgs.make_cxcorr_plot(cx[idx])
        cxfig.savefig(os.path.join(pulse_output, f'plot_{sat}.jpg'))


        