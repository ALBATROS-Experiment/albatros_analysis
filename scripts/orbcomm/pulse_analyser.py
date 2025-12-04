import os
import sys
from os import path
sys.path.insert(0, "/home/thomasb")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
import cupy as cp
import json
import sat_utils as su
import sat_utils_gpu as sug
import figures as fgs
import argparse
import time
import matplotlib.pyplot as plt
from albatros_analysis.scripts.xcorr import helper as hp
from albatros_analysis.scripts.xcorr import helper_gpu as hpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",)
    parser.add_argument(
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for debug and pulses")
    args = parser.parse_args()

    out_path = args.output_path
    T_SPECTRA = 4096/250e6
    satlist = [28654,25338,33591,57166,59051,44387]

    satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i
    #print(satmap)

    dir_parents, coords, ant_names, clock_offsets = [], [], [], []
    
    with open(args.config_file, "r") as f:
        config = json.load(f)
        print("\nAntenna Details:")
        for i, (ant, details) in enumerate(config["antennas"].items()):
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])
            clock_offsets.append(details['clock_offset'])
        global_start_t = config["correlation"]["start_timestamp"]
        global_end_t = config["correlation"]["end_timestamp"]
    #print("\nAntenna Coordinates:", coords)

    print("STARTING SPECIFIC PULSE ANALYSIS\n--------------------")
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    
    

    c_acclen = 3*10**6
    v_acclen = 10000
    bline_ants = ['Antenna 1', 'Antenna 2']
    file_save_names = ['MARS1', 'MARS2']
    sats_present = [3]
    res_sat = 57166

    
    pulse_rel_start_t =  16545 #36180
    pulse_rel_end_t = 16810 #36580 
    buffer_start = 0
    buffer_end = 0
    t1 = pulse_rel_start_t + global_start_t + buffer_start
    t2 = pulse_rel_end_t + global_start_t - buffer_end
    
    tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    print('tle_path', tle_path)
    
    temp_satmap = ['Uncorrected']
    for sat in sats_present:
        temp_satmap.append(sat) #sats present and temp_satmap have same order
    print('temp_satmap', temp_satmap)

    ant1_idx = ant_names.index(bline_ants[0])
    ant2_idx = ant_names.index(bline_ants[1])
    print('ant indices:', ant1_idx, ant2_idx)

    ant1_path = dir_parents[ant1_idx]
    ant2_path = dir_parents[ant2_idx]
    print('ant paths:', ant1_path, ant2_path)

    ant1_coords = coords[ant1_idx]
    ant2_coords = coords[ant2_idx]
    print('ant coords:', ant1_coords, ant2_coords)

    ant1_offset = clock_offsets[ant1_idx]
    ant2_offset = clock_offsets[ant2_idx]
    print('clock offsets (wrt MARS1):', ant1_offset, ant2_offset)


    #GET FILE INFO
    ant1_files, ant1_file_idx = butils.get_init_info(t1, t2, ant1_path)
    ant2_files, ant2_file_idx = butils.get_init_info(t1, t2, ant2_path)

    channels = np.asarray(bdc.get_header(ant1_files[0])["channels"],dtype='int64')
    chanstart = np.where(channels == 1834)[0][0]
    chanend = np.where(channels == 1852)[0][0]
    nchans = chanend - chanstart
    print('chanstart, chanend:', chanstart, chanend)

    ant1_chunk, ant2_chunk, specnum_offset = sug.get_chunk_data([ant1_files, ant2_files], 
                                                         [ant1_file_idx, ant2_file_idx], 
                                                         chanstart, 
                                                         chanend, 
                                                         c_acclen = c_acclen)

    #START CROSS CORRELATION
    print("STARTING CXCORR\n----------------")
    
    dN = int(10**5)
    print('dN value', dN)

    pulse_output = os.path.join(out_path, f'p_analysis_{pulse_rel_start_t}_{file_save_names[0]}_{file_save_names[1]}_{int(time.time())}')
    os.makedirs(pulse_output, exist_ok=True)
    
    cx = sug.get_cxcorr_many_sats(ant1_chunk,
                                  ant2_chunk, 
                                  tle_path, 
                                  [t1, t2], 
                                  sats_present,
                                  satmap,
                                  [ant1_coords, ant2_coords],
                                  dN,
                                  c_acclen = c_acclen)

    
    snr_array = np.empty((len(temp_satmap), nchans))
    cx_cpu = []
    for i, cxcorr in enumerate(cx):
        snr_array[i,:] =  sug.get_complex_snr(cxcorr)
        print(sug.get_complex_snr(cxcorr))
        cx_cpu.append(cp.asnumpy(cxcorr))

    detected_sats, detected_peaks, detected_snrs, rel_ratios = su.get_detections(cx_cpu, snr_array, temp_satmap)

    #GET VISIBILITY STUFF
    print("STARTING VIS\n----------------")
    vis, chanlist = sug.get_vis_gpu(t1,
                                    t2,
                                    [ant1_path, ant2_path], 
                                    [ant1_offset, ant2_offset],
                                    v_acclen = v_acclen)
    
    pol00, pol01, pol10, pol11 = vis[0,2,:,:], vis[0,3,:,:], vis[1,2,:,:], vis[1,3,:,:]
    p_vis, phase, chan_big_idx = su.get_fringes_phase(pol00, chanlist)
    chan_small_idx = np.where(chanlist == chan_big_idx)[0][0]
    print(chanlist)
    print('pol00 shape', pol00.shape)
    print('phase shape', phase.shape)
    print('channel index', chan_big_idx)

    visfig = fgs.makeplot_fringes_phase([ant1_coords, ant2_coords], 
                                        [t1,t2],
                                        chan_big_idx, 
                                        chanlist, 
                                        p_vis, 
                                        phase,
                                        satmap,
                                        sats_present,
                                        v_acclen)
    
    res_plot = fgs.plot_phase_residuals(phase,
                                        [ant1_coords, ant2_coords],
                                        [t1, t2],
                                        chan_big_idx,
                                        res_sat,
                                        T_SPECTRA = T_SPECTRA,
                                        v_acclen = v_acclen)
    
    for i, item in enumerate(temp_satmap):
        if item == "Uncorrected":
            continue
        sat = int(satmap[item])
        cxfig = fgs.zoomed_cxcorr_plot(cx[i], chan_small_idx)
        cxfig.savefig(os.path.join(pulse_output, f'zoomed_cx_{satmap[item]}.jpg'), dpi = 300)
        print(-sug.get_snr_from_coords(ant1_coords,
                                      ant2_coords,
                                      ant1_chunk,
                                      ant2_chunk,
                                      [t1, t2],
                                      satmap,
                                      sat,
                                      chan_small_idx,
                                      c_acclen = c_acclen))

    visfig.savefig(os.path.join(pulse_output, f'vis_plot.jpg'), dpi = 300)
    res_plot.savefig(os.path.join(pulse_output, f'residuals_{res_sat}.jpg'), dpi = 300)
    print('vis plot saved to:', os.path.join(pulse_output, f'vis_plot.jpg'))
    print('res plot saved to:', os.path.join(pulse_output, f'residuals_{res_sat}.jpg'))

