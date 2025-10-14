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
        c_acclen = config["correlation"]["coarse_acclen"]
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

    #RISEN SATS
    nrows = int((array_time)/T_SCAN)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, ref_coords, global_start_t, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
    num_sats_risen = [len(x) for x in rsats]
    for i, row in enumerate(rsats):
        for sat_ID, satele, sataz in row:
            arr[i,satmap[sat_ID]] = 1

    #PASSES
    passes = outils.get_simul_pulses(arr)
    print(passes)
    print(type(passes))
    npasses = len(passes)
    print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')


    print("STARTING SPECIFIC PULSE ANALYSIS\n--------------------")
    antenna_name = 'Antenna 2'
    specnumoffset = 115507586
    pulse_idx = 2
    buffer = 0

    nref_idx = ant_names.index(antenna_name)
    print(nref_idx)
    nref_path, nref_coords = dir_parents[nref_idx], coords[nref_idx]
    print('ref path', ref_path)
    print('nref path', nref_path)
    print('ref coords:', ref_coords)
    print('nref coords', nref_coords)

    temp_satmap = ['Uncorrected']
    times, sats_present = passes[pulse_idx]
    for sat in sats_present:
        temp_satmap.append(sat)
    print('temp_satmap', temp_satmap)

    rel_start_t, rel_end_t = 5*times[0], 5*times[1]
    t1, t2 = rel_start_t+global_start_t+buffer, rel_end_t+global_start_t
    print('rel pulse times', rel_start_t, rel_end_t)
    print('pulse times', t1, t2)

    tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    print('tle_path', tle_path)

    try:
        files_ra, idx_ra = butils.get_init_info(t1, t2, ref_path)
        files_nra, idx_nra = butils.get_init_info(t1, t2, nref_path)
    except Exception as e:
        print(e)
        print(f"WARNING: skipping pass. MISTAKE IN FILE CHECKER!!")
        #continue

    print("Setting Antenna as BFI Objects", '\n')

    # Set up the number of channels we look through
    channels = np.asarray(bdc.get_header(files_ra[0])["channels"],dtype='int64')
    chanstart = np.where(channels == 1834)[0][0]
    chanend = np.where(channels == 1852)[0][0]
    nchans = chanend - chanstart

    ra = bdc.BasebandFileIterator(
        files_ra,
        0,
        idx_ra,
        c_acclen,
        None,
        chanstart=chanstart,
        chanend=chanend,
        type="float",
    )
    nra = bdc.BasebandFileIterator(
        files_nra,
        0,
        idx_nra,
        c_acclen,
        None,
        chanstart=chanstart,
        chanend=chanend,
        type="float",
    )


    p0_ra = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
    p0_nra = cp.zeros((c_acclen, nchans), dtype="complex64")
    ra_start = ra.spec_num_start
    nra_start = nra.spec_num_start
    for i, (chunk_ra, chunk_nra) in enumerate(zip(ra, nra)):
        perc_missing_ra = (1 - len(chunk_ra["specnums"]) / c_acclen) * 100
        perc_missing_nra = (1 - len(chunk_nra["specnums"]) / c_acclen) * 100
        print("missing a1", perc_missing_ra, "missing a2", perc_missing_nra)
        if perc_missing_ra > 10 or perc_missing_nra > 10:
            ra_start = ra.spec_num_start
            nra_start = nra.spec_num_start
            continue
        
        bdc.make_continuous_gpu(chunk_ra['pol0'],chunk_ra['specnums']-ra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_ra)
        bdc.make_continuous_gpu(chunk_nra['pol0'],chunk_nra['specnums']-nra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_nra)
        break


    print("STARTING CXCORR\n----------------")
    N = int(2* c_acclen)
    dN = int(10**5)
    v_acclen = 10000
    print('N value', N)
    print('dN value', dN)
    print('buffer of', buffer)
    pulse_output = os.path.join(cxcorr_testing_output, f'nrefant{nref_idx}_pulse_{pulse_idx}')
    os.makedirs(pulse_output, exist_ok=True)
    
    #coarse xcorr
    cx = sug.get_cxcorr_many_sats(p0_ra,
                                 p0_nra, 
                                 tle_path, 
                                 [t1, t2], 
                                 sats_present,
                                 satmap,
                                 [ref_coords, nref_coords],
                                 N,
                                 dN)

    #visibility
    vis, chanlist = sug.get_vis_gpu(t1, 
                                    t2,
                                    [ref_path, nref_path], 
                                    [0, specnumoffset],
                                    v_acclen = v_acclen)
    pol00, pol01, pol10, pol11 = vis[0,2,:,:], vis[0,3,:,:], vis[1,2,:,:], vis[1,3,:,:]
    p_vis, phase, chan_big_idx = su.get_fringes_phase(pol00, chanlist)
    print(chanlist)
    print('pol00 shape', pol00.shape)
    print('phase shape', phase.shape)


    #make and save figures
    visfig = fgs.makeplot_fringes_phase(coords, 
                                        [t1,t2],
                                        chan_big_idx, 
                                        chanlist, 
                                        p_vis, 
                                        phase,
                                        sats_present,
                                        satmap,
                                        v_acclen)
    
    visfig.savefig(os.path.join(pulse_output, f'plot_vis.jpg'))

    for idx, sat in enumerate(temp_satmap):
        cxfig = fgs.make_cxcorr_plot(cx[idx])
        cxfig.savefig(os.path.join(pulse_output, f'plot_{sat}.jpg'))


        