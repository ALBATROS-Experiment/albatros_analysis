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
import figures as fgs
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for debug and pulses"
    )
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
    ra_coords, ra_path = coords[0], dir_parents[0] 
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
    rsats = outils.get_risen_sats(tle_path, ra_coords, global_start_t, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
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


    antenna_name = 'Antenna 7'
    bline_idx = ant_names.index(antenna_name) - 1
    ra_path, nra_path = dir_parents[0], dir_parents[bline_idx]
    ra_coords, nra_coords = coords[0], coords[bline_idx]
    print('paths', ra_path, nra_path)
    print('coordinates:', ra_coords, nra_coords)

    pulse_idx = 6

    temp_satmap = ['Uncorrected']
    times, sats_present = passes[pulse_idx]
    for sat in sats_present:
        temp_satmap.append(sat)
    print('temp_satmap', temp_satmap)

    t1, t2 = (5*times[0])+global_start_t, (5*times[1])+global_start_t
    print('pulse times', t1, t2)

    tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    N = int(2* c_acclen)
    dN = int(10**5)
    print('tle_path', tle_path)
    print('N value', N)
    print('dN value', dN)

    satlist = [28654,25338,33591,57166,59051,44387]
    satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i
    print('satmap', satmap)



    try:
        files_ra, idx_ra = butils.get_init_info(t1, t2, ra_path)
        files_nra, idx_nra = butils.get_init_info(t1, t2, nra_path)
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


    cx = su.get_cxcorr_many_sats(p0_ra,
                                p0_nra, 
                                tle_path, 
                                [t1, t2], 
                                sats_present,
                                satmap,
                                [ra_coords, nra_coords],
                                N,
                                dN)

    pulse_output = os.path.join(cxcorr_testing_output, f'bline_{bline_idx}_pulse_{pulse_idx}')
    os.makedirs(pulse_output, exist_ok=True)

    for idx, sat in enumerate(temp_satmap):
        cxfig = su.make_cxcorr_plot(cx[idx])
        cxfig.savefig(os.path.join(pulse_output, f'plot_{sat}_{time.time()}.jpg'))
        