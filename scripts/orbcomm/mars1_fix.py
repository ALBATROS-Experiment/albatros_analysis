import os
import sys
from os import path
sys.path.insert(0, "/home/thomasb")
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import importlib
import argparse
import subprocess
#from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import sat_utils as su
from albatros_analysis.scripts.xcorr import helper as hxc
importlib.reload(su)
importlib.reload(butils)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Config file containing all required data.")
    parser.add_argument('-g', '--tag', default='test')
    #parser.add_argument('-r', '--ref_antenna', type=str, default='MARS2', help='determines the reference antenna')
    #parser.add_argument('-t', "--testing", type=str, default=None , help='name of test that is being run')
    #parser.add_argument('-m', "--meteors_only", action='store_false', help='makes it so we only look at russian satellites')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    ant_names = ['MARS1']
    ant_coords = [[79.41716147, -90.76723869, 187.9577]]
    ant_paths = ['/project/rrg-sievers/albatros/mars/202507/mars1']
    clock_offset = [-1]

    for i, (ant, details) in enumerate(config["antennas"].items()):
        print(ant, details)
        ant_coords.append(details['coordinates'])
        ant_paths.append(details["path"])
        ant_names.append(details["name"])
        clock_offset.append(details['clock_offset'])
    batch_start_ts = config['correlation']['start_timestamp']
    batch_end_ts = config['correlation']['end_timestamp']
    print(batch_start_ts, batch_end_ts)


    path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
    path_pulses = os.path.join(path_batch, 'data/pulses.json')
    with open(path_pulses, 'r') as f:
        list_pulses = json.load(f)
    times_pulses = []
    for p in list_pulses:
        times_pulses.append([p['t_start'], p['t_end']])

    arr, fig = butils.get_present_files(batch_start_ts, batch_end_ts, [ant_paths[0]], T_SCAN = 5)
    runs = butils.get_simul_files(arr, batch_start_ts, 5, [0])

    for r in runs:
        rstart, rend = r[0], r[1]
        print(rstart, rend)

        #check there are no corrupted/bad files. cut from the back as needed.
        cuts = 0
        error = butils.check_data_holes(rstart, rend, ant_paths[0], min_filesize=500001224, tol = 60, verbose=False, force_ts = False)
        while error:
            print('CUTTING')
            rend = rend-60
            cuts +=1
            error = butils.check_data_holes(rstart, rend, ant_paths[0], min_filesize=500001224, tol = 60, verbose=False, force_ts = False)
        print(f'had to cut {cuts} times')

        #check that there are no unexpected specnum overflows within our region of non-corrupted files
        overflows = hxc.get_overflow_files(rstart, rend, ant_paths[0])
        if overflows.size>0:
            print('OVERFLOW!! NOO!!')
            # make a break?
        else:
            print('No overflows')

        #check that there is a satellite pass within the run. otherwise it's useless, and we continue.
        contained_pass = []
        tol = 100
        for start, end in times_pulses:
            inside = start > rstart and end < rend
            overlap_end = end > rend and start < rend - tol
            overlap_start = start < rstart and end > rstart + tol

            if overlap_start or overlap_end or inside:
                contained_pass.append([start, end])

        print(contained_pass)
        if len(contained_pass) ==0:
            print('NO OVERLAPS: MUST CONTINUE TO NEXT RUN')
            continue

        #if survived so far, now have a valid run.
        print('RUN HAS SURVIVED.')
        print('RUN START', rstart)
        print('RUN END', rend)
        print('RUN LENGTH', (rend-rstart)/60, 'min')

        #make the config file (use all antenna.)
        file = {}
        antennas = {}
        nant = len(ant_names)
        for j in range(nant):
            antennas[ant_names[j]] = {
                "name": ant_names[j],
                "path": ant_paths[j],
                "coordinates": list(ant_coords[j]),
                'clock_offset': clock_offset[j]
                }
        file['antennas'] = antennas
        file["correlation"] = {
            "start_timestamp": rstart,
            "end_timestamp": rend,
            "vis_acclen": 30000,
            "coarse_acclen": 3000000,
            "osamp": 64,
            "pfb_size": 65536,
            "new_acclen": 1024,
            "filt_thresh": 0.4,
            "tag": args.tag
        }
        file["frequency"] = {"start_channel": 1834, "end_channel": 1852} #usual satellite channels
        path_runs = f'/scratch/thomasb/batch_{batch_start_ts}_m1'
        os.makedirs(path_runs, exist_ok=True)
        path_run = os.path.join(path_runs, f'{rstart}')
        os.makedirs(path_run, exist_ok=True)
        config_path = os.path.join(path_run, 'config.json')
        # dump the config file
        with open(config_path, 'w') as f:
            json.dump(file, f, indent=4)

        print('config path is', config_path)
        satdet_path = f'/scratch/thomasb/batch_{batch_start_ts}_m1/{rstart}/satdet'
        print('satdet path is', satdet_path)

        subprocess.run([
            "python",
            "get_satdet.py",
            config_path,
            "-r", "MARS1",
            "-t", f"{batch_start_ts}_m1/{rstart}"
        ])

        #sys.exit()
        #open the satdet solution
        with open(os.path.join(satdet_path, 'satdet_3M_refMARS1.json'), 'r') as f:
            satdet = json.load(f)

        run_offsets = []
        for name, vals in satdet.items():
            if name =='summary':
                continue
            print(name)
            try:
                offs = vals[0]['specnumoffsets']
                print(offs)
                offs = [x for x in offs if x != 0]
            except IndexError:
                offs = 0
            print(offs)
            med = int(np.median(offs))
            print(med)
            run_offsets.append(med)

        m1_offset = -run_offsets[0]
        run_offsets_corr = np.array(run_offsets, dtype=int) - run_offsets[0]

        #check that the other make sense:
        for i in range(len(run_offsets)):
            print(f'\n{ant_names[i+1]}')
            print('from run (corrected)', run_offsets_corr[i])
            print('from satdet', clock_offset[i+1])
            print('diff', run_offsets_corr[i] - clock_offset[i+1])

        file['antennas']['MARS1']['clock_offset'] = m1_offset
        with open(config_path, 'w') as f:
            json.dump(file, f, indent=4)

        print('dumped new config!')