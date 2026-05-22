import os
import sys
from sys import path
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
import sat_utils_gpu as sug
import sat_utils as su
import figures as fgs
from scipy.optimize import least_squares
import json
import random
from scripts.xcorr import helper as hp
import importlib
from scipy.interpolate import interp1d

config_path = '/home/thomasb/albatros_analysis/scripts/orbcomm'
config_name = 'config2_corr.json'
satdet_path = '/scratch/thomasb/'
satdet_name = "satdet_data_1753132820_cxlen_0.3_seclen_12180_1762288490.json"
#satdet_name = "/pulsedata_1753133403/pulsedata_1753133403_1757540615.4208999.json"
T_SPECTRA = 4096/250e6
chanlist = np.arange(1834, 1852)
v_acclen = 30000
out_dir = '/scratch/thomasb'

satlist = [28654,25338,33591,57166,59051,44387]
satmap = {}
assert min(satlist) > len(satlist)
for i, sat_ID in enumerate(satlist):
    satmap[i] = sat_ID
    satmap[sat_ID] = i

bline_ants = set({'Antenna 1', 'Antenna 2'})


names, paths, offsets, coords, pulsedata = [], [], [], [], []

#config
with open(f"{config_path}/{config_name}", "r") as f:
    config = json.load(f)
    for i, (ant, details) in enumerate(config["antennas"].items()):
        if details['name'] in bline_ants:
            names.append(details['name'])
            paths.append(details['path'])
            offsets.append(details['clock_offset'])
            coords.append(details['coordinates'])

    global_start_time = config["correlation"]["start_timestamp"]
    global_end_time = config["correlation"]["end_timestamp"]

nants = len(names)
print('antenna:', names)
print('number of antenna:', nants)
print("Visibility Accumulation Length", v_acclen)
print('global start and end times:', global_start_time, global_end_time)

tle_path = outils.get_tle_file(global_start_time, 
                               "/project/rrg-sievers/mohanagr/OCOMM_TLES")
chunk_length = T_SPECTRA * v_acclen

#pulsedata
with open(f'{satdet_path}/{satdet_name}') as f:
    data = json.load(f)
    ants = data[f'{global_start_time}']
    for i, antname_bline in enumerate(names):
        for j, (antname_satdet, details) in enumerate(ants.items()):
            if antname_bline == antname_satdet:
                pulsedata.append(details['pulse_data'])

print('pulsedata', pulsedata)

if len(pulsedata) == 2:
    starts1 = {p['start'] for p in pulsedata[0]}
    starts2 = {p['start'] for p in pulsedata[1]}
    shared_starts = starts1 & starts2 
    shared_pulses = [p for p in pulsedata[0] if p['start'] in shared_starts]
else:
    shared_pulses = pulsedata[0]

#check across baselines that the pulse is detected in both?

my_dir = f't2_{names[0]}_{names[1]}_start_{global_start_time}'
final_out_dir = os.path.join(out_dir, my_dir)
os.makedirs(final_out_dir, exist_ok=True)

for pulse_idx, p in enumerate(shared_pulses):
    print(f'total of {len(shared_pulses)} pulses to do')
    
    print(f"---------STARTING PULSE {pulse_idx}---------")

    #--------times-----
    rel_start_t, rel_end_t = p['start'], p['end']
    t_start, t_end = global_start_time + rel_start_t, global_start_time + rel_end_t
    pulse_dur_secs = rel_end_t - rel_start_t
    pulse_dur_chunks = int(np.ceil((pulse_dur_secs)/(T_SPECTRA * v_acclen)))
    print('relative start, end:', rel_start_t, rel_end_t)
    print('duration secs, chunks:', pulse_dur_secs, pulse_dur_chunks)
    sats = [satmap[int(sat_id)] for sat_id in p['sats_present'].keys()]


    #get visibilities
    vis, channels = sug.get_vis_gpu(t_start,
                                    t_end, 
                                    paths,
                                    offsets,
                                    T_SPECTRA = T_SPECTRA,
                                    v_acclen = v_acclen)
    pol00 = vis[0, 2, :, :]
    
    #get phases
    phased_vis, phase, chan_b_idx = su.get_fringes_phase(pol00, channels)

    #make figure
    fig = fgs.makeplot_fringes_phase(coords, 
                                     [t_start, t_end],
                                     chan_b_idx, 
                                     channels, 
                                     phased_vis, 
                                     phase, 
                                     satmap,
                                     sats,
                                     v_acclen,
                                     T_SPECTRA = T_SPECTRA)

    #save stuff
    plot_path = os.path.join(final_out_dir, f'pulse_{rel_start_t}.jpg')
    fig.savefig(plot_path)
    print(f'saved figure pulse {rel_start_t}')



