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
from scipy.optimize import least_squares
import json
import random
from scripts.xcorr import helper as hp
import importlib
from scipy.interpolate import interp1d

config_path = '/home/thomasb/albatros_analysis/scripts/orbcomm'
config_name = 'config2.json'
satdet_path = '/scratch/thomasb/'
satdet_name = "pulsedata_1753200150_1757992159.256902.json"
#satdet_name = "/pulsedata_1753133403/pulsedata_1753133403_1757540615.4208999.json"
T_SPECTRA = 4096/250e6
chanlist = np.arange(1834, 1852)
out_dir = '/scratch/thomasb'

bline_ants = set({'Antenna 1', 'Antenna 8'})
necessary_ants = bline_ants|{"Antenna 1"}

antennas = []

#config
with open(f"{config_path}/{config_name}", "r") as f:
    config = json.load(f)
    for i, (ant, details) in enumerate(config["antennas"].items()):
        if details['name'] in necessary_ants:
            ant_dict = {}
            ant_dict['name'] = details['name']
            ant_dict['coordinates'] = details['coordinates']
            ant_dict['path'] = details['path']
            if i ==0:
                ant_dict['reference'] = 'T'
            else:
                ant_dict['reference'] = 'F'
            antennas.append(ant_dict)

    global_start_time = config["correlation"]["start_timestamp"]
    global_end_time = config["correlation"]["end_timestamp"]
    v_acclen = config["correlation"]["vis_acclen"]

v_acclen = 5000

nants = len(antennas)
print(antennas)
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
    for i, ant_from_list in enumerate(antennas):
        if ant_from_list['name'] == 'Antenna 1':
            antennas[i]['consensus_offset'] = 0
            continue
        for j, (ant_from_satdet, details) in enumerate(ants.items()):
            if ant_from_list['name'] == ant_from_satdet:
                antennas[i]['consensus_offset'] = details['consensus_offset']
                antennas[i]['pulse_data'] = details['pulse_data']

for term in antennas:
    print(term)

#implement some kind of check so that only the pulses present 
#in the baselines I care about are included, but for now it doesn't matter
info_pulses = antennas[1]['pulse_data']
print(info_pulses)

names, paths, offsets, coords = [], [], [], []
for i in range(nants):
    paths.append(antennas[i]['path'])
    offsets.append(antennas[i]['consensus_offset'])
    names.append(antennas[i]['name'])
    coords.append(antennas[i]['coordinates'])
print('names', names)
print('paths', paths)
print('offsets', offsets)
print('coords', coords)


bl_map = []
for i in range(nants):
    for j in range(i+1, nants):
        bl_elements = set({names[i], names[j]})
        bl_map.append(bl_elements)
print('baseline map:', bl_map)
bl_idx = [i for i, x in enumerate(bl_map) if x == bline_ants]
print('baseline index', bl_idx)

if nants>2:
    coords = coords[1:]
    names = names[1:]

my_dir = f'{names[0]}_{names[1]}_start_{global_start_time}'
final_out_dir = os.path.join(out_dir, my_dir)
os.makedirs(final_out_dir, exist_ok=True)

for pulse_idx, p in enumerate(info_pulses):
    print(f'total of {len(info_pulses)} pulses to do')
    
    print(f"---------STARTING PULSE {pulse_idx}---------")

    #--------times-----
    rel_start_t, rel_end_t = p['start'], p['end']
    t_start, t_end = global_start_time + rel_start_t, global_start_time + rel_end_t
    pulse_dur_secs = rel_end_t - rel_start_t
    pulse_dur_chunks = int(np.ceil((pulse_dur_secs)/(T_SPECTRA * v_acclen)))
    print('relative start, end:', rel_start_t, rel_end_t)
    print('duration secs, chunks:', pulse_dur_secs, pulse_dur_chunks)

    #----get initialized information----
    idxs, files = hp.get_init_info_all_ant(t_start, t_end, offsets, paths)

    #-------set up channels-------
    channels = bdc.get_header(files[0][0])["channels"].astype('int64')
    chanstart = np.where(channels == 1834)[0][0] 
    chanend = np.where(channels == 1852)[0][0]
    nchans=chanend-chanstart

    #--------call get_avg_fast----------

    time_pulse=time.time()

    vis, rowcount, obj = hp.get_avg_fast2(idxs, files, v_acclen, pulse_dur_chunks, chanstart, chanend)

    print(vis.shape)
    #pols, rowcounts, channels = hp.get_avg_fast(path_1, path_2, t_start, t_end, co_tot, v_acclen, pulse_dur_chunks, chanstart=chanstart, chanend=chanend)

    print(f"DONE PULSE {pulse_idx}. TIME:", time.time()-time_pulse) 

    vis = np.squeeze(vis[:,bl_idx,0,:])
    print(vis.shape)
    p_vis = np.angle(vis)

    #------------- auto-selection -----------
    mean_amp = np.mean(np.abs(vis), axis=0)
    chan_s_idx = np.argmax(mean_amp)
    chan_b_idx = chanlist[chan_s_idx]

    phase = np.unwrap(p_vis[:, chan_s_idx]) - p_vis[0, chan_s_idx]
    sats_present = list(p['sats_present'].keys())

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Pulse {rel_start_t} Channel {chan_b_idx}/{chan_s_idx}')

    im = ax[0].imshow(p_vis, aspect='auto', cmap='RdBu', interpolation='none')
    ax[0].set_xlabel("channel idx (~60 kHz interval)")
    ax[0].set_ylabel("chunk number (~0.5 s interval)")
    cbar = fig.colorbar(im, ax=ax[0], orientation='vertical')
    cbar.set_label("phase (radians)")

    ax[1].plot(phase, label='detected phase')
    #ax[1].plot(pred_phase, label='predicted phase')
    ax[1].set_xlabel(f"chunk number (~{np.round(chunk_length, decimals=2)} s interval)")
    ax[1].set_ylabel("phase (radians)")
    for sat in sats_present:
       pred_phase = outils.pred(coords[0], coords[1], t_start, t_end, chan_b_idx, int(sat), v_acclen=v_acclen)[:len(phase)]
       print('MAX PHASE DIFFERENCE:', np.max(np.diff(np.abs(pred_phase))))
       ax[1].plot(pred_phase, label=f'sat {sat}')
    ax[1].legend()

    plot_path = os.path.join(final_out_dir, f'pulse_{rel_start_t}.jpg')

    fig.savefig(plot_path)

    print(f'saved figure pulse {rel_start_t}')



