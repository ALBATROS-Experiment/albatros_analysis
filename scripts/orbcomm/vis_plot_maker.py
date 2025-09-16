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
#import extra_functions as ef
import random
#import h5py
from scripts.xcorr import helper as hp
import importlib
from scipy.interpolate import interp1d

def pred(coord1, coord2, start_t, end_t, pulse_idx, channel, satID):
    '''
    predicted phase given satellite
    '''
    bench_time = time.time()
    chunk_len = 30000 * (4096/250e6)
    tle_path = outils.get_tle_file(start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    pulse_len_s = end_t - start_t
    d = outils.get_sat_delay_new(coord1, coord2, tle_path, start_t, pulse_len_s+1, satID)

    pulse_len_chunks = np.ceil(pulse_len_s / chunk_len)
    pulse_freq = outils.chan2freq(channel, alias=True)

    interp_chunk_times = (np.arange(pulse_len_chunks) * chunk_len)

    #get the delay values for each of these chunks
    delay = np.interp(interp_chunk_times, np.arange(len(d)), d)

    #get the predicted phase at each chunk
    pred = (-delay + delay[0]) * 2 * np.pi * pulse_freq
    print("time taken pred", time.time() - bench_time)

    return pred  




#--------------------hard coded setup------------
config_path = '/home/thomasb/albatros_analysis/scripts/orbcomm'
config_name = 'config2.json'

satdet_path = '/scratch/thomasb/'
satdet_name = "pulsedata_1753200150_1757992159.256902.json"

T_SPECTRA = 4096/250e6

desired_pulse_start_time = 10545   #this is the current (not super thought out) way to identify pulses
bl1_ants = set({'Antenna 1', 'Antenna 2'})
bl2_ants = set({'Antenna 1', 'Antenna 4'})
required_antennas = bl1_ants | bl2_ants


#-------------------unpack information---------------

antennas = []
with open(f"{config_path}/{config_name}", "r") as f:
    config = json.load(f)
    for i, (ant, details) in enumerate(config["antennas"].items()):
        ant_dict = {}
        ant_dict['name'] = details['name']
        ant_dict['coordinates'] = details['coordinates']
        ant_dict['path'] = details['path']

        if i ==0:
            ant_dict['reference'] = 'T'
        else:
            ant_dict['reference'] = 'F'

        if ant_dict['name'] in required_antennas:
            antennas.append(ant_dict)

    global_start_time = config["correlation"]["start_timestamp"]
    global_end_time = config["correlation"]["end_timestamp"]
    v_acclen = config["correlation"]["vis_acclen"]
    
nants = len(antennas)
print(antennas)
print('number of antenna:', nants)
print("Visibility Accumulation Length", v_acclen)
print('global start and end times:', global_start_time, global_end_time)

tle_path = outils.get_tle_file(global_start_time, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
chunk_length = T_SPECTRA * v_acclen

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

#---------extract pulse information---------

desired_pulse_info_all = []
for i, details in enumerate(antennas):
    if details['name'] == 'Antenna 1':
        continue
    pulse_present = False
    for pulse in details['pulse_data']:
        if pulse['start'] == desired_pulse_start_time:
            desired_pulse_info_all.append(pulse)
            pulse_present = True

if pulse_present == False:
    print(f"CAUTION: not all antenna have pulse {desired_pulse_start_time}!!!!")


#----------extract antenna information---------
ant_offsets = []
ant_paths = []
ant_names = []
for i, ant in enumerate(antennas):
    ant_offsets.append(ant['consensus_offset'])
    ant_paths.append(ant['path'])
    ant_names.append(ant['name'])

#set up bline map, find desired baselines in terms of their index
bl_map = []
for i in range(nants):
    for j in range(i+1, nants):
        bl_elements = set({ant_names[i], ant_names[j]})
        bl_map.append(bl_elements)
print('baseline map:', bl_map)

bl1 = [i for i, x in enumerate(bl_map) if x == bl1_ants]
bl2 = [i for i, x in enumerate(bl_map) if x == bl2_ants]
print('bl1, bl2:', bl1, bl2)

#set up some times, channels, and compute the visibilities for all blines
#note that we are taking times and such wrt Ant1-Ant2 bline. 
#this is fine since the pulse times are all the same for each bline

#add some more double check here or something.

rel_start_t, rel_end_t = desired_pulse_info_all[0]['start'] , desired_pulse_info_all[0]['end']
pulse_start_t = rel_start_t + global_start_time
pulse_end_t = rel_end_t + global_start_time
pulse_len_chunks = int(np.ceil((pulse_end_t - pulse_start_t)/chunk_length))

print('relative start, end t:', rel_start_t, rel_end_t)
print('pulse_duration:', rel_end_t-rel_start_t)

print('GETTING INIT INFO --------------------')
idxs, files = hp.get_init_info_all_ant(pulse_start_t, pulse_end_t, ant_offsets, ant_paths)

channels = bdc.get_header(files[0][0])["channels"].astype('int64')
chanstart = np.where(channels == 1834)[0][0] 
chanend = np.where(channels == 1852)[0][0]
print('starting, ending channels:', chanstart, chanend)
chanlist = np.arange(1834, 1852)

print('GETTING VISIBILITIES------------------')
vis, rowcount, obj = hp.get_avg_fast2(idxs,files,v_acclen,pulse_len_chunks, chanstart, chanend)

#vis is shape (nchunks, nbl, npols, ncols)
#where nbl index is our baseline_map index
#want all chunks and all ncols (=channels)
#just have to pick what pol we want to work on.

print('total vis shape:', vis.shape)

#now we have some visibility data, so let's massage it a bit
vis_1, vis_2 = vis[: , bl1, 0, :], vis[: , bl2, 0, :]  #may need to interpolate
vis_2 = vis[: , bl2, 0, :]
print('vis 1, vis 2 shapes:', vis_1.shape, vis_2.shape)
vis_1, vis_2 = np.squeeze(vis_1), np.squeeze(vis_2)
p_vis1, p_vis2 = np.angle(vis_1), np.angle(vis_2)

#here we want to auto-select which channel to use for phase display
amp1, amp2 = np.abs(vis_1), np.abs(vis_2)
mean_amp1, mean_amp2 = [], []
for i in range(18):
    mean_amp1.append(np.mean(amp1[:,i]))
    mean_amp2.append(np.mean(amp2[:,i]))

#have to play a little game with channel indices (s for small, b for big)
chan_s_idx1, chan_s_idx2 = np.where(mean_amp1 == np.max(mean_amp1))[0][0], np.where(mean_amp2 == np.max(mean_amp2))[0][0]
if chan_s_idx1 != chan_s_idx2:
    print('CAUTION!!!! Same pulse is detected at different frequencies at different baselines')
chan_b_idx1, chan_b_idx2 = chanlist[chan_s_idx1], chanlist[chan_s_idx2]

#finally we can unwrap this guy
phase1, phase2 = np.unwrap(p_vis1[:, chan_s_idx1]), np.unwrap(p_vis2[:, chan_s_idx2])  #automate the sat and channel selection




#--------------------------plotting--------------------------------

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ax[0,0].set_ylabel("Chunk Number (~0.5s interval)")
ax[0,0].set_title("Wrapped Phase, All Channels")
im1 = ax[0,0].imshow(p_vis1, aspect='auto', cmap='RdBu', interpolation="none")
fig.colorbar(im1)

ax[0,1].set_ylabel("Phase (Radians)")
ax[0,1].set_title(f"Unwrapped Phase, {chan_b_idx1}")
ax[0,1].plot(phase1)
#ax[0,1].plot(pred_phase_1)

ax[1,0].set_xlabel("Channel Index (~60kHz interval)")
ax[1,0].set_ylabel("Chunk Number (~0.5s interval)")
im2 = ax[1,0].imshow(p_vis2, aspect='auto', cmap='RdBu', interpolation="none")
fig.colorbar(im2)

ax[1,1].set_xlabel("Chunk Number (~0.5s interval)")
ax[1,1].set_ylabel("Phase (Radians)")
ax[1,1].plot(phase2)
#ax[0,1].plot(pred_phase)

fig.savefig('/scratch/thomasb' + f'/2bline_pulse{desired_pulse_start_time}_{global_start_time}.jpg')