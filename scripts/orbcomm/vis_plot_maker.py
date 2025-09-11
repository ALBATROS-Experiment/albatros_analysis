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
from scripts.xcorr import helper
from scipy.optimize import least_squares
import json
#import extra_functions as ef
import random
#import h5py
from scripts.xcorr import helper as hp
import importlib
from scipy.interpolate import interp1d


#---------------define some functions----------------

#CHECK TO SEE IF THIS MAKES SENSE/IS ALLOWED
def interp_rect(vis):

    #interpolates rectangular form
    t = np.arange(len(vis))
    valid = ~vis.mask

    real_interp = interp1d(t[valid], vis[valid].real, kind='linear', fill_value="extrapolate")
    imag_interp = interp1d(t[valid], vis[valid].imag, kind='linear', fill_value="extrapolate")
    
    vis_interp = real_interp(t) + 1j * imag_interp(t) 
    return vis_interp


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


#--------------------set up some paths------------

config_path = '/home/thomasb/albatros_analysis/scripts/orbcomm'
config_name = 'config.json'

satdet_path = '/scratch/thomasb'
satdet_name = "pulsedata_1753133403_1757540615.4208999.json"


#-------------------unpack some information----------

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

        antennas.append(ant_dict)

    global_start_time = config["correlation"]["start_timestamp"]
    global_end_time = config["correlation"]["end_timestamp"]
    v_acclen = config["correlation"]["vis_acclen"]
    T_SPECTRA = 4096/250e6
print(antennas)
print("Visibility Accumulation Length", v_acclen)
print(global_start_time)
print(global_end_time)
print(T_SPECTRA)

V_T_ACCLEN = v_acclen* T_SPECTRA
tle_path = outils.get_tle_file(global_start_time, "/project/rrg-sievers/mohanagr/OCOMM_TLES")

print(len(antennas))


antennas[0]['consensus_offset'] = 0
with open(f'{satdet_path}/{satdet_name}') as f:
    pulsedata = json.load(f)
    ants = pulsedata[f'{global_start_time}']
    for i, (ant, details) in enumerate(ants.items()):
        antennas[i+1]['consensus_offset'] = details['consensus_offset']
        antennas[i+1]['pulse_data'] = details['pulse_data']

print(antennas)

#-------------get some visibilities (all baselines)---------------

#manually pick what pulse I want to look at
desired_pulse_idx = 4
nant = len(antennas)
ant_offsets = []
ant_paths = []
ant_names = []

#extract information
for i, ant in enumerate(antennas):
    ant_offsets.append(ant['consensus_offset'])
    ant_paths.append(ant['path'])
    ant_names.append(ant['name'])

#set up bline map, find desired baselines

bl_map = []
for i in range(nant):
    for j in range(i+1, nant):
        bl_map.append(f'{ant_names[i]}_{ant_names[j]}')
print('baseline map:', bl_map)

desired_bl_1 = 'Antenna 1_Antenna 2'
desired_bl_2 = 'Antenna 2_Antenna 7'
bl1 = [i for i, x in enumerate(bl_map) if x == desired_bl_1]
bl2 = [i for i, x in enumerate(bl_map) if x == desired_bl_2]
print('bl1, bl2:', bl1, bl2)

#set up some times, channels, and compute the visibilities for all blines
#note that we are taking times and such wrt Ant1-Ant2 bline. 
#this is fine since the pulse times are all the same for each bline
desired_pulse_info = antennas[1]['pulse_data'][desired_pulse_idx]
rel_start_t, rel_end_t = desired_pulse_info['start'], desired_pulse_info['end']
pulse_start_t = rel_start_t + global_start_time
pulse_end_t = rel_end_t + global_start_time
pulse_len_chunks = int(np.ceil((pulse_end_t - pulse_start_t)/(T_SPECTRA * v_acclen)))

print('relative start, end t:', rel_start_t, rel_end_t)
print('pulse_duration:', rel_end_t-rel_start_t)

print('GETTING INIT INFO --------------------')
idxs, files = helper.get_init_info_all_ant(pulse_start_t, pulse_end_t, ant_offsets, ant_paths)

channels = bdc.get_header(files[0][0])["channels"].astype('int64')
chanstart = np.where(channels == 1834)[0][0] 
chanend = np.where(channels == 1852)[0][0]
print('starting, ending channels:', chanstart, chanend)

print('GETTING VISIBILITIES------------------')
vis, rowcount, obj = helper.get_avg_fast2(idxs,files,v_acclen,pulse_len_chunks, chanstart, chanend)

#vis is shape (nchunks, nbl, npols, ncols)
#where nbl index is our baseline_map index
#want all chunks and all ncols (=channels)
#just have to pick what pol we want to work on.

print('total vis shape:', vis.shape)
vis_1, vis_2 = vis[: , bl1, 0, :], vis[: , bl2, 0, :]  #may need to interpolate
vis_2 = vis[: , bl2, 0, :]
print('vis 1, vis 2 shapes:', vis_1.shape, vis_2.shape)
vis_1, vis_2 = np.squeeze(vis_1), np.squeeze(vis_2)
p_vis1, p_vis2 = np.angle(vis_1), np.angle(vis_2)
phase1, phase2 = np.unwrap(p_vis1[:, 3]), np.unwrap(p_vis2[:, 3])  #automate the sat and channel selection

chanlist = np.arange(1834, 1852)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ax[0,0].set_ylabel("Chunk Number (~0.5s interval)")
ax[0,0].set_title("Wrapped Phase, All Channels")
im1 = ax[0,0].imshow(p_vis1, aspect='auto', cmap='RdBu', interpolation="none")
fig.colorbar(im1)

ax[0,1].set_ylabel("Phase (Radians)")
ax[0,1].set_title("Unwrapped Phase, Detection Channel")
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

fig.savefig('/scratch/thomasb' + f'/2bline_pulse{desired_pulse_idx}_{global_start_time}.jpg')