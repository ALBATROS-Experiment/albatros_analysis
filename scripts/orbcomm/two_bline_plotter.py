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
import sat_utils as su
import sat_utils_gpu as sug
import figures as fgs

#--------------------hard coded setup------------
config_name = "config2.json"
satdet_name = "pulsedata_1753132820_len_67200_1760024247.5361912.json"
config_path = os.path.join('/home/thomasb/albatros_analysis/scripts/orbcomm', config_name)
satdet_path = os.path.join('/scratch/thomasb/', satdet_name)

T_SPECTRA = 4096/250e6
v_acclen = 5000
c_acclen = 10**6
chunk_length = T_SPECTRA * v_acclen
chunk_length_coarse = T_SPECTRA * c_acclen

bl1_ants = set({'Antenna 1', 'Antenna 7'})
bl2_ants = set({'Antenna 1', 'Antenna 4'})
required_ants = bl1_ants | bl2_ants

rel_start_t, rel_end_t = 
satID = 

bl1_offsets = []
bl2_offsets = []

bl1_names, bl1_paths, bl1_coords = [], [], []
bl2_names, bl2_paths, bl2_coords = [], [], []

antname_to_idx = []


with open(config_path, "r") as f:
    config = json.load(f)
    for i, (ant, details) in enumerate(config["antennas"].items()):

        if details['name'] in required_ants:
            antname_to_idx.append(details['name'])

        if details['name'] in bl1_ants:
            bl1_names.append(details['name'])
            bl1_coords.append(details['coordinates'])
            bl1_paths.append(details['path'])

        if details['name'] in bl2_ants:
            bl2_names.append(details['name'])
            bl2_coords.append(details['coordinates'])
            bl2_paths.append(details['path'])

    global_start_t = config["correlation"]["start_timestamp"]
    global_end_t = config["correlation"]["end_timestamp"]


print("Visibility Accumulation Length", v_acclen)
print('global start and end times:', global_start_t, global_end_t)

tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
p_start, p_end = rel_start_t+global_start_t, rel_end_t+global_start_t

#set up bline map, find desired baselines in terms of their index
#bl_map = []
#nants = len(required_ants)
#for i in range(nants):
    #for j in range(i+1, nants):
     #   bl_elements = set({ant_names[i], ant_names[j]})
      #  bl_map.append(bl_elements)
#print('baseline map:', bl_map)

#bl1 = [i for i, x in enumerate(bl_map) if x == bl1_ants]
#bl2 = [i for i, x in enumerate(bl_map) if x == bl2_ants]
#print('bl1, bl2:', bl1, bl2)

bl1_dist = int(np.round(su.get_bline_dist(bl1_coords[0], bl1_coords[1]), decimals=-1))
bl2_dist = int(np.round(su.get_bline_dist(bl2_coords[0], bl2_coords[1]), decimals=-1))

bl1_pol0, bl1_pol1, _, _, _ = su.get_vis_cpu(p_start,
                                             p_end,
                                             bl1_paths,
                                             bl1_offsets,
                                             T_SPECTRA = 4096/250e6,
                                             v_acclen = 5000)

#rel_start_t, rel_end_t = desired_pulse_info_all[0]['start'] , desired_pulse_info_all[0]['end']
#sats = list(desired_pulse_info_all[0]['sats_present'].keys())
#pulse_start_t = rel_start_t + global_start_time
#pulse_end_t = rel_end_t + global_start_time
#pulse_len_secs = rel_end_t - rel_start_t
#pulse_len_chunks = int(np.ceil((pulse_end_t - pulse_start_t)/chunk_length))
#pulse_len_chunks_coarse = int(np.ceil((pulse_end_t - pulse_start_t)/chunk_length_coarse))

#print('relative start, end t:', rel_start_t, rel_end_t)
#print('pulse_duration:', rel_end_t-rel_start_t)

#print('GETTING INIT INFO --------------------')
#idxs, files = hp.get_init_info_all_ant(pulse_start_t, pulse_end_t, ant_offsets, ant_paths)

#channels = bdc.get_header(files[0][0])["channels"].astype('int64')
#chanstart = np.where(channels == 1834)[0][0] 
#chanend = np.where(channels == 1852)[0][0]
#print('starting, ending channels:', chanstart, chanend)
#chanlist = np.arange(1834, 1852)

#print('GETTING VISIBILITIES------------------')
#vis, rowcount, obj = hp.get_avg_fast2(idxs,files,v_acclen,pulse_len_chunks, chanstart, chanend)

#vis is shape (nchunks, nbl, npols, ncols)
#where nbl index is our baseline_map index
#want all chunks and all ncols (=channels)
#just have to pick what pol we want to work on.

print('total vis shape:', vis.shape)

#now we have some visibility data, so let's massage it a bit
vis_1, vis_2 = vis[:1400 , bl1, 0, :], vis[:1400 , bl2, 0, :]  #may need to interpolate
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

noise_data_1 = vis_1[:, 5:17].ravel()  
noise_data_2 = vis_2[:, 5:17].ravel()



#finally we can unwrap this guy
phase1, phase2 = np.unwrap(p_vis1[:, chan_s_idx1]) - p_vis1[0, chan_s_idx1], np.unwrap(p_vis2[:, chan_s_idx1] - p_vis2[0, chan_s_idx1])  #automate the sat and channel selection



#-----------------------coarse plotting----------------------------

sat_ID = 57166 #hard-coded for now


cxcorr1 = sug.get_cxcorr_many_sats()
cxcorr2 = sug.get_cxcorr_many_sats()

p_vis1, phase1 = su.get_fringes_phase()
p_vis2, phase2 = su.get_fringer_phase()

fig1 = fgs.makeplot_cxcorr_phase2(cxcorr1, 
                                 cxcorr2, 
                                 coords1, 
                                 coords2, 
                                 global_start_t, 
                                 rel_start_t, 
                                 chan_idx_small, 
                                 sat_ID, 
                                 phase1, 
                                 phase2, 
                                 T_SPECTRA=4096/250e6, 
                                 c_acclen=10**6, 
                                 v_acclen=5000)

fig2 = 

