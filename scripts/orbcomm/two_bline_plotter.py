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
out_path = '/scratch/thomasb'

T_SPECTRA = 4096/250e6
v_acclen = 10000
c_acclen = 10**6
chunk_length = T_SPECTRA * v_acclen
chunk_length_coarse = T_SPECTRA * c_acclen

bl1_ants = set({'Antenna 1', 'Antenna 2'})
bl2_ants = set({'Antenna 1', 'Antenna 8'})
required_ants = bl1_ants | bl2_ants

rel_start_t, rel_end_t = 40740, 41200
satID = 59051

bl1_offsets = [0, 115507586]
bl2_offsets = [0, -69608]

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

print(bl1_offsets)
print(bl1_coords)
print(bl1_paths)

print(bl2_offsets)
print(bl2_coords)
print(bl2_paths)

#sys.exit()

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

bl1_vis, bl1_chanlist = sug.get_vis_gpu(p_start,
                                        p_end,
                                        bl1_paths,
                                        bl1_offsets,
                                        v_acclen = v_acclen)

bl2_vis, bl2_chanlist = sug.get_vis_gpu(p_start,
                                        p_end,
                                        bl2_paths,
                                        bl2_offsets,
                                        v_acclen = v_acclen)

bl1_pol00 = bl1_vis[0,2,:,:]
bl2_pol00 = bl2_vis[0,2,:,:]
bl1_p_vis, bl1_phase, bl1_chan_big_idx = su.get_fringes_phase(bl1_pol00, bl1_chanlist)
bl2_p_vis, bl2_phase, bl2_chan_big_idx = su.get_fringes_phase(bl2_pol00, bl2_chanlist)

for i in range(len(bl1_chanlist)):
    assert bl1_chanlist[i] == bl2_chanlist[i]
assert bl1_chan_big_idx == bl2_chan_big_idx

chanlist = bl1_chanlist
chan_big_idx = bl1_chan_big_idx

print(bl1_p_vis.shape)

suptitle = f'Channel {chan_big_idx}'
fig = fgs.makeplot_fringes_phase2(bl1_coords, 
                                  bl2_coords,
                                  p_start,
                                  p_end,
                                  chan_big_idx, 
                                  bl1_p_vis, 
                                  bl2_p_vis,
                                  bl1_phase, 
                                  bl2_phase,
                                  satID,
                                  v_acclen,
                                  suptitle=suptitle)

fig.savefig(os.path.join(out_path, f'two_bline_{p_start}_{p_end}.jpg'))

