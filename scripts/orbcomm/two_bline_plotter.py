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
import sat_utils as sc


def get_complex_snr(signal_data, noise_data):
    signal = np.max(np.abs(signal_data))
    im_std = np.std(noise_data.imag)
    re_std = np.std(noise_data.real)
    std = np.sqrt(im_std**2 + re_std**2)

    return signal/std


def dist(coord1, coord2):
    ''' 
    Returns the actual physical distance between two coordinates. 
    '''

    lat1, lon1, alt1 = coord1[0], coord1[1], coord1[2]
    lat2, lon2, alt2 = coord2[0], coord2[1], coord2[2]

    mean_lat = np.radians((lat1 + lat2) / 2)

    meters_per_deg_lat = 111_320 
    meters_per_deg_lon = 111_320 * np.cos(mean_lat)

    delta_lat_deg = lat2 - lat1
    delta_lon_deg = lon2 - lon1
    delta_alt = float(alt2 - alt1)

    delta_lat_m = float(delta_lat_deg * meters_per_deg_lat)
    delta_lon_m = float(delta_lon_deg * meters_per_deg_lon)

    dist_total = np.sqrt(delta_lat_m**2 + delta_lon_m**2 + delta_alt**2)

    return dist_total


#--------------------hard coded setup------------
config_path = '/home/thomasb/albatros_analysis/scripts/orbcomm'
config_name = 'config2.json'
#config_name = 'config.json'

satdet_path = '/scratch/thomasb/'
satdet_name = "pulsedata_1753200150_1757992159.256902.json"
#satdet_name = "pulsedata_1753133403/pulsedata_1753133403_1757540615.4208999.json"

T_SPECTRA = 4096/250e6
cap = 1400

desired_pulse_start_time = 10545   #this is the current (not super thought out) way to identify pulses
bl1_ants = set({'Antenna 1', 'Antenna 7'})
bl2_ants = set({'Antenna 1', 'Antenna 4'})
bl1_coords, bl2_coords = [], []
bl1_paths, bl2_paths = [], []
required_antennas = bl1_ants | bl2_ants


#-------------------unpack information---------------

antennas = []
with open(f"{config_path}/{config_name}", "r") as f:
    config = json.load(f)
    for i, (ant, details) in enumerate(config["antennas"].items()):
        ant_dict = {}
        ant_dict['name'] = details['name']
        ant_dict['path'] = details['path']

        if i ==0:
            ant_dict['reference'] = 'T'
        else:
            ant_dict['reference'] = 'F'

        if ant_dict['name'] in bl1_ants:
            bl1_coords.append(details['coordinates'])
            bl1_paths.append(details['path'])
        if ant_dict['name'] in bl2_ants:
            bl2_coords.append(details['coordinates'])
            bl2_paths.append(details['path'])

        if ant_dict['name'] in required_antennas:
            antennas.append(ant_dict)

    global_start_time = config["correlation"]["start_timestamp"]
    global_end_time = config["correlation"]["end_timestamp"]
    v_acclen = config["correlation"]["vis_acclen"]
    c_acclen = config["correlation"]["coarse_acclen"]

v_acclen = 5000
    
nants = len(antennas)
print(antennas)
print('number of antenna:', nants)
print("Visibility Accumulation Length", v_acclen)
print('global start and end times:', global_start_time, global_end_time)

tle_path = outils.get_tle_file(global_start_time, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
chunk_length = T_SPECTRA * v_acclen
chunk_length_coarse = T_SPECTRA * c_acclen

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

print(desired_pulse_info_all)

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

bl1_dist = int(np.round(dist(bl1_coords[0], bl1_coords[1]), decimals=-1))
bl2_dist = int(np.round(dist(bl2_coords[0], bl2_coords[1]), decimals=-1))
#set up some times, channels, and compute the visibilities for all blines
#note that we are taking times and such wrt Ant1-Ant2 bline. 
#this is fine since the pulse times are all the same for each bline

#add some more double check here or something.

rel_start_t, rel_end_t = desired_pulse_info_all[0]['start'] , desired_pulse_info_all[0]['end']
sats = list(desired_pulse_info_all[0]['sats_present'].keys())
pulse_start_t = rel_start_t + global_start_time
pulse_end_t = rel_end_t + global_start_time
pulse_len_secs = rel_end_t - rel_start_t
pulse_len_chunks = int(np.ceil((pulse_end_t - pulse_start_t)/chunk_length))
pulse_len_chunks_coarse = int(np.ceil((pulse_end_t - pulse_start_t)/chunk_length_coarse))
if pulse_len_chunks > cap:
    pulse_len_chunks = cap

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

snr1 = get_complex_snr(amp1[:,chan_s_idx1], noise_data_1)
snr2 = get_complex_snr(amp2[:,chan_s_idx1], noise_data_2)

print('SNRS:', snr1, snr2)


#finally we can unwrap this guy
phase1, phase2 = np.unwrap(p_vis1[:, chan_s_idx1]) - p_vis1[0, chan_s_idx1], np.unwrap(p_vis2[:, chan_s_idx1] - p_vis2[0, chan_s_idx1])  #automate the sat and channel selection



#-----------------------coarse plotting----------------------------

sat_ID = 57166 #hard-coded for now

bl1_uncorr, bl1_corr = sc.get_coarse_offset(bl1_paths, bl1_coords, desired_pulse_start_time, global_start_time, sat_ID)
bl2_uncorr, bl2_corr = sc.get_coarse_offset(bl2_paths, bl2_coords, desired_pulse_start_time, global_start_time, sat_ID)


#--------------------------plotting--------------------------------


plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 22
    })

#setup
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
#fig.subplots_adjust(wspace=0.4)
#fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(right=0.85) 

#sort out frequencies
#frequencies = []
#for freq in chanlist:
#    frequencies.append(outils.chan2freq(freq)/(10**6))
#frequencies = np.round(frequencies, decimals=1)
#nu_tick_vals = np.arange(0, len(frequencies), 5)
#nu_tick_idxs = frequencies[nu_tick_vals]

center = c_acclen
dN = 100000
channel = np.round(outils.chan2freq(chan_b_idx1)/(10**6), decimals = 2)

#sort out times
time_in_secs = np.round(np.arange(len(phase1)) * chunk_length).astype(int)
t_tick_spacing = 20 
t_tick_vals = np.arange(0, time_in_secs[-1] + t_tick_spacing, t_tick_spacing)
t_tick_idxs = np.searchsorted(time_in_secs, t_tick_vals)
t_tick_idxs = t_tick_idxs[t_tick_idxs < len(time_in_secs)]
T_spec_ms = int(np.round(T_SPECTRA * 10 **6))


spectra = np.arange(-dN, dN)

#ax[0,0].set_ylabel(f"Chunks (~{np.round(chunk_length, decimals=2)}s)")
#ax[0,0].set_title("Wrapped Phase")
#im1 = ax[0,0].imshow(p_vis1, aspect='auto', cmap='RdBu', interpolation="none")
#cbar = fig.colorbar(im1, ax=ax[0,0])
#cbar.ax.tick_params(labelsize=10)
#ax[0,0].set_xticks(nu_tick_vals)
#ax[0,0].set_xticklabels([])

#top left
fig.suptitle(f"METEOR M2-3 at {channel}MHz")
bl1_corrdata = bl1_corr[chan_s_idx1,center-dN:center+dN]
bl1_peak = np.argmax(np.abs(bl1_corrdata))
bl1_corr_snr = get_complex_snr(bl1_corrdata, bl1_corrdata[dN:])
ax[0,0].plot(spectra, np.abs(bl1_corrdata), label=f"Peak at {bl1_peak - dN}")
ax[0,0].set_xticklabels([])
ax[0,0].set_title(f"Coarse x-corr")
ax[0,0].legend(loc='upper right', fontsize=12)
ax[0,0].set_ylabel("Amplitude")


#top right

predicted1 = outils.pred(bl1_coords[0], bl1_coords[1], pulse_start_t, pulse_end_t, chan_b_idx1, 57166, v_acclen=v_acclen)[:len(phase1)]
bl1_phase_data = predicted1[:10000] - phase1[:10000]
bl1_phasenoise = np.std(bl1_phase_data)

marker_indices = np.arange(0, len(phase1), 100)


ax[0,1].set_ylabel("Phase (rads)")
ax[0,1].set_title(f"Unwrapped Phase")
ax[0,1].plot(predicted1, label = 'Prediction', color='orange')
ax[0,1].plot(phase1, label='Measurement', linestyle='--', c='blue')
ax[0,1].legend(fontsize=12)
ax[0,1].set_xticks(t_tick_idxs)
ax[0,1].set_xticklabels([])
ax[0,1].annotate(f"MARS1-\nMARS7\n{bl1_dist}m", xy=(1.05, 0.5), xycoords='axes fraction',
                  rotation=0, va='center', ha='left', fontsize=15)



#ax[1,0].set_xlabel("Freq. Channel (MHz)")
#ax[1,0].set_ylabel(f"Chunks (~{np.round(chunk_length, decimals=2)}s)")
#im2 = ax[1,0].imshow(p_vis2, aspect='auto', cmap='RdBu', interpolation="none")
#cbar = fig.colorbar(im2, ax=ax[1,0])
#cbar.ax.tick_params(labelsize=10)
#ax[1,0].set_xticks(nu_tick_vals)
#ax[1,0].set_xticklabels(nu_tick_idxs)
#ax[1,0].tick_params(axis='x', labelrotation=45)



bl2_corrdata = bl2_corr[chan_s_idx1,center-dN:center+dN]
bl2_peak = np.argmax(np.abs(bl2_corrdata))
bl2_corr_snr = get_complex_snr(bl2_corrdata, bl2_corrdata[dN:])
ax[1,0].plot(spectra, bl2_corrdata, label=f"Peak at {bl2_peak - dN}")
ax[1,0].tick_params(axis='x', labelsize=12)
ax[1,0].set_xlabel(f"Spectrum Shift ({T_spec_ms}" + r'$\mu$s units)')
ax[1,0].legend(loc='upper right', fontsize=12)
ax[1,0].set_ylabel("Amplitude")

#bottom right
predicted2 = outils.pred(bl2_coords[0], bl2_coords[1], pulse_start_t, pulse_end_t, chan_b_idx1, 57166, v_acclen=v_acclen)[:len(phase2)]
bl2_phasedata = predicted2[:10000] - phase2[:10000]
bl2_phasenoise = np.std(bl2_phasedata)

ax[1,1].set_xlabel(f"Time (seconds)")
ax[1,1].set_ylabel("Phase (rads)")
ax[1,1].set_xticks(t_tick_idxs)
ax[1,1].set_xticklabels([time_in_secs[i] for i in t_tick_idxs])
ax[1,1].plot(predicted2, label='Prediction', color='orange')
ax[1,1].plot(phase2, label='Measurement', linestyle = '--', color='blue')
ax[1,1].legend(fontsize=12)
ax[1, 1].annotate(f"MARS1-\nMARS4\n{bl2_dist}m", xy=(1.05, 0.5), xycoords='axes fraction',
                  rotation=0, va='center', ha='left', fontsize=15)

plt.tight_layout()

print('xcorr SNRs', bl1_corr_snr, bl2_corr_snr)
print('phase noises', bl1_phasenoise, bl2_phasenoise)

fig.savefig('/scratch/thomasb' + f'/2bline_pulse{desired_pulse_start_time}_{global_start_time}.jpg')

#Good afternoon folks, here’s a plot for a satellite pulse we see in our data. xcorr SNR is about 180 on the first baseline and 110 on the second, with an x-corr accumulation length of a million spectra = ~16secs.