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
import sat_coarse as sc


def twobytwo_coarse_and_phase(cxcorr1, cxcorr2, coords1, coords2, global_start_t, rel_start_t, chan_idx_small, sat_ID, phase1, phase2, T_SPECTRA=4096/250e6, c_acclen=10**6, v_acclen=5000):

    #parameter setup
    center = c_acclen
    dN = 10**5
    chunk_length = c_acclen * T_SPECTRA
    spectra = np.arange(-dN, dN)
    pulse_start_t = rel_start_t + global_start_t

    chanlist = np.arange(1834, 1852)
    chan_idx_big = chanlist[chan_idx_small]
    chan_mhz = np.round(outils.chan2freq(chan_idx_big)/(10**6), decimals = 2)

    #times setup
    time_in_secs = np.round(np.arange(len(phase1)) * chunk_length).astype(int)
    t_tick_spacing = 20 
    t_tick_vals = np.arange(0, time_in_secs[-1] + t_tick_spacing, t_tick_spacing)
    t_tick_idxs = np.searchsorted(time_in_secs, t_tick_vals)
    t_tick_idxs = t_tick_idxs[t_tick_idxs < len(time_in_secs)]
    T_spec_ms = int(np.round(T_SPECTRA * 10 **6))

    #data setup
    ampdata_1 = np.abs(cxcorr1[chan_idx_small,center-dN:center+dN])
    peak_1 = np.argmax(ampdata_1)
    ampdata_2 = np.abs(cxcorr2[chan_idx_small,center-dN:center+dN])
    peak_2 = np.argmax(ampdata_2)

    pred1 = outils.pred(coords1[0], coords1[1], pulse_start_t, pulse_start_t+1000, chan_idx_big, sat_ID, v_acclen=v_acclen)[:len(phase1)]
    pred2 = outils.pred(coords2[0], coords2[1], pulse_start_t, pulse_start_t+1000, chan_idx_big, sat_ID, v_acclen=v_acclen)[:len(phase2)]

    #plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22
        })

    #top left
    fig.suptitle(f"METEOR M2-3 at {chan_mhz}MHz")
    ax[0,0].plot(spectra, ampdata_1, label=f"Peak at {peak_1 - dN}")
    ax[0,0].set_xticklabels([])
    ax[0,0].set_title(f"Coarse x-corr")
    ax[0,0].legend(loc='upper right', fontsize=12)
    ax[0,0].set_ylabel("Amplitude")

    #top right
    ax[0,1].set_ylabel("Phase (rads)")
    ax[0,1].set_title(f"Unwrapped Phase")
    ax[0,1].plot(pred1, label = 'Prediction', color='orange')
    ax[0,1].plot(phase1, label='Measurement', linestyle='--', c='blue')
    ax[0,1].legend(fontsize=12)
    ax[0,1].set_xticks(t_tick_idxs)
    ax[0,1].set_xticklabels([])
    ax[0,1].annotate(f"MARS1-\nMARS7\n{bl1_dist}m", xy=(1.05, 0.5), xycoords='axes fraction',
                    rotation=0, va='center', ha='left', fontsize=15)

    #bottom left
    ax[1,0].plot(spectra, ampdata_2, label=f"Peak at {peak_2 - dN}")
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel(f"Spectrum Shift ({T_spec_ms}" + r'$\mu$s units)')
    ax[1,0].legend(loc='upper right', fontsize=12)
    ax[1,0].set_ylabel("Amplitude")

    #bottom right
    ax[1,1].set_xlabel(f"Time (seconds)")
    ax[1,1].set_ylabel("Phase (rads)")
    ax[1,1].set_xticks(t_tick_idxs)
    ax[1,1].set_xticklabels([time_in_secs[i] for i in t_tick_idxs])
    ax[1,1].plot(pred2, label='Prediction', color='orange')
    ax[1,1].plot(phase2, label='Measurement', linestyle = '--', color='blue')
    ax[1,1].legend(fontsize=12)
    ax[1, 1].annotate(f"MARS1-\nMARS4\n{bl2_dist}m", xy=(1.05, 0.5), xycoords='axes fraction',
                    rotation=0, va='center', ha='left', fontsize=15)

    plt.tight_layout()


    fig.savefig('/scratch/thomasb' + f'/2bline_pulse{rel_start_t}_{global_start_t}.jpg')

    #Good afternoon folks, here’s a plot for a satellite pulse we see in our data. xcorr SNR is about 180 on the first baseline and 110 on the second, with an x-corr accumulation length of a million spectra = ~16secs.