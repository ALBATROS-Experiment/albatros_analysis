import os
import sys
sys.path.append(os.path.expanduser('~'))
import numpy as np 
from matplotlib import pyplot as plt
from datetime import datetime as dt
import figures as fgs
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
import numba as nb
import time
import importlib
import json
from scipy.optimize import minimize
from skyfield.api import load, wgs84
import cupy
import offset_helper as oh


config_path = '/home/thomasb/albatros_analysis/scripts/xcorr/config/config_bright_test.json'
disk_path = '/scratch/thomasb/raw_1753216650:1753217000_bit=1_ant=7_pol=2_cha=1834:1852_tim=650_upx=64_acc=512_ipfb=0.4_complex64_regular_20260110T114008.npy'


data_all = np.load(disk_path)

#need to extract the starting specnumber from the data we get: first baseband spectrum index
#once we have that and the time offset we have our mapping
#and can compare to other pulses and if their maps agree


with open(config_path, "r") as f:
    config = json.load(f)
ant_names, ant_coords, dir_parents, spec_offsets = [], [], [], []
# Call get_starting_index for all antennas except reference
for i, (ant, details) in enumerate(config["antennas"].items()):
    dir_parents.append(details["path"])
    spec_offsets.append(details["clock_offset"])
    ant_coords.append(details["coordinates"])

batch_start_ts = config["correlation"]["start_timestamp"]
batch_end_ts = config["correlation"]["end_timestamp"]
chanstart = config["frequency"]["start_channel"]
chanend = config["frequency"]["end_channel"]
osamp = config["correlation"]["osamp"]

print('batch start ts', batch_start_ts)
print('batch end ts', batch_end_ts)



#-----------------------SET GLOBAL VARIABLE STUFF--------------
acclen = 1024
bb_spectrum_T = 4096/250e6
T_SPECTRA = bb_spectrum_T * osamp
nchans = data_all.shape[3]
sat_freqs = 250e6 - ((np.arange(nchans)/osamp + 1834)/(4096/250e6))  #put as chan_start instead of 1834?
ant_idxs = [0, 2, 3, 4, 5, 6]


#--------------------------PULSE DEPENDENT STUFF------------
satID = 57166
chan_new = 140
freq = sat_freqs[chan_new]

cut_spectra_start = 60000
cut_seconds_end = 150

pulse_start_ts = batch_start_ts + T_SPECTRA* cut_spectra_start
pulse_end_ts = batch_end_ts - cut_seconds_end

data_slice_cut = data_all[:, :, cut_spectra_start:, chan_new]



#-------------------------PLOT PHASES PRE-FIT------------------
chisq_unfitted, phases_fig_unfitted = objective_times(0,
                                    pulse_start_ts,
                                    pulse_end_ts,
                                    data_slice_cut,
                                    ant_idxs,
                                    ant_coords,
                                    freq,
                                    satID,
                                    plot=True,
                                    bb_spectrum_T=bb_spectrum_T,
                                    osamp=osamp)


#------------------------GET COST CURVE---------------------
offsets, chisqs, cost_fig = cost_curve(-2, 
                                    2, 
                                    101, 
                                    pulse_start_ts, 
                                    pulse_start_ts, 
                                    data_slice_cut, 
                                    ant_idxs, 
                                    ant_coords, 
                                    freq, 
                                    satID, 
                                    bb_spectrum_T=bb_spectrum_T, 
                                    osamp=osamp,
                                    include_fig=True)
offset_guess = offsets[np.argmin(chisqs)]


#--------------------FIT FOR MINIMUM USING GUESS-------------------
result = minimize(objective_times, 
    offset_guess,
    args=(pulse_start_ts, 
            pulse_start_ts, 
            data_slice_cut, 
            ant_idxs, 
            ant_coords, 
            freq, 
            satID,
            False,
            bb_spectrum_T,
            osamp,
            acclen), 
    method='Nelder-Mead',
    tol=1e-12)
offset_fitted = result.x[0]


#----------------------------PLOT ZOOMED COST CURVE------------------------
offset_rounded = np.round(offset_fitted, decimals = 2)
print(offset_rounded)
_, _, cost_fig_zoomed = cost_curve(offset_rounded-0.2, 
                                    offset_rounded+0.2, 
                                    101, 
                                    pulse_start_ts, 
                                    pulse_end_ts, 
                                    data_slice_cut, 
                                    ant_idxs, 
                                    ant_coords, 
                                    freq, 
                                    satID,
                                    include_fig=True,
                                    bb_spectrum_T=bb_spectrum_T, 
                                    osamp=osamp,
                                    vline=offset_fitted)


#--------------------------PLOT PHASES POST-FIT----------------------------------
chisq_fitted, phases_fig_fitted = objective_times(offset_fitted,
                                pulse_start_ts,
                                pulse_end_ts,
                                data_slice_cut,
                                ant_idxs,
                                ant_coords,
                                freq,
                                satID,
                                plot=True,
                                bb_spectrum_T=bb_spectrum_T,
                                osamp=osamp)


#------------------------SAVE FIGURES AND PRINT STUFF---------------------------
figpath = os.path.join(out_path, f"timing_discrepancies_{batch_start_ts}_{satID}")
cost_fig.savefig(os.path.join(figpath, 'cost_curve_initial.png'))
cost_fig_zoomed.savefig(os.path.join(figpath, 'cost_curve_zoomed.png'))
phases_fig_fitted.savefig(os.path.join(figpath, 'fitted_phases.png'))
phases_fig_unfitted.savefig(os.path.join(figpath, 'unfitted_phases.png'))

print('initial chisq', chisq_unfitted)
print('final chisq', chisq_fitted)
print('fitted offset', offset_fitted)