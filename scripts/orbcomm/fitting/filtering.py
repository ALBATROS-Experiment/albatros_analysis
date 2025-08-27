import os
import sys
from os import path
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
import json
import argparse
from scipy.optimize import least_squares
import coord_helper as ch
import h5py
import importlib
from skyfield.api import load, EarthSatellite, Topos, wgs84
from scipy.ndimage import label as splabel
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial



def roll_poly(phase, fit_window=50, p_deg=3):
    N = len(phase)
    t = np.arange(N)

    res_sum = np.zeros(N)
    res_sum_sq = np.zeros(N)
    count = np.zeros(N)

    for i in range(N - fit_window + 1):
        idx = slice(i, i + fit_window)
        t_win = t[idx]
        y_win = phase[idx]

        #model by polynomial by rolling window
        try:
            pfit = Polynomial.fit(t_win, y_win, deg=p_deg)
            y_fit = pfit(t_win)
            res = y_win - y_fit

            res_sum[idx] += res
            res_sum_sq[idx] += res ** 2
            count[idx] += 1

        except:
            continue  

    count[count == 0] = 1
    res_avg = res_sum / count
    var = (res_sum_sq / count) - res_avg ** 2
    var[var < 0] = 0 
    local_std = np.sqrt(var)

    return local_std



def cut_mask(mask):
    #want the longest continuous strip of good data
    #which involves cutting the mask

    mask_int = mask.astype(int)
    diff = np.diff(mask_int)

    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]

    lengths = ends - starts
    if len(lengths) == 0:
        return None, None, np.zeros_like(mask, dtype=bool)

    max_idx = np.argmax(lengths)
    start, end = starts[max_idx], ends[max_idx]
    segment_mask = np.zeros_like(mask, dtype=bool)
    segment_mask[start:end] = True

    return start, end, segment_mask



def roll_curve(phase, window_size=40):
    diff2 = np.diff(np.diff(phase))  # second derivative
    curve_rms = np.zeros(len(phase))
    count = np.zeros(len(phase))

    for i in range(len(diff2) - window_size + 1):
        idx = slice(i, i + window_size)
        local_rms = np.sqrt(np.mean(diff2[idx]**2))

        # middle of the window
        center = i + window_size // 2
        curve_rms[center] += local_rms
        count[center] += 1

    # don't break math
    count[count == 0] = 1
    curve_rms = curve_rms / count

    return curve_rms






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "day",
        type = str, 
        default = "07_11"
    )

    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        default=1,
        help="how many bits for the data you want to look at. usually 1",
    )


    #later make this configurable for multiple, ig.
    parser.add_argument(
        "-bl",
        "--baseline_idx", 
        type=int, 
        default=1,
        help="Which baseline we are getting visibilities for"
    )

    parser.add_argument(
        "-w", 
        "--working_directory", 
        type=str, 
        default="/project/s/sievers/thomasb/mars_data_23", 
        help="where the magic happens. where it goes and grabs data, and also where it throws it out."
    )

    parser.add_argument(
        "-dg", "--debug", action="store_true", help="debug option that spits out a ton of plots to see stuff"
    )

    args = parser.parse_args()

working_directory = args.working_directory
bits = args.bits
day = args.day
baseline_idx = args.baseline_idx
config_file_name = f'config_{day}.json'

if args.debug:
    debug_dir = f"{working_directory}/{bits}bit/{day}/debugplots"
    os.makedirs(debug_dir, exist_ok=True)

with open(f"{working_directory}/{bits}bit/{day}/{config_file_name}", "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    # unpack information from the json file
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        if (i == 0) or (i ==baseline_idx):
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
    global_start_time = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    v_acclen = config["correlation"]["vis_acclen"]
    visibility_window = config["correlation"]["visibility_window"]
    T_SPECTRA = config["correlation"]["point_PFB"] / config["correlation"]["sample_rate"]

ref_coords = coords[0]
v_nchunks = int((visibility_window)/(v_acclen* T_SPECTRA))

context = [visibility_window, T_SPECTRA, v_acclen, v_nchunks, ref_coords]
print(context)
print(global_start_time)



pl = []
data_file = f'visraw_bline{baseline_idx}_{global_start_time}.h5'

with h5py.File(f'{working_directory}/{bits}bit/{day}/{data_file}', 'r') as f:
    baseline = f[f'baseline_{baseline_idx}']
    for p in baseline:
        pulse_info = []
        pulse_info.append(p) #index 0
        pulse_info.append([int(baseline[p].attrs['start_time']), int(baseline[p].attrs['end_time']), int(baseline[p].attrs['global_start_time'])]) # index 1
        pulse_info.append(json.loads(baseline[p].attrs['sats'])) #index 2
        #pulse_info.append(f[p].attrs['tle_path']) #index 3
        pulse_info.append(outils.get_tle_file(int(baseline[p].attrs['global_start_time']) + int(baseline[p].attrs['start_time']), "/project/s/sievers/mohanagr/OCOMM_TLES"))
        pulse_info.append(baseline[f'{p}'][:])  # index 4
        pl.append(pulse_info)

#note on ordering of the pulse_data list:
# example entry:   [[relative_start_time, relative_end_time, global_start_time], {satID:[chan1, chan2]}, tle_path, [observed data array]]

gpd = []
std_tol = 0.15
roll_tol = 0.6
channel_list = np.arange(1834, 1852, dtype=int)

for pulse in pl:
    pi = []

    #unpack
    label = pulse[0]
    rel_s_time, global_start_time = pulse[1][0], pulse[1][2]
    satID = int(list(pulse[2].keys())[0])  #for now only worry about one satellite
    channels_present = list(pulse[2].values())[0]
    tle_path = pulse[3]
    vis = pulse[4]
    
    amp = np.abs(vis)
    p_vis = np.angle(vis[:])
    mean_amp = []
    for i in range(18):
        mean_amp.append(np.mean(amp[:,i]))

    chan_small_idx = np.where(mean_amp == np.max(mean_amp))[0][0]
    chan_big_idx = channel_list[chan_small_idx]
    phase = np.unwrap(p_vis[:,chan_small_idx])
    diff = np.diff(phase)


    if args.debug == True:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        ax[0,0].set_xlabel("Channel Index (~60kHz interval)")
        ax[0,0].set_ylabel("Chunk Number (~0.5s interval)")
        ax[0,0].set_title("Phase over Satellite Pulse")
        im = ax[0,0].imshow(p_vis, aspect='auto', cmap='RdBu', interpolation="none")
        fig.colorbar(im)

        ax[0,1].set_xlabel("Channel Index (~60kHz interval)")
        ax[0,1].set_ylabel("Mean Amplitude")
        ax[0,1].set_title("Mean Amplitude across Channels")
        ax[0,1].plot(mean_amp)

        ax[1,0].set_xlabel("Chunk Number (~0.5s interval)")
        ax[1,0].set_ylabel("Phase (Radians)")
        ax[1,0].plot(phase)

        ax[1,1].set_xlabel("Chunk Number (~0.5s interval)")
        ax[1,1].set_ylabel("First Derivative")
        ax[1,1].plot(diff)

        plt.savefig(f'{working_directory}/{bits}bit/{day}/debugplots/{label}_pre_cut.png')

    std = roll_poly(phase)
    mask = std < std_tol 

    start, end, _ = cut_mask(mask)
    if (start is None) or (end is None):
        continue
    cut_phase = phase[start:end]

    curve = roll_curve(cut_phase)
    curve_mask = curve < roll_tol     
    mask[start:end] &= curve_mask  

    start_final, end_final, final_mask = cut_mask(mask) #get longest continuous value where this works
    if (start_final is None) or (end_final is None):
        continue
    silent_phase = phase[start_final:end_final]

    start_s = start_final * (context[2]*context[1]) + rel_s_time
    end_s = end_final * (context[2]*context[1]) + rel_s_time

    if end_final - start_final < 200:
        continue

    if args.debug == True:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.4, wspace = 0.3)

        ax[0,0].set_xlabel("Channel Index (~60kHz interval)")
        ax[0,0].set_ylabel("std")
        ax[0,0].set_title("ROLLING STD COMPUTATION")
        ax[0,0].plot(std)
        ax[0,0].axhline(y=std_tol, color='red', linestyle='--', linewidth=1)

        ax[0,1].set_xlabel("Chunk Number (~0.5s interval)")
        ax[0,1].set_ylabel("Phase (Radians)")
        ax[0,1].set_title("FIRST CUT PHASE")
        ax[0,1].plot(cut_phase)

        ax[1,0].set_xlabel("Chunk Number (~0.5s interval)")
        ax[1,0].set_ylabel("second derivative of cut phase")
        ax[1,0].set_title("ROLLING CURVE")
        ax[1,0].plot(curve)
        ax[1,0].axhline(y=roll_tol, color='red', linestyle='--', linewidth=1)

        ax[1,1].set_xlabel("Chunk Number (~0.5s interval)")
        ax[1,1].set_ylabel("second derivative of cut phase")
        ax[1,1].set_title("SECOND CUT")
        ax[1,1].plot(silent_phase)

        plt.savefig(f'{working_directory}/{bits}bit/{day}/debugplots/{label}_cutting.png')

    pi.append(label)
    pi.append([start_s, end_s, global_start_time])
    pi.append([satID, chan_big_idx])
    pi.append(tle_path)
    pi.append(vis[start_final:end_final, :])
    print(vis[start_final:end_final, :].shape)

    gpd.append(pi)

with h5py.File(f'{working_directory}/{bits}bit/{day}/vis_autoselected_bline_{baseline_idx}_{global_start_time}.h5', 'a') as f:
    #beware: baseline_idx is a number, baseline is a group object in the h5 file.
    baseline = f.require_group(f'baseline_{baseline_idx}')
    for idx in range(len(gpd)):
        #name the data after the label
        dataset_name = f'{gpd[idx][0]}'
        if dataset_name in baseline:
            del baseline[dataset_name]
        pulse_array = baseline.create_dataset(f'{gpd[idx][0]}', data=gpd[idx][4])
        pulse_array.attrs['start_time']        = gpd[idx][1][0]
        pulse_array.attrs['end_time']          = gpd[idx][1][1]
        pulse_array.attrs['global_start_time'] = gpd[idx][1][2]
        pulse_array.attrs['sat']               = gpd[idx][2][0]
        pulse_array.attrs['chan']              = gpd[idx][2][1]
        pulse_array.attrs['tle_path']          = gpd[idx][3]
        #add std stuff!!!!

    baseline.attrs["vis_window"] = visibility_window
    baseline.attrs["v_acclen"] = v_acclen
    baseline.attrs["T_SPECTRA"] = T_SPECTRA
    baseline.attrs["v_nchunks"] = v_nchunks
    baseline.attrs["ref_coords"] = ref_coords #necessary?
    baseline.attrs["baseline_idx"] = baseline_idx
    baseline.attrs["std_tol"] = std_tol
    baseline.attrs["roll_tol"] = roll_tol


