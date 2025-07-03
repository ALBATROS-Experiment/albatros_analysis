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
import extra_functions as ef
import h5py
import numbers


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
        "--baseline", 
        type=int, 
        default=1,
        help="Which baseline we are getting visibilities for"
    )

    parser.add_argument(
        "-w", 
        "--working_directory", 
        type=str, 
        default="/project/s/sievers/thomasb/mars_data_24", 
        help="where the magic happens. where it goes and grabs data, and also where it throws it out."
    )

    parser.add_argument(
        "-dg", "--debug", action="store_true", help="debug option that spits out a ton of plots to see stuff"
    )

    args = parser.parse_args()


working_directory = args.working_directory
bits = args.bits
day = args.day
baseline = args.baseline


with open(f"{working_directory}/{bits}bit/{day}/config_{day}.json", "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    # unpack information from the json file
    # Call get_starting_index for all antennas except reference
    print('\n', "Antenna Details:")
    for i, (ant, details) in enumerate(config["antennas"].items()):
        if (i == 0) or (i ==args.baseline):
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
    global_start_time = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    v_acclen = config["correlation"]["vis_acclen"]
    visibility_window = config["correlation"]["visibility_window"]
    T_SPECTRA = config["correlation"]["point_PFB"] / config["correlation"]["sample_rate"]

print("Antenna Paths:", dir_parents, '\n')
print("Antenna Coordinates:", coords, '\n')
print("Visibility Accumulation Length", v_acclen, '\n')

v_nchunks = int((visibility_window)/(v_acclen* T_SPECTRA))
context = [visibility_window, T_SPECTRA, v_acclen, v_nchunks, coords[0]]


# if I want to run multiple baselines at once, then I should make pulse list an array with one entry for each baseline
pulse_list = []

with h5py.File(f'{working_directory}/{bits}bit/{day}/vis_selected_bline_{baseline}_{global_start_time}.h5', 'r') as f:
    for p in f:
        pulse_info = []
        pulse_info.append(p)
        pulse_info.append([f[p].attrs['start_time'], f[p].attrs['end_time'], int(f[p].attrs['global_start_time'])])
        pulse_info.append([int(f[p].attrs['sat']), int(f[p].attrs['chan'])])
        pulse_info.append(f[p].attrs['tle_path'])
        pulse_info.append(f[f'/{p}'][:])
        pulse_list.append(pulse_info)




#important to have option to plot everything, but somehow make sure that each pass represented?


#TO DO: - plots, residuals, etc
# also option for satpasses: i.e. distribution of passes

if args.debug:
    fig, ax = plt.subplots(int(np.ceil(len(observed_data)/2)), 2)
    fig.set_size_inches(8, 8)
    ax = ax.flatten()
    fig.suptitle(f"Before Fitting")
    for pulse_idx in range(len(observed_data)):
        print(pulse_idx)
        predicted_data = ef.pred(a2_coords, 0, pulse_idx, info, context)
        ax[pulse_idx].set_title(f"Pulse Idx {pulse_idx}")
        ax[pulse_idx].plot(observed_data[pulse_idx])
        ax[pulse_idx].plot(predicted_data)
    plt.tight_layout()
    fig.savefig(path.join(out_path,f"pre_fit_calib_plots_{global_start_time}.jpg"))
    print(path.join(out_path,f"prefit_plot_coordfit_{global_start_time}.jpg"))


    fig = ef.satpass_plotter(pulse_list, coords[baseline])
    fig.savefig(f'satpass_{day}_bline{baseline}')


print("--------------------SOLID-----------------")
solid = ef.solid_irls(coords[baseline], 0, ef.pred, pulse_list, context, iterations = 10)

print("--------------------JOINT-----------------")
joint = ef.joint_irls(coords[baseline], ef.pred, pulse_list, context, iterations = 0)


fits = {}
fits['solid'] = solid.tolist()
fits['joint'] = joint.tolist()


path_to_json = f'{working_directory}/{bits}bit/coords_v3.json'

ef.add_to_json(day, baseline, fits, path_to_json)




# add debug plot of where these all are?

# do all the fitting methods you can think of
