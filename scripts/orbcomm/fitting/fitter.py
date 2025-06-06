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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d",
        "--data_file",
        type = str, 
        default = "mars_2024_ant1_day1/vis_all_1721800002.h5"
    )

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="./config/config_mars_2024_day1.json",
        help="Config file containing all required data.",
    )


    #later make this configurable for multiple, ig.
    parser.add_argument(
        "-b",
        "--baseline", 
        type=int, 
        default=1,
        help="Which baseline we are getting visibilities for"
    )

    parser.add_argument(
        "-o", 
        "--output_path", 
        type=str, 
        default="/project/s/sievers/thomasb/mars_2024_ant1_day1", 
        help="Output directory for data and debug"
    )

    parser.add_argument(
        "-dg", "--debug", action="store_true", help="debug option that spits out a ton of plots to see stuff"
    )

    args = parser.parse_args()



with open(f"{args.config_file}", "r") as f:
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



observed_data = []
info = []


with h5py.File(f'/project/s/sievers/thomasb/{args.data_file}', 'r') as f:
    for pulse in f:
        # Access a dataset
        print(pulse)
        p = f[f'{pulse}']
        print(p)
        pulse_info = [int(p.attrs["start_time"]), 
                      int(p.attrs["end_time"]), 
                      int(p.attrs["satID"]),
                      int(p.attrs["chan"]),
                      int(p.attrs["glob_start_time"]),
                      str(p.attrs["tle_path"]) ]

        info.append(pulse_info)
        data = f[f'{pulse}'][:]
        observed_data.append(data)
print(info)




#important to have option to plot everything, but somehow make sure that each pass represented?
if args.debug:
    fig, ax = plt.subplots(int(np.ceil(len(observed_data)/2)), 2)
    fig.set_size_inches(8, 8)
    ax = ax.flatten()
    fig.suptitle(f"Before Fitting")
    for pulse_idx in range(len(observed_data)):
        print(pulse_idx)
        predicted_data = ch.phase_pred(a2_coords, pulse_idx, info, context)
        ax[pulse_idx].set_title(f"Pulse Idx {pulse_idx}")
        ax[pulse_idx].plot(observed_data[pulse_idx])
        ax[pulse_idx].plot(predicted_data)
    plt.tight_layout()
    fig.savefig(path.join(out_path,f"pre_fit_calib_plots_{global_start_time}.jpg"))
    print(path.join(out_path,f"prefit_plot_coordfit_{global_start_time}.jpg"))




#must include some kind of filter here, and should work with the wrapped raw visibilities here too.abs

#then apply the actual fitting stuff here.
#should call the helper functions and all of that. 
