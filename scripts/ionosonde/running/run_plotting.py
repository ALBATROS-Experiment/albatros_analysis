import contextlib
import copy
import json
import time
import numpy as np

import os, sys

sys.path.insert(0, os.path.expanduser("~"))

from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde.ionogram_processing import process_and_plot

import logging
logger = logging.getLogger(__name__)

out_dir_root = "/scratch/mayas/ionograms_to_share_4bit"

def plot_one(folder_path, time, ant, summary_file):
    logger.info("=== run started ===")
    
    # Work on a fresh copy of the args for every run
    args = copy.deepcopy(default_args)
    args.start_time = time
    args.which_ant = ant
    args.baseband_dir = f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/"
    args.out_dir = folder_path

    logger.info(f"path={folder_path}")

    ref_freq, est_plasma_freq, data_to_save = process_and_plot(ref_idx_plotting = True, args = args)

    summary_file.write(f"{time},{ant},{folder_path},{args.baseband_dir},{est_plasma_freq},{ref_freq}\n")

    logger.info("=== run finished ===")

    return data_to_save

def iter_through_all(num_max = None):
    time_sup_folder_names = [folder for folder in os.listdir(out_dir_root) if folder.isdigit()]
    counter = 0
    data_dict = {}

    for time_sup_folder in time_sup_folder_names:

        time_folder_names = [folder for folder in os.listdir(os.path.join(out_dir_root, time_sup_folder)) if folder.isdigit()]

        for time_folder in time_folder_names:

            ant_folder_names = [folder for folder in os.listdir(os.path.join(out_dir_root, time_sup_folder, time_folder)) if folder[:4] == "mars"]

            for ant_folder in ant_folder_names:
                
                total_path = os.path.join(out_dir_root, time_sup_folder, time_folder, ant_folder)
                time = int(time_folder)
                data_to_save = plot_one(total_path, time, ant_folder[-1], failure_summary)
                data_dict |= data_to_save
                counter +=1

                if num_max is not None and counter >= num_max:
                    return data_dict
    return data_dict

with open(os.path.join(out_dir_root, "graphing_summary.csv"), "w") as failure_summary:
    failure_summary.write("Time,MARS Station,Correlation Directory,Baseband Directory,Estimated Plasma Frequency,Reference Frequency\n")
    data_dict = iter_through_all()

    

t0 = time.time()
np.savez_compressed(os.path.join(out_dir_root, "all_ionogram_data"), **data_dict)
logger.info("Time to save all data: ", t0)
print("DONE!! Data saved to ", os.path.join(out_dir_root, "all_ionogram_data"))