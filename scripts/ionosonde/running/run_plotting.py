import logging
logger = logging.getLogger(__name__)

import contextlib
import copy
import json

import os, sys

sys.path.insert(0, os.path.expanduser("~"))

from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde.ionogram import process_and_plot

out_dir_root = "/scratch/mayas/ionograms_to_share_4bit"

def plot_one(folder_path, time, ant, summary_file):
    # Work on a fresh copy of the args for every run
    args = copy.deepcopy(default_args)
    args.start_time = time
    args.which_ant = ant
    args.baseband_dir = f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/"
    args.out_dir = folder_path
    exception = None

    logger.info("=== run started ===")
    logger.info(f"path={folder_path}")
    
    process_and_plot(args = args)
    status = "success"

    with open(os.path.join(folder_path, "params.json"), "w") as json_file:
        args.all_freqs = list(args.all_freqs)
        args.ionosonde_freqs = list(args.ionosonde_freqs)
        json.dump(vars(args), json_file, indent=4) 

    logger.info("=== run finished [%s] ===", status)

    summary_file.write(f"{status == 'success'},{folder_path},{exception}\n")

with open(os.path.join(out_dir_root, "graphing_summary.csv"), "a") as failure_summary:
    failure_summary.write("Success?,MARS Station,Time,Exception\n")

    time_sup_folder_names = os.listdir(out_dir_root)

    for time_sup_folder in time_sup_folder_names:

        time_folder_names = os.listdir(os.path.join(out_dir_root, time_sup_folder))

        for time_folder in time_folder_names:

            ant_folder_names = os.listdir(os.path.join(out_dir_root, time_sup_folder, time_folder))

            for ant_folder in ant_folder_names:
                total_path = os.path.join(out_dir_root, time_sup_folder, time_folder, ant_folder)
                plot_one(total_path, int(time_folder), ant_folder[-1], failure_summary)