import contextlib
import copy

import logging
logger = logging.getLogger(__name__)

import os, sys

sys.path.insert(0, os.path.expanduser("~"))

from albatros_analysis.scripts.ionosonde.params import default_args
from albatros_analysis.scripts.ionosonde.data import process_from_data

out_dir_root = "/scratch/mayas/ionograms_to_share"

def run_one(ant, folder_name, time, failure_summary):
    # Work on a fresh copy of the args for every run
    args = copy.deepcopy(default_args)
    args.start_time = time
    args.corr_time = 15
    args.baseband_dir = baseband_dir
    args.out_dir = f"{out_dir_root}/{folder_name}/{time}/mars{ant}/"

    #os.makedirs(args.out_dir, exist_ok=True)

    exception = None

    logger.info("=== run started ===")
    logger.info(
        "antenna=%s folder=%s timestamp=%s (start_time=%s)",
        ant, folder_name, time, args.start_time,
    )
    try:
        process_from_data(args)
        status = "success"
    except Exception as e:
        logger.exception("run failed")
        status = "FAILURE"
        exception = e

    logger.info("=== run finished [%s] ===", status)

    failure_summary.write(f"{status == 'success'},{ant},{time},{exception}\n")

with open(os.path.join(out_dir_root, "failure_summary.csv"), "a") as failure_summary:
    failure_summary.write("Success?,MARS Station,Time,Exception\n")

    for ant in [2, 3, 7]:
        baseband_dir = f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/"
        folder_names = os.listdir(baseband_dir)

        for folder_name in folder_names:
            folder_path = os.path.join(baseband_dir, folder_name)
            file_names = os.listdir(folder_path)

            times = [int(f[:-4]) for f in file_names if f.endswith(".raw")]

            if not times:
                # Nothing to process in this folder, so just skip
                continue

            # The ionosonde broadcasts every 5 minutes, with the exception of
            # the hour and 40 minute marks. But will process those too as null
            # tests.
            num_sec = 300
            for time in range((min(times) // num_sec + 1) * num_sec, (max(times) // num_sec + 1) * num_sec, num_sec):
                run_one(ant, folder_name, time, failure_summary)