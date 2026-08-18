import copy

import logging
logger = logging.getLogger(__name__)

import os, sys

sys.path.insert(0, os.path.expanduser("~"))

from albatros_analysis.scripts.ionosonde.params import default_args
logger.info(default_args)
from albatros_analysis.scripts.ionosonde.data import process_from_data

def run_one(ant, folder_name, time, failure_summary, out_dir_root, baseband_dir):
    # Work on a fresh copy of the args for every run
    args = copy.deepcopy(default_args)
    args.start_time = time
    args.which_ant = ant
    args.baseband_dir = baseband_dir
    args.out_dir = f"{out_dir_root}/{folder_name}/{time}/mars{ant}/"

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

def iterate_through_folders(ant, out_dir_root, baseband_dir, num_sec = 300, max_num = None):
    summary_path = os.path.join(out_dir_root, "correlation_summary.csv")
    os.makedirs(out_dir_root, exist_ok = True)
    summary_exists = os.path.isfile(summary_path)

    with open(summary_path, "a") as failure_summary:
        # If the summary didn't previously exist, write the table headings
        if not summary_exists:
            failure_summary.write("Success?,MARS Station,Time,Exception\n")

        folder_names = os.listdir(baseband_dir)
        counter = 0

        for folder_name in folder_names:
            folder_path = os.path.join(baseband_dir, folder_name)
            file_names = os.listdir(folder_path)

            times = [int(f[:-4]) for f in file_names if f.endswith(".raw")]

            if not times:
                # Nothing to process in this folder, so just skip
                continue

            for time in range((min(times) // num_sec + 1) * num_sec, (max(times) // num_sec + 1) * num_sec, num_sec):
                run_one(ant, folder_name, time, failure_summary, out_dir_root, baseband_dir)
                counter += 1

                if max_num and (counter >= max_num):
                    return