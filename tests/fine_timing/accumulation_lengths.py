import numpy as np 
import json
import h5py
import os

'''
Want to validate fine timing by looking at the generated timing solutions for different accumulation lengths.
In principle, they should give something internally consistent.
Done for 256, 512, 1024, 2048 spectra of accumulation length.

'''


#if you want to re-run the data
crunch_numbers = True

batch_start_ts = 1753200150

path_scripts = f'/home/thomasb/albatros_analysis/scripts/orbcomm'
path_batches = f'/scratch/thomasb/batch_{batch_start_ts}_testing'

names = ['acclen/256', 'acclen/512', 'acclen/1024', 'acclen/2048']

# same UTC fit for all
UTC_fit = {"fit": {"UTC_per_spec": 1.638402806904218e-05, "UTC_offset": 1753200128.4140258}}

# same pulse for all
pulse = [{"antenna": "MARS2","sat": 57166,"channel": 3,"t_start": 1753204614,"t_end": 1753204909,"start_specnum": 273770916}]

# run both discrep and fine timing for both cases. 
if crunch_numbers:
    for name in names:
        #make directory name
        path_run = os.path.join(path_batches, name)
        os.makedirs(path_run, exist_ok=True)

        path_discrep = os.path.join(path_run, 'timing_discrepancies')
        os.makedirs(path_discrep, exist_ok=True)
        with open(os.path.join(path_discrep, 'times_all.json'), 'w') as f:
            json.dump(UTC_fit, f, indent=4)

        path_data = os.path.join(path_run, 'data')
        os.makedirs(path_data, exist_ok=True)
        with open(os.path.join(path_data, 'pulses.json'), 'w') as f:
            json.dump(pulse, f, indent=4)

        # get discrep
        subprocess.run(["python", 
            f"{os.path.join(path_scripts, 'get_batch_discrepancies.py')}",
            f"{os.path.join(path_run, 'config.json')}",
            "-t", f"{testing}"
        ], check=True)

        # get finetiming
        subprocess.run(["python", 
            f"{os.path.join(path_scripts, 'get_batch_finetiming.py')}",
            f"{os.path.join(path_run, 'config.json')}",
            "-t", f"{testing}"
        ], check=True)

