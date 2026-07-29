import numpy as np
import os
import json
import subprocess
import h5py

'''
The aim of this test is to make sure we get the same fine timing solution when generating re-pfbed data for the old and new streaming methods.

We isolate a single pulse, and use the old re-pfb computed version against the new re-pfb version. The only difference is the re-pfbed data itself.

We store everything in batch_1753200150_testing: the old version goes in old_repfb, the new in new_repfb. 

We use the known UTC-spectrum map so we don't have to compute a ton of data for no reason.

This notebook runs through both computations and then checks the differences, returning some plots to see.

All the specific test outputs are in the new data's directory: new_pfb. The usual debugplots are stored in the respective directories.

If you want to run this completely fresh, delete everything but the old_repfb data.
'''
#if you want to re-run the data
crunch_numbers = True

batch_start_ts = 1753200150

path_config = f'/home/thomasb/albatros_analysis/scripts/orbcomm/config/config_batch2.json'
path_scripts = f'/home/thomasb/albatros_analysis/scripts/orbcomm'
path_batches = f'/scratch/thomasb/batch_{batch_start_ts}_testing'

#names for the two cases
names = ['old_repfb', 'new_repfb']

#same fit for both
UTC_fit = {"fit": {"UTC_per_spec": 1.638402806904218e-05, "UTC_offset": 1753200128.4140258}}

#data already generated for the old case so as extra check, make it generate the start specnum for new repfb-er.
pulse = [{"antenna": "MARS2","sat": 57166,"channel": 3,"t_start": 1753204614,"t_end": 1753204909,"start_specnum": 273770916}]

# run both discrep and fine timing for both cases. 
if crunch_numbers:
    for i in range(2):
    
        testing = names[i]
        path_run = os.path.join(path_batches, testing)

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
            f"{path_config}",
            "-t", f"{testing}"
        ], check=True)

        # get finetiming
        subprocess.run(["python", 
            f"{os.path.join(path_scripts, 'get_batch_finetiming.py')}",
            f"{path_config}",
            "-t", f"{testing}"
        ], check=True)


# quick numerical checks
# strongly encourage to go look through the debugplots for visual confirmation


# starting specnum
with open(os.path.join(path_batches, f'{names[0]}/data/pulses.json'), 'r') as f:
    ts = json.load(f)
    starting_specnum_old = ts[0]['start_specnum']

with open(os.path.join(path_batches, f'{names[1]}/data/pulses.json'), 'r') as f:
    ts = json.load(f)
    starting_specnum_new = ts[0]['start_specnum']

print('difference in specnum', np.abs(starting_specnum_new - starting_specnum_old))

# check the discrepancy fit
with open(os.path.join(path_batches, f'{names[0]}/timing_discrepancies/times_all.json'), 'r') as f:
    fits = json.load(f)
    data_key = next(k for k in fits if k != "fit")
    data_fit = fits[data_key]

    chisq_old = data_fit['chisq_fitted']
    offset_old = data_fit['offset_fitted']


with open(os.path.join(path_batches, f'{names[1]}/timing_discrepancies/times_all.json'), 'r') as f:
    fits = json.load(f)
    data_key = next(k for k in fits if k != "fit")
    data_fit = fits[data_key]

    chisq_new = data_fit['chisq_fitted']
    offset_new = data_fit['offset_fitted']

print('difference in chisq', np.abs(chisq_new - chisq_old))
print('difference in offset', 0.001 * np.abs(offset_new - offset_old))

# check fine_timing
with h5py.File(os.path.join(path_batches, f'{names[0]}/fine_timing/timing_solution.h5'), 'r') as f:
            for name, obj in f.items():
                taus_old = obj['taus'][:]

with h5py.File(os.path.join(path_batches, f'{names[1]}/fine_timing/timing_solution.h5'), 'r') as f:
            for name, obj in f.items():
                taus_new = obj['taus'][:]

print('biggest difference in taus', np.max(np.abs(taus_old-taus_new)))