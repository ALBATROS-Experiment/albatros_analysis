import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import h5py
import json

'''
The aim of this test is to demonstrate that the pipeline yields similar results for different pulse start time and end times.
The idea is that it is robust to slightly shifted regions, and that the underlying signal is properly identified and fitted for.
This is done by generating three satellite pulses on the same data, but with different UTC start and end times.
Using a previously determiend UTC map, for the different pulses we examine:
    - fit to UTC time (should be similar fit)
    - cutting to low phase noise region (should be same region)
    - phase unwrapping plots (same signal should look similar)
    - tau values
Both scripts generate extensive debugplots, so it may also be useful to go into the directories to check they all look alright.
'''

batch_start_ts = 1753200150
testing = 'pulse_times'

#define the actual times you want to check
pstart1, pend1 = 1753204500, 1753205100  #big gap
pstart2, pend2 = 1753204565, 1753205056  #original one
pstart3, pend3 = 1753204620, 1753204900  #zoomed in

#define the paths to where everything lives
path_scripts = '/home/thomasb/albatros_analysis/scripts/orbcomm'
path_config = os.path.join(path_scripts, 'config/config_batch2.json')

path_batch = f'/scratch/thomasb/batch_{batch_start_ts}_testing/{testing}'
os.makedirs(path_batch, exist_ok=True)
path_data = os.path.join(path_batch, 'data')
os.makedirs(path_data, exist_ok=True)
path_finetiming = os.path.join(path_batch, 'fine_timing')
os.makedirs(path_finetiming, exist_ok=True)
path_discrepancies = os.path.join(path_batch, 'timing_discrepancies')
os.makedirs(path_discrepancies, exist_ok=True)



#make and save articifial pulse list
pulse_list = [
    {"antenna": "MARS2","sat": 57166,"channel": 3,"t_start": pstart1,"t_end": pend1},
    {"antenna": "MARS2","sat": 57166,"channel": 3,"t_start": pstart2,"t_end": pend2},
    {"antenna": "MARS2","sat": 57166,"channel": 3,"t_start": pstart3,"t_end": pend3},
]
with open(os.path.join(path_data, 'pulses.json'), 'w') as f:
    json.dump(pulse_list, f, indent=4)

#make and save known UTC fit parameters
UTC_fit = {
    "fit": {"UTC_per_spec": 1.638402806904218e-05,"UTC_offset": 1753200128.4140258}
    }

with open(os.path.join(path_discrepancies, 'times_all.json'), 'w') as f:
    json.dump(UTC_fit, f, indent=4)

#run timing discrepancies
subprocess.run(["python", 
    f"{os.path.join(path_scripts, 'get_batch_discrepancies.py')}",
    f"{path_config}",
    "-t", f"{testing}"
], check=True)

#run fine timing
subprocess.run(["python", 
    f"{os.path.join(path_scripts, 'get_batch_finetiming.py')}",
    f"{path_config}",
    "-t", f"{testing}"
], check=True)

#check UTC differences
starting_spectra = []
with open(os.path.join(path_discrepancies, 'times_all.json'), 'r') as f:
    UTC_fits = json.load(f)
    for key, value in UTC_fits.items():
        print(key)

#check delay differences
taus_all = []
with h5py.File(path_taus_testing, 'r') as f:
    for name, obj in f.items():
        taus_old = obj['taus'][:]
        taus_all.append(taus_old)

