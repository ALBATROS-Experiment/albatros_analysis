import numpy as np
import sys, os

sys.path.insert(0, os.path.expanduser("~"))

from albatros_analysis.scripts.ionosonde.data import process_from_data
from albatros_analysis.scripts.ionosonde.ionogram import process_and_plot

for ant in [2, 3, 7]:
    folder_names = os.listdir(f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/")

    for folder_name in folder_names:
        file_names = os.listdir(f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/{folder_name}")

        times = []
        
        for f_name in file_names:
            if f_name[-4:] == ".raw":
                    times.append(int(f_name[:-4]))

        # The ionosonde broadcasts every 5 minutes, with the exception of the hour and 40 minute marks
        # But will process those too as null tests
        for time in range((min(times) // 600 + 1) * 600, (max(times) // 600 + 1) * 600, 600):
            default_args.start_time = time - 5
            default_args.corr_time = 15
            default_args.baseband_dir = f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/"
            default_args.out_dir = f"/scratch/mayas/ionograms_to_share/{time}/mars{ant}/"
            process_from_data(default_args)