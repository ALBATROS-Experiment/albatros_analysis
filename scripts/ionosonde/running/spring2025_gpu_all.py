import os, sys

sys.path.insert(0, os.path.expanduser("~"))

from albatros_analysis.scripts.ionosonde.running import running_utils

out_dir_root = "/scratch/mayas/ionograms_to_share_4bit"

for ant in [2, 3, 7]:
    baseband_dir = f"/scratch/mohanagr/drive{ant}_mars_spring2025/baseband/"
    running_utils.iterate_through_folders(ant, out_dir_root, baseband_dir)

# To only run for one antenna:
# python3 -c 'import os, sys; sys.path.insert(0, os.path.expanduser("~")); from albatros_analysis.scripts.ionosonde.running import running_utils; out_dir_root = "/scratch/mayas/ionograms_to_share_4bit"; baseband_dir = f"/scratch/mohanagr/drive2_mars_spring2025/baseband/"; running_utils.iterate_through_folders(2, out_dir_root, baseband_dir, max_num = 2)'