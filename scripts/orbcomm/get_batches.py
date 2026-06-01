import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime
from albatros_analysis.src.utils import baseband_utils as butils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("interval_start", type=int)
    parser.add_argument("interval_end", type=int)
    parser.add_argument('ant_idxs', type=list)
    args = parser.parse_args()


args.interval_start, args.interval_end = interval_start, interval_end

ant_path_list = [
    '/project/rrg-sievers/albatros/mars/202507/mars1/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars2/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars3/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars4/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars5/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars6/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars7/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars8/baseband'
]

coords = [
    [79.41716147, -90.76723869, 187.9577],
    [79.41719805, -90.75873919, 183.0684],
    [79.38845641, -91.01920296, 25.1938],
    [79.41830257, -90.66739545, 59.6242],
    [79.39798424, -90.79984241, 41.699],
    [79.41147412, -90.69526613, 31.6314],
    [79.44375769, -90.71820263, 414.9131],
]

names_file=['ALB1','ALB2','ALB3','ALB4','ALB5','ALB6','ALB7','ALB8']
names_actual=["Antenna 1","Antenna 2","Antenna 3","Antenna 4","Antenna 5","Antenna 6","Antenna 7","Antenna 8"]
nant = len(names_file)

#get_present_files returns an array that tells you when files are present
arr, fig = butils.get_present_files(interval_start, interval_end, ant_path_list)
fig.savefig(f'/file_presence_{interval_start}_{interval_end}.png')

#runs give you timestamps within which there is active data from every desired antenna
runs = butils.get_simul_files(arr, interval_start, 10, ant_idxs)
print(runs)


sys.exit()

#make the config files
for i, (t_start, t_end) in enumerate(runs):
    file = {}

    antennas = {}
    for j in range(nant):
        antennas[names_file[j]] = {}
        antennas[names_file[j]]["name"] = names_actual[j]
        antennas[names_file[j]]["path"] = ant_path_list[j]
        antennas[names_file[j]]["coordinates"] = coords[j]
    file['antennas'] = antennas
    file["correlation"] = {
        "start_timestamp": t_start,
        "end_timestamp": t_end,
        "vis_acclen": 30000,
        "coarse_acclen": 3000000,
        "osamp": 64,
        "pfb_size": 65536,
        "new_acclen": 1024
    }
    file["frequency"] = {
        "start_channel": 1834,
        "end_channel": 1852
    }
    print(file)
    
    # dump the config files
    with open(os.path.join('/home/thomasb/albatros_analysis/scripts/orbcomm/config', f'config_nov25_batch{i+1}.json'), 'w') as f:
        json.dump(file, f, indent=4)