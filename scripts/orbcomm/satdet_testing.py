import os
import sys
import time
from os import path
sys.path.insert(0, "/home/thomasb")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
import cupy as cp
import json
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import sat_utils as su
import figures as fgs


config_file = 'config.json'
with open(config_file, "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    ant_names = []

    print("\nAntenna Details:")
    for i, (ant, details) in enumerate(config["antennas"].items()):
        print(ant, details)
        coords.append(details['coordinates'])
        dir_parents.append(details["path"])
        ant_names.append(details["name"])

    global_start_t = config["correlation"]["start_timestamp"] #global as it is in unix time, reference frame
    global_end_t = config["correlation"]["end_timestamp"]
    c_acclen = config["correlation"]["coarse_acclen"]
    v_acclen = config["correlation"]["vis_acclen"]


passes = [[[0, 61], [0, 1]], 
          [[61, 97], [1]], 
          [[168, 281], [2]], 
          [[680, 748], [4]], 
          [[832, 855], [5]], 
          [[855, 917], [3, 5]], 
          [[917+10, 963], [3]], 
          [[1157, 1191], [0]], 
          [[1191, 1268], [0, 1]], 
          [[1268, 1297], [1]], 
          [[1376, 1440], [2]]]

bline_idx = 5
pulse_idx = 6
out_dir = '/scratch/thomasb'

temp_satmap = ['Uncorrected']
times, sats_present = passes[pulse_idx]
for sat in sats_present:
    temp_satmap.append(sat)
print(temp_satmap)

t1, t2 = (5*times[0])+global_start_t, (5*times[1])+global_start_t

ra_path, nra_path = dir_parents[0], dir_parents[bline_idx]
ra_coords, nra_coords = coords[0], coords[bline_idx]

tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
N = int(2* c_acclen)
dN = int(10**5)

satlist = [28654,25338,33591,57166,59051,44387]
satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
assert min(satlist) > len(satlist)
for i, sat_ID in enumerate(satlist):
    satmap[i] = sat_ID
    satmap[sat_ID] = i


try:
    files_ra, idx_ra = butils.get_init_info(t1, t2, ra_path)
    files_nra, idx_nra = butils.get_init_info(t1, t2, nra_path)
except Exception as e:
    print(e)
    print(f"WARNING: skipping pass. MISTAKE IN FILE CHECKER!!")
    #continue

print("Setting Antenna as BFI Objects", '\n')

# Set up the number of channels we look through
channels = np.asarray(bdc.get_header(files_ra[0])["channels"],dtype='int64')
chanstart = np.where(channels == 1834)[0][0]
chanend = np.where(channels == 1852)[0][0]
nchans = chanend - chanstart

ra = bdc.BasebandFileIterator(
                files_ra,
                0,
                idx_ra,
                c_acclen,
                None,
                chanstart=chanstart,
                chanend=chanend,
                type="float",
            )
nra = bdc.BasebandFileIterator(
    files_nra,
    0,
    idx_nra,
    c_acclen,
    None,
    chanstart=chanstart,
    chanend=chanend,
    type="float",
)


p0_ra = cp.zeros((c_acclen, nchans), dtype="complex64") #remember that BDC returns complex64. wanna do phase-centering in 128.
p0_nra = cp.zeros((c_acclen, nchans), dtype="complex64")
ra_start = ra.spec_num_start
nra_start = nra.spec_num_start
for i, (chunk_ra, chunk_nra) in enumerate(zip(ra, nra)):
    perc_missing_ra = (1 - len(chunk_ra["specnums"]) / c_acclen) * 100
    perc_missing_nra = (1 - len(chunk_nra["specnums"]) / c_acclen) * 100
    print("missing a1", perc_missing_ra, "missing a2", perc_missing_nra)
    if perc_missing_ra > 10 or perc_missing_nra > 10:
        ra_start = ra.spec_num_start
        nra_start = nra.spec_num_start
        continue
    
    bdc.make_continuous_gpu(chunk_ra['pol0'],chunk_ra['specnums']-ra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_ra)
    bdc.make_continuous_gpu(chunk_nra['pol0'],chunk_nra['specnums']-nra_start,np.arange(nchans),c_acclen,nchans=nchans, out=p0_nra)
    break


center = c_acclen
dN = 100000

cx = su.get_cxcorr_many_sats(p0_ra,
                             p0_nra, 
                             tle_path, 
                             [t1, t2], 
                             sats_present,
                             satmap,
                             [ra_coords, nra_coords],
                             N,
                             dN)

for idx, sat in enumerate(temp_satmap):
    cxfig = su.make_cxcorr_plot(cx[idx])
    cxfig.savefig(os.path.join(out_dir, f'sat_plot_testing_{sat}_{bline_idx}_newplot.jpg'))
    