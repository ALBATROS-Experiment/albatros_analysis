import numpy as np
import cupy as cp
import numba as nb
import rfitools
import healpy as hp
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
import time
import os
import sys



module = cp.RawModule(path='/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/dirtymapmaker.ptx')
dirty_map_kernel = module.get_function('dirty_map_kernel')

with np.load('/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/test_viz_onetime.npz') as f:
    vis=f['vis']
    delays=f['delays']
    freqs=f['freqs']
    
NSIDE=256
NPIX=hp.nside2npix(NSIDE)
print("vis shape", vis.shape)
myvis = vis[0, :, :].copy()
d_delays = cp.asarray(delays, dtype='float32')
nfreq = len(freqs)
nbl = vis.shape[1]
d_vis = cp.asarray(myvis, dtype='complex64')
d_freqs = cp.asarray(freqs, dtype='float32')
d_map = cp.zeros(NPIX,dtype='float32')

print(d_vis.shape,"vis", d_vis.flags)
print(d_freqs.shape,"freqs", d_freqs.flags)
print(d_delays.shape,"delays", d_delays.flags)
print(d_map.shape,"map", d_map.flags)
# print(freqs)
# sys.exit()
threads_per_block = 256
blocks_per_grid = (NPIX + threads_per_block - 1) // threads_per_block

niter=100
start_event = cp.cuda.Event()
stop_event = cp.cuda.Event()
for i in range(niter):
    start_event.record()
    dirty_map_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (d_delays, d_vis, d_freqs, d_map, nbl, nfreq, NPIX)
        )
    stop_event.record()
    stop_event.synchronize()
    elapsed_s = cp.cuda.get_elapsed_time(start_event, stop_event)/1000

    print(f"GPU time: {elapsed_s:.3f} seconds")

sys.exit()
print(map1)
np.save("/scratch/thomasb/mapmaking_dumps/average_map_mohan.npy", map1)