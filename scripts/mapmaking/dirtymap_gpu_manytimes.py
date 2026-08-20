import cupy as cp
import os
import sys
import numba as nb
import healpy as hp
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import time
import rfitools
from concurrent.futures import ThreadPoolExecutor

@nb.njit(parallel=True)
def geo_delay(bls, alt, az):
    npix = alt.shape[0]
    nbl = bls.shape[0] #each row is x, y, z
    print("nbl", nbl, "npix", npix)
    delays = np.empty((nbl,npix),dtype='float64')
    c=299792458
    for pix in nb.prange(npix):
        cos_alt = np.cos(np.pi * alt[pix] / 180)
        sin_alt = np.sin(np.pi * alt[pix] / 180)
        cos_az = np.cos(np.pi * az[pix] / 180)
        sin_az = np.sin(np.pi * az[pix] / 180)
        for bl in range(nbl):
            ea,no,up = bls[bl]
            p = 0
            p += sin_az*cos_alt * ea #east
            p += cos_az*cos_alt * no #north
            p += sin_alt * up #up
            delays[bl, pix] = p/c
    return delays

path_kernel = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/dirtymapmaker.ptx'
module = cp.RawModule(path=path_kernel)
dirty_map_kernel = module.get_function('dirty_map_kernel')

#sim
#path_data = '/scratch/thomasb/mapmaking_dumps/simulation_vis.npz'

# old data path
path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/vis_dump.npz'

# new data path
#pidx = 0
#path_data = f'/scratch/thomasb/mapmaking_dumps/all_sats_science_band/satpass_{pidx}.npz'


# ============================================================
# Load Stuff
with np.load(path_data) as f:
    vis = f['vis']
    print("vis shape:", vis.shape)
    print('vis value', vis[0,0,0])

    # new version
    # mask = f['mask']
    # print("mask shape:", mask.shape)
    # times = f['times']
    # freqs = f['freqs']
    
    # # old version
    cnt = f['cnt']
    print("cnt shape:", cnt.shape)
    tstart = f['tstart']
    print("tstart:", tstart)
    deltat = f['deltat']
    print("deltat:", deltat)
    deltaf = f['deltaf']
    print("deltaf:", deltaf)

# for testing
# vis = np.mean(vis,axis=0)
# vis = vis[None, :, :]
# cnt = np.sum(cnt,axis=0)
# cnt = cnt[None, :, :]

#vis_gpu = cp.asarray((1-mask)*vis, dtype=cp.complex64)

#sys.exit()
# ============================================================
# Coords, etc

NSIDE = 512
compute = 'cpu'
coords = {
    0: [79.417161473, -90.767238685, 187.9577],   # MARS1
    1: [79.417198047, -90.758739192, 183.0684],   # MARS2
    2: [79.388456412, -91.019202963, 25.1938],    # MARS4
    3: [79.418302573, -90.667395452, 59.6242],    # MARS5
    4: [79.397984238, -90.799842408, 41.6994],    # MARS6
    5: [79.411474117, -90.695266129, 31.6314],    # MARS7
    6: [79.443757694, -90.718202634, 414.9131]    # MARS8
}

antmap = {0: "MARS1", 1: "MARS2", 2: "MARS4", 3: "MARS5", 4: "MARS6", 5: "MARS7", 6: "MARS8"}

# ============================================================
# Baseline Stuff

nant = 7
triu_idx = np.triu_indices(nant, k=1)
nbl = len(triu_idx[0])
print("total baselines:", nbl)

good_bls = np.arange(1, nbl) #hard-coded for now: remove MARS1-MARS2 because of bad RFI
nbl_good = len(good_bls)
print("using baselines:", good_bls)
print("number of baselines used:", nbl_good)

bl_enus = rfitools.get_all_bls(coords,np.arange(len(coords)))
print("baseline ENU shape:", bl_enus.shape)

# ============================================================
# pixel count
NPIX = hp.nside2npix(NSIDE)
print("HEALPix resolution:", hp.nside2resol(NSIDE, arcmin=True), "arcmin")
print("number of pixels:", NPIX)

# general pixel coordinates
co_dec, ra = hp.pix2ang(NSIDE,np.arange(NPIX))
dec = np.pi/2 - co_dec
src = SkyCoord(ra=ra * u.rad,dec=dec * u.rad,frame='icrs')

# reference antenna
ant0 = EarthLocation.from_geodetic(lat=coords[0][0], lon=coords[0][1], height=coords[0][2])

# ============================================================
# Frequencies

# Old Version
fstart = 360 * 250e6 / 4096
freqs = (np.arange(vis.shape[1]) * deltaf+ fstart)

# new version
# freqs = freqs*250e6/4096

print("frequency range:",freqs[0], "to",freqs[-1],"Hz")

# ============================================================
# Select visibility times

ntimes = vis.shape[0]
time_indices = np.arange(ntimes)
n_maps = len(time_indices)

print("total visibility times:", ntimes)
print("number of maps:", n_maps)

map_sum = np.zeros(NPIX, dtype=np.float32)
map_count = 0

# ============================================================
# GPU and CPU setup

num_gpus = cp.cuda.runtime.getDeviceCount()
print("number of GPUs:", num_gpus)

current_id = cp.cuda.runtime.getDevice()
print("active GPU:", current_id)

threads_per_block = 256
blocks_per_grid = (NPIX + threads_per_block - 1) // threads_per_block

# ============================================================
# Loop in Time
for map_idx, t_idx in enumerate(time_indices):

    print("\n==========================================")
    print(f"Mapping {map_idx + 1}/{n_maps}")
    print(f"(visibility {t_idx})") 

    total_start = time.time()

    #==Center time of visibility
    obstime = Time(tstart + t_idx * deltat + deltat/2, format="unix", scale="utc") # old version
    #obstime = Time(times[t_idx], format="unix", scale="utc") # new version
    print("center time:", obstime.unix)

    #==Pixel coords into alt/az
    t1 = time.time()
    altaz =src.transform_to(AltAz(location=ant0,obstime=obstime))
    t2 = time.time()
    print('altaz calculation:', t2-t1, 'seconds')

    #==Pixel delays
    t1 = time.time()
    delays = geo_delay(bl_enus, altaz.alt.value, altaz.az.value)
    t2 = time.time()
    print("delay calculation:", t2 - t1, "seconds")

    #(optional) save fringes
    # fringes = np.exp(2j * np.pi * delays * 21e6)
    # np.save('/scratch/thomasb/mapmaking_dumps/fringes1.npy', fringes)

    #== Good frequency indices (RFI stuff)
    #good_freq_idx = np.bitwise_and.reduce(mask[t_idx, :, good_bls] > 0, axis=0) # OLD VERSION (double check axes)
    #good_freq_idx = ~np.any(cnt[t_idx, :, good_bls],axis=0) # NEW VERSION
    #print('Good freq index shape', good_freq_idx.shape)
    #good_freqs = freqs[good_freq_idx]
    good_freqs = freqs
    nfreq = len(good_freqs)
    print('Number of good frequencies', nfreq)

    #== Normalization Factor
    norm = np.sum(1-mask[t_idx, 1:, :]) #exclude MARS1-MARS2
    print('full expected', np.prod(vis.shape[1:]))
    print('norm', norm)

    if norm == 0:
        print("Skipping visibility: no good frequencies")
        continue

    if compute == 'cpu':
        # CPU VERSION
        # ======================
        print('RUNNING CPU VERSION')
        delays_cpu = delays[good_bls, :].T.astype(np.float32, copy=True) #exclude MARS1-2
        #data_cpu = data.astype(np.complex64, copy=True)
        data_cpu = ((1-mask[t_idx, 1:, :]) * vis[t_idx, 1:, :]).T.astype(np.complex64, copy=True)
        freqs_cpu = good_freqs.astype(np.float32, copy=True)
    
        print('delays shape', delays_cpu.shape)
        print('delays dtype', delays_cpu.dtype)
        print('first delay term', delays_cpu[0, 0])
        print('vis shape', data_cpu.shape)
        print('vis dtype', data_cpu.dtype)
        print('first vis term', data_cpu[0, 0])
        print('freqs shape', freqs_cpu.shape)
        print('first freqs term', freqs_cpu[0])
        print('frequencies dtype', freqs_cpu.dtype)
        #sys.exit()

        # run mapmaker
        t1=time.time()
        map_cpu = rfitools.get_map(data_cpu, freqs_cpu, delays_cpu, NPIX)
        t2=time.time()
        print("cpu time", t2-t1)
    
    if compute == 'gpu':
        # GPU VERSION
        # ======================
        print('RUNNING GPU VERSION')

        vis_gpu = cp.asarray((1-mask)*vis, dtype=cp.complex64)
        
        # Transfer arrays to GPU
        t1 = time.time()
        d_delays = cp.asarray(delays, dtype=cp.float32)
        d_delays = d_delays[1:, :].copy()
        #d_vis = cp.asarray(data,dtype=cp.complex64).T.copy
        d_vis = vis_gpu[t_idx, 1:, :].copy()
        d_freqs = cp.asarray(good_freqs,dtype=cp.float32)
        t2 = time.time()
        d_map = cp.zeros(NPIX, dtype=cp.float32) # make GPU map
        print("GPU setup time:",t2 - t1,"seconds")

        print('delays shape', d_delays.shape)
        print('delays dtype', d_delays.dtype)
        print('first delay term', d_delays[0, 0])
        print('vis shape', d_vis.shape)
        print('first vis term', d_vis[0, 0])
        print('vis dtype', vis.dtype)
        print('freqs shape', d_freqs.shape)
        print('first frequency term', d_freqs[0])
        print('map shape', d_map.shape)
        print('nbl_good', nbl_good)
        print('nfreq', nfreq)
        print('NPIX', NPIX)
        #sys.exit()

        # Run kernel
        t1 = time.time()
        dirty_map_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (d_delays, d_vis, d_freqs, d_map, nbl_good, nfreq, NPIX)
        )

        # CUDA kernels are asynchronous
        cp.cuda.Stream.null.synchronize()
        t2 = time.time()
        print("GPU map calculation:",t2 - t1,"seconds")

        # Copy map back to CPU
        t1 = time.time()
        map_cpu = cp.asnumpy(d_map)
        t2 = time.time()
        print("GPU -> CPU:",t2 - t1,"seconds")

    # Add this visibility's map to running sum
    map_sum += map_cpu/norm
    map_count += 1

    total_end = time.time()
    print("TOTAL:",total_end - total_start,"seconds")

# ============================================================
# Final time average

print("\n==========================================")
print("Finished all maps.")
print("Number of maps averaged:", map_count)

avg_map = map_sum / map_count
print("average map shape:",avg_map.shape)

# ============================================================
# Save final map

#output_path = (f'/scratch/thomasb/mapmaking_dumps/sim_map_{compute}_nside{NSIDE}.npz')
output_path = (f'/scratch/thomasb/mapmaking_dumps/first_test_real_data/{compute}_sat{pidx}_nside{NSIDE}.npz')
#output_path = (f'/scratch/thomasb/mapmaking_dumps/test_map_{compute}_sat{pidx}.npz')
np.savez(
    output_path,
    avg_map=avg_map,
    NSIDE=NSIDE,
    n_maps=map_count
)

print("Saved results to:", output_path)