import numpy as np
import cupy as cp
import numba as nb
import healpy as hp
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
import time
import os
import sys
import rfitools

@nb.njit(parallel=True)
def geo_delay(bls,alt,az):
    npix = alt.shape[0]
    nbl = bls.shape[0] #each row is x, y, z
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

@nb.njit(parallel=True)
def get_map(data,freqs,delays,npix):
    print(data.shape, freqs.shape, delays.shape)
    map1 = np.zeros(npix, dtype=np.complex128)
    nfreq, nbl = data.shape
    N_vis = nfreq * nbl
    for p in nb.prange(npix):
        pixel_sum = 0j
        for f in range(nfreq):
            nu = freqs[f]
            for b in range(nbl):
                tau = delays[p, b] #delay shape is npix, nbl for CPU

                # Calculate the fringe factor for this specific visibility
                fringe = np.exp(2j * np.pi * nu * tau)
                
                # Accumulate the dot product
                pixel_sum += fringe * data[f, b]
        
        # Calculate the mean and assign to the pixel map
        map1[p] = pixel_sum / N_vis
    return map1

module = cp.RawModule(path='/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/ditymapmaker.ptx')
dirty_map_kernel = module.get_function('dirty_map_kernel')

path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/vis_dump.npz'
NSIDE = 1024

coords = {
        0: [79.417161473, -90.767238685, 187.9577], #MARS1
        1: [79.417198047, -90.758739192, 183.0684], #MARS2
        2: [79.388456412, -91.019202963, 25.1938], #CSA
        3: [79.418302573, -90.667395452, 59.6242], #Mars 5
        4: [79.397984238, -90.799842408, 41.6994], #Mars 6
        5: [79.411474117, -90.695266129, 31.6314], #Mars 7
        6: [79.443757694, -90.718202634, 414.9131] #MARS8
    }
antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}


with np.load(path_data) as f:
    vis=f['vis']
    cnt=f['cnt']
    tstart=f['tstart']
    deltat=f['deltat']
    deltaf=f['deltaf']

print('vis shape', vis.shape)
vis_avg = np.mean(vis,axis=0)
cnt_avg = np.sum(cnt,axis=0)

fstart = 360*250e6/4096
nant=7
triu_idx=np.triu_indices(nant,k=1) #upper triangle except main diag
nbl = len(triu_idx[0])
print(nbl, triu_idx)

# get number of pixels and find out pixel resolution on sky
NPIX=hp.nside2npix(NSIDE)
print("resol", hp.nside2resol(NSIDE,arcmin=True), "npix", NPIX)

#define reference antenna (MARS1) as location ant0
ant0=EarthLocation.from_geodetic(lat=coords[0][0], lon=coords[0][1], height=coords[0][2])

#get ra/dec coords for all pixels
co_dec,ra=hp.pix2ang(NSIDE,np.arange(NPIX))
dec=np.pi/2-co_dec
src = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')

#for fixed time index get the observation time
t_idx=0
obstime=Time(tstart+t_idx*deltat+deltat/2,format="unix",scale="utc")

#get altaz coordinates for each pixel at fixed observation time
print("tstart", obstime.unix, "generating pixel alt az")
t1=time.time()
altaz = src.transform_to(AltAz(location=ant0,obstime=obstime))
t2=time.time()
print("altaz call took", t2-t1)

# get all baselines and corresponding baseline delays
bl_enus = rfitools.get_all_bls(coords,np.arange(len(coords)))
delays = geo_delay(bl_enus, altaz.alt.value, altaz.az.value)

# take out MARS1-MARS2 (too much rfi)
good_bls = np.arange(1,nbl)
print(good_bls)

# get frequency array, good frequency array
freqs = np.arange(vis.shape[1]) * deltaf + fstart #Hz
#good_freq_idx = np.bitwise_and.reduce(cnt[t_idx,:, good_bls]>0, axis=0) #single time
good_freq_idx = np.bitwise_and.reduce(cnt_avg[:,good_bls]>10, axis=1) # averaged
good_freqs = freqs[good_freq_idx].copy()
print("num good freq", len(good_freqs))
nfreq = len(good_freqs)

# CPU version
#==================
print('RUNNING CPU VERSION')
delaysT = delays[good_bls,:].T.copy() #exclude MARS1-2

#single time
# data = vis[t_idx, :, :]          # shape (nfreq, nbl)
# data = data[:, good_bls]         # select baselines
# data = data[good_freq_idx, :]    # select frequencies
# data = data.copy()
#averaged
data = vis_avg[:, good_bls][good_freq_idx, :].copy()

_ = get_map(data,good_freqs,delaysT,100)

t1=time.time()
d_map = get_map(data,good_freqs,delaysT,NPIX)
t2=time.time()
print("cpu time", t2-t1)

np.savez('/scratch/thomasb/map_test_cpu.npz', map = d_map)
sys.exit()


# GPU setup
num_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Total GPUs: {num_gpus}")
current_id = cp.cuda.runtime.getDevice()
print(f"Active GPU ID: {current_id}")


# BENCHMARKING
#=====================
# print('RUNNING BENCHMARK GPU VERSION')
# nbl = 27
# nfreq = 300
# d_vis = cp.random.randn(nbl*nfreq).reshape(nbl,nfreq)
# mydelays = delays[good_bls, :]
# d_delays = cp.asarray(mydelays, dtype='float32')
# d_delays = cp.vstack([d_delays, d_delays[:nbl-d_delays.shape[0],:]])
# d_freqs = cp.linspace(20e6,22e6,nfreq)

# REAL DATA
#=====================
print('RUNNING REAL GPU VERSION')
mydelays = delays[good_bls, :]
d_vis = cp.asarray(vis[t_idx, good_freq_idx, 1:], dtype=cp.complex64).T.copy()
d_delays = cp.asarray(mydelays,dtype=cp.float32)
d_freqs = cp.asarray(good_freqs,dtype=cp.float64)

# MAKE MAPS
#=====================
d_map = cp.zeros(NPIX,dtype='float32')

print('VIS SHAPE', d_vis.shape)
print(d_vis.flags)
print('FIRST VIS VAL', d_vis[0,0], '\n')

print('FREQS SHAPE', d_freqs.shape)
print(d_freqs.flags)
print('FREQS VALUE', d_freqs[0], '\n')

print('DELAYS SHAPE', d_delays.shape)
print(d_delays.flags)
print('DELAYS VALUE', d_delays[0], '\n')

print('MAP SHAPE', d_map.shape)
print(d_map.flags)
print('MAPS VALUE', d_map[0])

# sys.exit()
threads_per_block = 256
blocks_per_grid = (NPIX + threads_per_block - 1) // threads_per_block

# run the kernel
print('starting kernel run')
dirty_map_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (d_delays, d_vis, d_freqs, d_map, 20, nfreq, NPIX)
    )
print(d_map.shape)
print(type(d_map))
print(d_map[0])
d_map = cp.asnumpy(d_map)
print(type(d_map))
print(d_map[0])
np.savez('/scratch/thomasb/map_test.npz', d_map)

# start_event = cp.cuda.Event()
# stop_event = cp.cuda.Event()

# flop_tot = 50 * nbl * nfreq * NPIX + 10*NPIX
# for i in range(10):
#     start_event.record()

#     dirty_map_kernel(
#         (blocks_per_grid,), (threads_per_block,),
#         (d_delays, d_vis, d_freqs, d_map, 20, nfreq, NPIX)
#     )
#     stop_event.record()
#     stop_event.synchronize()
#     elapsed_s = cp.cuda.get_elapsed_time(start_event, stop_event)/1000
#     print(f"GPU time: {elapsed_s:.3f} seconds, roughly {flop_tot/elapsed_s/1e12 :.2f} TFLOPS")