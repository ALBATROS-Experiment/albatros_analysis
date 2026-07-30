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

# module = cp.RawModule(path='/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/dirtymapmaker.ptx')
# dirty_map_kernel = module.get_function('dirty_map_kernel')



# @nb.njit(parallel=True)
# def get_map(data,freqs,delays,npix):
#     print(data.shape, freqs.shape, delays.shape)
#     map1 = np.zeros(npix, dtype=np.complex128)
#     nfreq, nbl = data.shape
#     N_vis = nfreq * nbl
#     for p in nb.prange(npix):
#         pixel_sum = 0j
#         for f in range(nfreq):
#             nu = freqs[f]
#             for b in range(nbl):
#                 tau = delays[p, b] #delay shape is npix, nbl for CPU

#                 # Calculate the fringe factor for this specific visibility
#                 fringe = np.exp(-2j * np.pi * nu * tau)
                
#                 # Accumulate the dot product
#                 pixel_sum += fringe * data[f, b]
#         # Calculate the mean and assign to the pixel map
#         map1[p] = pixel_sum / N_vis
#     return map1

# @nb.njit(parallel=True)
# def geo_delay(bls,alt,az):
#     npix = alt.shape[0]
#     nbl = bls.shape[0] #each row is x, y, z
#     delays = np.empty((nbl,npix),dtype='float64')
#     c=299792458
#     for pix in nb.prange(npix):
#         cos_alt = np.cos(np.pi * alt[pix] / 180)
#         sin_alt = np.sin(np.pi * alt[pix] / 180)
#         cos_az = np.cos(np.pi * az[pix] / 180)
#         sin_az = np.sin(np.pi * az[pix] / 180)
#         for bl in range(nbl):
#             ea,no,up = bls[bl]
#             p = 0
#             p += sin_az*cos_alt * ea #east
#             p += cos_az*cos_alt * no #north
#             p += sin_alt * up #up
#             delays[bl, pix] = p/c
#     return delays

# with np.load('vis_dump.npz') as f:
#     vis=f['vis']
#     cnt=f['cnt']
#     tstart=f['tstart']
#     deltat=f['deltat']
#     deltaf=f['deltaf']


# fstart = 360*250e6/4096

# nant=7
# triu_idx=np.triu_indices(nant,k=1)
# nbl = len(triu_idx[0])

# coords = {
#         0: [79.417161473, -90.767238685, 187.9577], #MARS1
#         1: [79.417198047, -90.758739192, 183.0684], #MARS2
#         2: [79.388456412, -91.019202963, 25.1938], #CSA
#         3: [79.418302573, -90.667395452, 59.6242], #Mars 5
#         4: [79.397984238, -90.799842408, 41.6994], #Mars 6
#         5: [79.411474117, -90.695266129, 31.6314], #Mars 7
#         6: [79.443757694, -90.718202634, 414.9131] #MARS8
#     }
# antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}
# ant0=EarthLocation.from_geodetic(lat=coords[0][0], lon=coords[0][1], height=coords[0][2])

# NSIDE=1024

# NPIX=hp.nside2npix(NSIDE)
# print("resol", hp.nside2resol(NSIDE,arcmin=True), "npix", NPIX)
# co_dec,ra=hp.pix2ang(NSIDE,np.arange(NPIX))
# dec=np.pi/2-co_dec

# t_idx=0
# print("tstart", tstart, "generating pixel alt az")
# obstime=Time(tstart+t_idx*deltat+deltat/2,format="unix",scale="utc")
# src = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')

# t1=time.time()
# altaz =src.transform_to(AltAz(location=ant0,obstime=obstime))
# t2=time.time()
# print("altaz call took", t2-t1)

# from concurrent.futures import ThreadPoolExecutor

# # times = Time(tstart+np.arange(0,100),format="unix",scale="utc")
# # def worker(sl):
# #     return src[sl].transform_to(AltAz(location=ant0,obstime=obstime))

# # ncpu = os.cpu_count()//2
# # chunks = np.array_split(np.arange(len(src)), ncpu)
# # for i in range(10):
# #     t1=time.time()
# #     with ThreadPoolExecutor(ncpu) as ex:
# #         out = list(ex.map(worker, chunks))
# #     t2=time.time()
# #     print('parallel time', t2-t1)


# bl_enus = rfitools.get_all_bls(coords,np.arange(len(coords)))
# delays = geo_delay(bl_enus, altaz.alt.value, altaz.az.value)

# # good_bls = [2,3,4,7,8,9,10,16,17,18]
# # good_bls = [5,10,17,19,20]
# good_bls = np.arange(1,nbl)
# print(good_bls)

# freqs = np.arange(vis.shape[1]) * deltaf + fstart #Hz
# good_freq_idx = np.bitwise_and.reduce(cnt[t_idx,:,1:]>0, axis=1) #across all baselines, excluding MARS1-2 because we wont use it
# good_freqs = freqs[good_freq_idx].copy()
# print("num good freq", len(good_freqs))
# nfreq = len(good_freqs)
# delaysT = delays[1:,:].T.copy() #exclude MARS1-2
# data = vis[t_idx, good_freq_idx, 1:].copy() #exclude MARS1-2

# _ = get_map(data,good_freqs,delaysT,100)

# t1=time.time()
# map1 = get_map(data,good_freqs,delaysT,NPIX)
# t2=time.time()
# print("cpu time", t2-t1)


#GPU CALLS BELOW
#---------------------_#
num_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Total GPUs: {num_gpus}")
current_id = cp.cuda.runtime.getDevice()
print(f"Active GPU ID: {current_id}")
dummy = cp.random.randn(250000 * 25000).reshape(250000,25000)
print("dummy size", dummy.nbytes/1e9)
print(cp.cuda.runtime.deviceCanAccessPeer(1, 0))
with cp.cuda.Device(1):
    current_id = cp.cuda.runtime.getDevice()
    print(f"Active GPU ID: {current_id}")
    cp.cuda.runtime.deviceEnablePeerAccess(0) #this gives me an additional 10 GB/s not sure why. OK so 6 lanes per p2p pair. each lane supports 50 GB/s
    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    start_event.record()
    dummy1 = cp.asarray(dummy)
    stop_event.record()
    stop_event.synchronize()
    elapsed_s = cp.cuda.get_elapsed_time(start_event, stop_event)/1000
    print(f"GPU P2P transfer time: {elapsed_s:.3f} seconds, roughly {dummy.nbytes/elapsed_s/1e9 :.2f} GB/s")
t1=time.time()
numpy = cp.asnumpy(dummy)
t2=time.time()
print("D2H", dummy.nbytes/(t2-t1)/1e9, "GB/s")
sys.exit()

mydelays = delays[1:,:].copy()
# d_vis = cp.asarray(data, dtype='complex64')
# d_vis = d_vis.T.copy()
# d_freqs = cp.asarray(good_freqs, dtype='float64')
d_delays = cp.asarray(mydelays, dtype='float32')
nbl = 27
nfreq = 300
d_vis = cp.random.randn(nbl*nfreq).reshape(nbl,nfreq)
d_delays = cp.vstack([d_delays, d_delays[:nbl-d_delays.shape[0],:]])
d_freqs = cp.linspace(20e6,22e6,nfreq)
d_map = cp.zeros(NPIX,dtype='float32')
print(d_vis.shape, d_vis.flags)
print(d_freqs.shape, d_freqs.flags)
print(d_delays.shape, d_delays.flags)
print(d_map.shape, d_map.flags)
# sys.exit()
threads_per_block = 256
blocks_per_grid = (NPIX + threads_per_block - 1) // threads_per_block


dirty_map_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (d_delays, d_vis, d_freqs, d_map, 20, nfreq, NPIX)
    )


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