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
import tqdm

def map_many_times(dir_cosines: cp.ndarray, baselines: cp.ndarray, vis: cp.ndarray, norm: cp.ndarray, freqs: cp.ndarray, dt: float, npix: int, ntime: int) -> cp.ndarray:
    local_baselines = baselines.copy()
    nbl = baselines.shape[0]
    nfreq = freqs.shape[0]
    map_sum = cp.zeros(npix, dtype='float64')
    cos_dt = cp.cos(7.29211e-5 * dt)
    sin_dt = cp.sin(7.29211e-5 * dt)
    rotmat = cp.asarray([[cos_dt, sin_dt], [-sin_dt, cos_dt]], dtype='float32') # [bx2, by2] = [bx, by] @ rotmat

    threads_per_block = 256
    blocks_per_grid = (NPIX + threads_per_block - 1) // threads_per_block
    # ev1 = cp.cuda.Event()
    # ev2 = cp.cuda.Event()
    for t in range(ntime):

        print(f'Flagged in time {t} is {norm[t] / (vis.shape[1] * vis.shape[2]):.2f}', )
        
        if norm[t]>0:
            # print("baselines earlier", local_baselines[:, :2])
            map_temp = cp.zeros(npix,dtype='float32')
            delays = (local_baselines @ dir_cosines)/299792458.0  # (nbl, 3) @ (3, npix)

            # print("delays flags", delays.shape, delays.flags)
            # print("vis flags", vis[t, :, :].shape, vis[t, :, :].flags)
            #kernel call
            # ev1.record()
            dirty_map_kernel(
                (blocks_per_grid,),
                (threads_per_block,),
                (delays, vis[t, :, :], freqs, map_temp, nbl, nfreq, NPIX)
            )
            # ev2.record()
            # ev2.synchronize()
            # print(f"elapsed {cp.cuda.get_elapsed_time(ev1,ev2)/1000:.2f}s")
            map_sum[:] += map_temp/norm[t]
        
        #rotate all baselines' x,y by dphi for next timestep
        bl_xy = local_baselines[:,:2]
        bl_xy[:] = bl_xy @ rotmat
        # print("baselines later", local_baselines[:, :2]) # bls verified to change

    return map_sum/ntime

save = True

# common to real data and sim data

# ============================================================
# Coords, etc

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
# pixel count
NSIDE = 512
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
# Baseline Stuff

nant = len(coords.keys())
triu_idx = np.triu_indices(nant, k=1)
nbl = len(triu_idx[0])
bl_itrs = rfitools.get_all_bls(coords,np.arange(len(coords)), enu=False)
print("baseline itrs shape:", bl_itrs.shape)


path_kernel = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/dirtymapmaker.ptx'
module = cp.RawModule(path=path_kernel)
dirty_map_kernel = module.get_function('dirty_map_kernel')



### SIM STUFF ###

# old data path
# path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/vis_dump.npz'
# path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/test_viz_onetime.npz'
# path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/test_viz_ntime_200_dt_1s_df_3kHz.npz'
# path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/test_viz_ntime_500_dt_1s_df_3kHz.npz'
# path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/test_viz_ntime_186_dt_1s_df_1kHz.npz'
# with np.load(path_data) as f:
#     vis = f['vis']
#     freqs = f['freqs']
#     times = f['times']
# #     ntime = vis.shape[0]
# #     deltat = 0.5
# #     tstart = 1753264834.1802247
# #     times = np.arange(10)*deltat + tstart
# # #     tstart=f['tstart']
# # #     deltat=f['deltat']
# # #     deltaf=f['deltaf']
# # #     fstart = 360*250e6/4096
# # #     freqs = np.arange(vis.shape[1]) * deltaf + fstart #Hz
# # #     times = tstart + np.arange(vis.shape[0])*deltat + 0.5*deltat
# #     print("vis shape:", vis.shape)
# #     print(times)
# # sys.exit()

# vis_gpu = cp.asarray(vis, dtype=cp.complex64)

# ntime = vis_gpu.shape[0]
# nbl = vis_gpu.shape[1]
# nfreq = vis_gpu.shape[2]
# assert len(times) == ntime
# assert len(freqs) == nfreq
# # vis_gpu = cp.asarray(vis.transpose(0,2,1), dtype=cp.complex64) # for test summer data
# d_freqs = cp.asarray(freqs,dtype=cp.float32) # change to good_freqs later
# print("shape of freqs is", d_freqs.shape)

# #==Center time of visibility
# #obstime = Time(tstart + t_idx * deltat + deltat/2, format="unix", scale="utc") # old version
# obstime = Time(times[0], format="unix", scale="utc") # new version
# print("center time:", obstime.unix)

# #==Pixel coords into alt/az
# altaz =src.transform_to(AltAz(location=ant0,obstime=obstime))
# ha, dec = rfitools.azalt_to_hadec(altaz.az.rad, altaz.alt.rad, ant0.lat.rad) #apparent hadec
# gha = ha - ant0.lon.rad # HA at Greenwich
# l,m,n = np.cos(gha)*np.cos(dec), -np.sin(gha)*np.cos(dec), np.sin(dec)
# dir_cosines = cp.asarray([l, m, n], dtype='float32') #we'll keep these fixed and rotate baselines

# #== Normalization Factor
# norm = cp.ones(vis_gpu.shape[0],dtype='float64') * nbl * nfreq # for simulations

# npix = NPIX
# dt = times[1]-times[0]
# baselines = cp.asarray(bl_itrs, dtype='float32')

# print("norm.shape", norm.shape)
# print("baselines.shape", baselines.shape)
# print("dir_cosines.shape", dir_cosines.shape)
# ev1 = cp.cuda.Event()
# ev2 = cp.cuda.Event()
# ev1.record()
# # ntime=1
# avg_map = map_many_times(dir_cosines, baselines, vis_gpu, norm, d_freqs, dt, npix, ntime)
# ev2.record()
# ev2.synchronize()
# print(f"elapsed {cp.cuda.get_elapsed_time(ev1,ev2)/1000:.2f}s")
# # ============================================================

# if save:
#     # output_path = (f'/scratch/thomasb/mapmaking_dumps/average_map{compute}_sat{pidx}.npz')
#     output_path = (f'/scratch/thomasb/mohan/mapmaking_test_jul22/sim_map_nside_{NSIDE}_nbl_{nbl}_tstart_{times[0]:.0f}UTC_ntime_{len(times)}_dt_{dt*1e3:.0f}ms_fstart_{freqs[0]/1e3:.0f}kHz_bw_{(freqs[-1]-freqs[0])/1e3:.0f}kHz.npz')
#     # output_path = ('/scratch/thomasb/mohan/average_map_test_vis.npz')
#     np.savez(
#         output_path,
#         avg_map=avg_map,
#         nside=NSIDE,
#         times = times,
#         freqs = freqs,
#     )
#     print("Saved results to:", output_path)

# sys.exit()



### REAL DATA STUFF ###
# pidxs = [7]
pidxs = np.arange(11)
for pidx in pidxs:
    print("starting pidx", pidx)
    # path_data = f'/scratch/thomasb/mapmaking_dumps/all_sats_science_band/satpass_{pidx}.npz'
    # path_data = f'/scratch/thomasb/mapmaking_dumps/all_sats_science_band_flipped/satpass_{pidx}.npz'
    # path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/refactor_jul22_test.npz'
    # path_data = '/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/vis_dump.npz'
    # path_data = f'/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/refactor_jul22_test_old_data_pass{pidx}.npz'
    # path_data = f'/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/refactor_bugfix_jul22_test_old_data_pass{pidx}.npz'
    # path_data = f'/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/refactor_bugfix_jul22_test_old_data_ntime500_pass7.npz'
    path_data = f'/home/mohanagr/Jupyter/MARS Fringe analysis (Summer 2025)/output/refactor_bugfix_jul22_test_old_data_ntime500_pass{pidx}.npz'
    print("loaded", path_data)
    # # ============================================================
    # # Load Stuff
    # 
    # 
    # with np.load(path_data) as f:
    #     vis=np.nan_to_num(f['vis'], nan=0.)
    #     print("vis any nan?", np.any(np.isnan(vis)))
    #     vis = np.ascontiguousarray(vis.transpose(0,2,1)) #old data was ntime, nfreq, nbl
    #     cnt=f['cnt']
    #     cnt = np.ascontiguousarray(cnt.transpose(0,2,1)) #old data was ntime, nfreq, nbl
    #     mask = (cnt<1).astype(np.int64)
    #     tstart=f['tstart']
    #     deltat=f['deltat']
    #     deltaf=f['deltaf']
    #     fstart=360*250e6/4096
    #     times = (np.arange(vis.shape[0]) + 0.5) * deltat + tstart
    #     freqs = np.arange(vis.shape[2]) * deltaf + fstart
        
    with np.load(path_data) as f:
        # vis = f['vis']
        vis = f['data']
        print("vis shape:", vis.shape)
        print('vis value', vis[0,0,0])
    
        # new version
        mask = f['mask']
        print("mask shape:", mask.shape)
        times = f['times']
        freqs = f['freqs']
    
    print("len times", len(times), "tstart", times[0])
    print("dt", times[1]-times[0])
    
    
    vis_gpu = cp.asarray((1-mask)*vis, dtype=cp.complex64)
    
    print("vis gpu shape", vis_gpu.shape)

    # ============================================================
    # Frequencies
    
    # Old Version
    # fstart = 360 * 250e6 / 4096
    # freqs = (np.arange(vis.shape[1]) * deltaf+ fstart)
    
    # new version
    # freqs = freqs*250e6/4096
    
    print("frequency range:",freqs[0], "to",freqs[-1],"Hz")
    
    # ============================================================
    # Select visibility times
    
    ntimes = vis.shape[0]
    time_indices = np.arange(ntimes)
    
    print("total visibility times:", ntimes)
    
    # ============================================================
    # GPU setup
    
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print("number of GPUs:", num_gpus)
    
    current_id = cp.cuda.runtime.getDevice()
    print("active GPU:", current_id)
    
    
    
    good_bls = np.arange(1, nbl) #hard-coded for now: remove MARS1-MARS2 because of bad RFI
    # good_bls = np.arange(0, nbl) #sim
    nbl_good = len(good_bls)
    print("using baselines:", good_bls)
    print("number of baselines used:", nbl_good)
    
    nfreq = len(freqs)
    d_freqs = cp.asarray(freqs,dtype=cp.float32) # change to good_freqs later
    print("shape of freqs is", d_freqs.shape)
    
    ev1 = cp.cuda.Event()
    ev2 = cp.cuda.Event()
    
    #==Center time of visibility
    #obstime = Time(tstart + t_idx * deltat + deltat/2, format="unix", scale="utc") # old version
    obstime = Time(times[0], format="unix", scale="utc") # new version
    print("center time:", obstime.unix)
    
    #==Pixel coords into alt/az
    t1 = time.time()
    altaz =src.transform_to(AltAz(location=ant0,obstime=obstime))
    t2 = time.time()
    print('altaz calculation:', t2-t1, 'seconds')
    t1 = time.time()
    ha, dec = rfitools.azalt_to_hadec(altaz.az.rad, altaz.alt.rad, ant0.lat.rad) #apparent hadec
    t2 = time.time()
    print('hadec calculation:', t2-t1, 'seconds')
    
    gha = ha - ant0.lon.rad # HA at Greenwich
    l,m,n = np.cos(gha)*np.cos(dec), -np.sin(gha)*np.cos(dec), np.sin(dec)
    dir_cosines = cp.asarray([l, m, n], dtype='float32') #we'll keep these fixed and rotate baselines
    
    ntime = vis_gpu.shape[0]
    
    #== Normalization Factor
    norm = cp.asarray(np.sum(1-mask[:, good_bls, :].reshape(ntime, -1), axis=1),dtype='float64')
    print("max possible norm is", len(good_bls)*nfreq, "min,median,max norm", np.min(norm), np.median(norm), np.max(norm))
    # print("norm is", norm)
    # norm = cp.ones(vis_gpu.shape[0],dtype='float64') * nbl * nfreq # for simulations
    
    npix = NPIX
    dt = times[1]-times[0]
    baselines = cp.asarray(bl_itrs[good_bls], dtype='float32')
    d_vis = vis_gpu[:, good_bls, :].copy() # use good bls
    
    print("norm.shape", norm.shape)
    print("baselines.shape", baselines.shape)
    print("dir_cosines.shape", dir_cosines.shape)
    
    ev1.record()
    # ntime=1
    avg_map = map_many_times(dir_cosines, baselines, d_vis, norm, d_freqs, dt, npix, ntime)
    ev2.record()
    ev2.synchronize()
    print(f"elapsed {cp.cuda.get_elapsed_time(ev1,ev2)/1000:.2f}s")
    # ============================================================

    if save:
        # output_path = (f'/scratch/thomasb/mapmaking_dumps/average_map{compute}_sat{pidx}.npz')
        output_path = (f'/scratch/thomasb/mohan/mapmaking_test_jul22/map_nside_{NSIDE}_nbl_{nbl_good}_tstart_{times[0]:.0f}UTC_ntime_{ntime}_dt_{dt*1e3:.0f}ms_fstart_{freqs[0]/1e3:.0f}kHz_bw_{(freqs[-1]-freqs[0])/1e3:.0f}kHz_bugfix_finalrun_pass{pidx}.npz')
        # output_path = ('/scratch/thomasb/mohan/average_map_test_vis.npz')
        np.savez(
            output_path,
            avg_map=avg_map,
            nside=NSIDE,
            times = times,
            freqs = freqs,
            bls = good_bls
        )
        print("Saved results to:", output_path)