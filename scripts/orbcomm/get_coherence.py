import os
import sys
sys.path.append(os.path.expanduser('~'))
import numpy as np 
import numba as nb
import time
import importlib
import json
import argparse
import h5py
from matplotlib import pyplot as plt
from datetime import datetime as dt

from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils

from scipy.optimize import minimize,check_grad,least_squares
from scipy.ndimage import median_filter
from scipy.ndimage import binary_opening, binary_closing, label
from skyfield.api import load, wgs84

import helper_finetiming as hf
import figures as fgs

sys.path.append(os.path.expanduser('~'))

#=========================================================================
#VIS STUFF
#=========================================================================

@nb.njit(parallel=True)
def apply_delay(arr, out, delay, freqs):
    # apply delay to an array of complex electric field or their correlation
    # does exp( j 2 pi nu tau) sign of tau is user dependent
    # freqs should correspond to the columns of the nspec x nchan array
    nspec = arr.shape[0]
    nchan = arr.shape[1]
    for i in nb.prange(nspec):
        for j in range(nchan):
            out[i, j] = arr[i, j] * np.exp(2j * np.pi * freqs[j] * delay[i])
    return out

@nb.njit(parallel=True)
def xcorr_avg(arr1,arr2,acclen):
    #helper function
    nblocks = arr1.shape[0]//acclen
    nchan = arr1.shape[1]
    out = np.zeros((nblocks,nchan),dtype=arr1.dtype)
    for i in nb.prange(nblocks):
        for j in range(acclen):
            for k in range(nchan):
                out[i,k] += arr1[i*acclen + j,k]*np.conj(arr2[i*acclen + j,k])
        out[i,:]/=acclen
    return out

def get_vis(data,satID,freqs, pstart,pend,antpos,ant_idxs,tle_path, T_SPECTRA,osamp,acclen):   
    n_blines = len(ant_idxs)*(len(ant_idxs)-1)//2
    multi_vis = np.zeros((n_blines, data.shape[2]//acclen, data.shape[3]), dtype='complex64', order='c')
    print("multi_vis shape", multi_vis.shape)

    bl_proc=0
    for i in range(len(ant_idxs)):
            for j in range(i+1, len(ant_idxs)):
                ai = ant_idxs[i]
                aj = ant_idxs[j]
                a1_coords=antpos[ai]
                a2_coords=antpos[aj]
                dly = outils.get_sat_delay(
                                    a1_coords,
                                    a2_coords,
                                    tle_path,
                                    pstart,
                                    int(pend - pstart)+2,
                                    satID,
                                    altaz=False
                                )
                delay = np.interp(
                    np.arange(0, data.shape[2]) * T_SPECTRA, np.arange(0, int(pend - pstart)+2), dly
                )
                spec1=data[ai,0,:,:]
                spec2=data[aj,0,:,:]
                spec2_phased = np.empty_like(spec2)
                # print("spec2", spec2_phased.shape)
                # print(spec2_phased.flags)
                spec2_phased = apply_delay(spec2, spec2_phased, -delay, freqs)
                Vxx = xcorr_avg(spec1,spec2_phased,acclen)
                spec1=data[ai,1,:,:]
                spec2=data[aj,1,:,:]
                spec2_phased = np.empty_like(spec2)
                spec2_phased = apply_delay(spec2, spec2_phased, -delay, freqs)
                Vyy = xcorr_avg(spec1,spec2_phased,acclen)
                multi_vis[bl_proc,:,:] = (Vxx+Vyy)/2
                bl_proc+=1
                print("done", ai,aj,".Processed",bl_proc, "baselines")
    return multi_vis


#=========================================================================
# SETUP
#=========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    #get antenna coordinates
    antpos=[]
    for i, (ant, details) in enumerate(config["antennas"].items()):
        antpos.append(details["coordinates"])
    print(antpos)
    #get metadata
    corr = config["correlation"]
    batch_start_ts = corr["start_timestamp"]
    osamp, acclen = corr['osamp'], corr['new_acclen']
    T_SPECTRA = 4096/250e6 * osamp
    antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}
    ant_idxs = [0, 2, 3, 4, 5, 6]
    nant_used = len(ant_idxs)
    #define and make output path
    path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
    path_coherence = os.path.join(path_batch, 'coherence')
    os.makedirs(path_coherence, exist_ok=True)
    #get metadata from scratch
    with open(os.path.join(path_batch, "data/pulses.json"), "r") as f:
        pulse_list = json.load(f)
    with open(os.path.join(path_batch, "data/cutting_finetiming.json"), "r") as f:
        dict_cutting_finetiming = json.load(f)
    with open(os.path.join(path_batch, 'timing_discrepancies/times_all_incoherent.json'), "r") as f:
        map_dict = json.load(f)
    #timing discrepancy corrections
        #BATCH 1
    # spec_per_UTC = 
    # UTC_offset = 
        #BATCH 2
    UTC_per_spec = 1.638401491028474e-05
    UTC_offset = 1753200128.4654782
    
    #=========================================================================
    #start iteration
    for idx_pulse in range(len(pulse_list)):
        #extract info from json
        pulse = pulse_list[idx_pulse]
        ts_pulse_start,ts_pulse_end = pulse['t_start'], pulse['t_end']
        satID, chan_det = pulse['sat'], pulse['channel']
        #check for american sat, print statement
        assert satID in {59051, 57166, 28654,25338,33591}
        if satID in {28654,25338,33591}:
            print('American Sat! Skipping!')
            continue
        print(f'\nStarting Pulse {idx_pulse}, satID {satID}')
        print('detection channel', chan_det)
        #make path for pulse figure outputs
        path_pulse = os.path.join(path_coherence, f'pulse_{ts_pulse_start}_{satID}')
        os.makedirs(path_pulse, exist_ok=True)
        #get old channels for fname
        if chan_det%2 == 0:
            chans_old = np.arange(chan_det-2, chan_det+2) + 1834
        else:
            chans_old = np.arange(chan_det-1, chan_det+3) + 1834
        print('Old channels:', chans_old)
        #get data file name
        fname_data = f"data_raw_osamp=64_start={ts_pulse_start}_end={ts_pulse_end}_chans={chans_old[0]}:{chans_old[-1]}.npy"
        #extract stuff from finetiming cutting
        cut = dict_cutting_finetiming[fname_data]
        chan_new_start, chan_new_end  = cut["new_chans"]
        chans_new = np.arange(chan_new_start, chan_new_end)
        spec_cut_start, spec_cut_end = cut["spec_cut_start"], cut['spec_cut_end']
        spec_pstart = cut['spec_start_corrected']
        #get frequencies
        freqs = 250e6 - (chans_new/64 + chans_old[0])*250e6/4096
        freqs_normalized = freqs.copy()/1e9 #normalize by nanoseconds, gets cancelled out by taus in ns
        #load up data
        print('Loading up Data!')
        data = np.load(os.path.join(path_batch, 'data', fname_data), mmap_mode='r')
        data = data[:, :, spec_cut_start:spec_cut_end, chans_new]
        print('Data shape:', data.shape)
        #set get discrep-corrected start time
        pstart = UTC_per_spec*spec_pstart + UTC_offset
        print('corrected pulse start time:', pstart)
        tle_path = outils.get_tle_file(pstart, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
        #=========================================================================
        #get and plot unphased visibilities
        vis = get_vis(data,satID,freqs,pstart,ts_pulse_end,antpos,ant_idxs,tle_path,T_SPECTRA,osamp,acclen)
        nblines, ntimes, nchans = vis.shape
        fig_phases,ax = plt.subplots(5,3, constrained_layout=True)
        fig_phases.set_size_inches(10,15)
        ax=np.ravel(ax)
        plt.suptitle(f"Phases Uncorrected, int. time {T_SPECTRA*acclen:4.2f}s")
        blnum = 0
        for i in range(len(ant_idxs)):
            for j in range(i+1, len(ant_idxs)):
                ai = ant_idxs[i]
                aj = ant_idxs[j]
                ax[blnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {blnum})")
                img=ax[blnum].imshow(np.angle(vis[blnum,:,:]),aspect='auto',interpolation='none',cmap='RdBu')
                cbar=plt.colorbar(img,ax=ax[blnum])
                blnum+=1
        fig_phases.savefig(os.path.join(path_pulse, 'phases.png'))
        plt.close(fig_phases)
        #=========================================================================
        #EXTRACT TIMING SOLUTION
        with h5py.File(os.path.join(path_batch, 'fine_timing/timing_solution.h5'), 'r') as f:
            group_pulse = f[fname_data]
            taus_all = group_pulse['taus']
            taus_all = taus_all[:]
            print('taus all shape', taus_all.shape)
        #=========================================================================
        #get and plot phased visibilities
        fig_phases_aligned,ax = plt.subplots(5,3, constrained_layout=True)
        fig_phases_aligned.set_size_inches(10,15)
        ax=np.ravel(ax)
        blnum=0
        fig_phases_aligned.suptitle(f"Phases Corrected, int. time {T_SPECTRA*acclen:4.2f}s")
        vis_phased = np.empty_like(vis)
        blnum=0
        for i in range(nant_used):
            for j in range(i+1, nant_used):
                ai = ant_idxs[i]
                aj = ant_idxs[j]
                if i==0:
                    tau1=0
                else:
                    tau1 = taus_all[i-1,:]
                tau2 = taus_all[j-1,:]
                rel_delay = tau1-tau2
                rel_delay -= rel_delay[0] #if we only want relative.
                ax[blnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {blnum})")
                phasor = np.exp(-2j*np.pi*freqs_normalized[None,:]*rel_delay[:,None])
                print('phasor shape', phasor.shape)
                vis_phased[blnum,:,:] = vis[blnum,:,:]*phasor
                img=ax[blnum].imshow(np.angle(vis_phased[blnum,:,:]),aspect='auto',interpolation='none',cmap='RdBu')
                cbar=plt.colorbar(img,ax=ax[blnum])
                blnum+=1
        fig_phases_aligned.savefig(os.path.join(path_pulse,'phases_aligned.png'))
        plt.close(fig_phases_aligned)
        coh_uncorr = np.abs(np.mean(vis, axis = 1))
        coh_corr = np.abs(np.mean(vis_phased, axis = 1))
        coh_ratio = coh_uncorr/coh_corr
        print(coh_ratio.shape)

        with h5py.File(os.path.join(path_coherence, f'coherence_ratios2.h5'), 'a') as f:
            if fname_data not in f:
                grp = f.create_group(fname_data)
            else:
                grp = f[fname_data]
            if 'ratios' in grp:
                del grp['ratios']
            ratios = grp.create_dataset('ratios', data=coh_ratio)
    print('done!')