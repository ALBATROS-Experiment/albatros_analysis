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
from skyfield.api import load, wgs84

import helper_discrepancies as hd
import helper_finetiming as hf
import figures as fgs

sys.path.append(os.path.expanduser('~'))

path_config = "/home/thomasb/albatros_analysis/scripts/orbcomm/config/config_batch2.json"
fname_pulses = "data/pulses2.json"
fname_cutter = "timing_discrepancies/cutting.json"
fname_cutter2 = 'fine_timing/cutting.json'
fname_map = 'timing_discrepancies/times_all_incoherent.json'

path_batch = '/scratch/thomasb/batch_1753200150'
finetiming_out = os.path.join(path_batch, 'fine_timing')
os.makedirs(finetiming_out, exist_ok=True)
debugplot_out = os.path.join(finetiming_out, 'debugplots')
os.makedirs(debugplot_out, exist_ok=True)

with open(path_config, "r") as f:
    config = json.load(f)
antpos=[]
for i, (ant, details) in enumerate(config["antennas"].items()):
        antpos.append(details["coordinates"])
print(antpos)

batch_start_ts = config["correlation"]["start_timestamp"]
batch_end_ts = config["correlation"]["end_timestamp"]
osamp = config['correlation']['osamp']
acclen = config['correlation']['new_acclen']

ant_idxs = [0, 2, 3, 4, 5, 6]
n_blines = len(ant_idxs)*(len(ant_idxs)-1)//2
antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}
nant_used = len(ant_idxs)
T_SPECTRA = 4096/250e6 * osamp

#batch2
spec_per_UTC = 1.638400946109685e-05
UTC_offset = 1753200128.469018

plot = True

path_pulses = os.path.join(path_batch, fname_pulses)
with open(path_pulses, "r") as f_pulses:
    pulse_list = json.load(f_pulses)

path_cutter = os.path.join(path_batch, fname_cutter)
with open(path_cutter, "r") as f_cutter:
    cutter_dict = json.load(f_cutter)

path_cutter2 = os.path.join(path_batch, fname_cutter2)
with open(path_cutter2, "r") as f_cutter2:
    cutter_dict2 = json.load(f_cutter2)

path_map = os.path.join(path_batch, fname_map)
with open(path_map, "r") as f_map:
    map_dict = json.load(f_map)

taus_fitted_all = {}
overflow = False
pulse_start_spec = 0


for idx_pulse in range(len(pulse_list)):
    #get pulse start/end times from file
    print(f'=======STARTING PULSE INDEX {idx_pulse}')
    pulse = pulse_list[idx_pulse]
    pulse_start_ts,pulse_end_ts=pulse['t_start'],pulse['t_end']

    #determine sat type and if masking is necessary
    satID = pulse['sat']
    assert satID in {28654,25338,33591,57166,59051}
    if satID in {59051, 57166}:
        masking = False
    if satID in {28654,25338,33591}:
        masking = True
    print(f'Masking is {masking}')

    #get baseband detection channels
    chan_det = pulse['channel']
    print('detection channel', chan_det)
    if chan_det%2 == 0:
        chans_old = np.arange(chan_det-2, chan_det+2) + 1834
    else:
        chans_old = np.arange(chan_det-1, chan_det+3) + 1834

    #get data filename
    fname_data = f"data_raw_osamp=64_start={pulse_start_ts}_end={pulse_end_ts}_chans={chans_old[0]}:{chans_old[-1]}.npy"
    
    if plot ==True:
        pulse_out = os.path.join(debugplot_out, f'pulse_{pulse_start_ts}')
        os.makedirs(pulse_out, exist_ok=True)
    cut = cutter_dict[fname_data]
    cut2 = cutter_dict2[fname_data]
    map = map_dict[fname_data]

    #via cutting json, get new channels and cut spectra
    spectra_cut_start,spectra_cut_end=cut2['spectra_start'],cut2['spectra_end']
    chan_new_start, chan_new_end  = cut["new_chans"]
    chans_new = np.arange(chan_new_start, chan_new_end)
    nchans = len(chans_new)
    print(nchans)
    print(chans_new)

    #get proper start time/end time via discrep fits
    new_pulse_start = map["start_spectrum"] + spectra_cut_start * 64
    if new_pulse_start < pulse_start_spec - 2**30: 
        overflow = True
    if overflow:
        new_pulse_start += 2**32
    pulse_start_spec = new_pulse_start
    print('starting specnum', pulse_start_spec)
    pulse_start_fitted = spec_per_UTC*pulse_start_spec + UTC_offset
    print('fitted starting UTC',pulse_start_fitted)

    #get TLE path, frequencies in our data
    tle_path = outils.get_tle_file(pulse_start_fitted, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    freqs = (chans_new/64 + chans_old[0])*250e6/4096
    freqs = 250e6-freqs
    print(freqs.shape)

    #load up the data
    data_all = np.load(os.path.join(path_batch, 'data', fname_data))
    data = data_all[:, :, spectra_cut_start:spectra_cut_end, chans_new]
    print(data.shape)
    nant = data.shape[0]

    #=======GET VISIBILITIES
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
                                    pulse_start_fitted,
                                    int(pulse_end_ts - pulse_start_fitted)+2,  #BEWARE maybe the fitting messed this up
                                    satID,
                                    altaz=False
                                )
                delay = np.interp(
                    np.arange(0, data.shape[2]) * T_SPECTRA, np.arange(0, int(pulse_end_ts - pulse_start_fitted)+2), dly
                )
                spec1=data[ai,0,:,:]
                spec2=data[aj,0,:,:]
                spec2_phased = np.empty_like(spec2)
                # print("spec2", spec2_phased.shape)
                # print(spec2_phased.flags)
                spec2_phased = mt.apply_delay(spec2, spec2_phased, -delay, freqs)
                Vxx = mt.xcorr_avg(spec1,spec2_phased,acclen)
                spec1=data[ai,1,:,:]
                spec2=data[aj,1,:,:]
                spec2_phased = np.empty_like(spec2)
                spec2_phased = mt.apply_delay(spec2, spec2_phased, -delay, freqs)
                Vyy = mt.xcorr_avg(spec1,spec2_phased,acclen)
                multi_vis[bl_proc,:,:] = (Vxx+Vyy)/2
                bl_proc+=1
                print("done", ai,aj,".Processed",bl_proc, "baselines")











    #=====MAKE PHASE PLOTS, CALCULATE THERMAL NOISE
    nant=7
    antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}
    fig1,ax = plt.subplots(5,3, constrained_layout=True)
    fig1.set_size_inches(10,15)
    ax=np.ravel(ax)
    pnum=0
    print(n_blines)
    plt.suptitle(f"stokes I (phase), int. time {T_SPECTRA*acclen:4.2f}s")
    x = np.arange(data.shape[3])
    thermal_noise = np.zeros((n_blines, multi_vis.shape[1]),dtype='float64')
    vis_noise = np.zeros((n_blines, multi_vis.shape[1]),dtype='float64')
    for i in range(len(ant_idxs)):
            for j in range(i+1, len(ant_idxs)):
                ai = ant_idxs[i]
                aj = ant_idxs[j]
                ax[pnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {pnum})")
                a=np.unwrap(np.angle(multi_vis[pnum,:,:]),axis=1)
                for k in range(a.shape[0]):
                    m, c = np.polyfit(x, a[k,:], 1)
                    residual = a[k,:] - (m*x + c)
                    # sigma_phi = np.sqrt(np.mean(residual**2))
                    sigma_phi = np.std(np.angle(np.exp(1j*residual)))
                    thermal_noise[pnum,k] = sigma_phi
                    vis_noise[pnum,k] = np.std(np.cos(residual))
                b=np.unwrap(a,axis=0)
                img=ax[pnum].imshow(np.angle(multi_vis[pnum,:,:]),aspect='auto',interpolation='none',cmap='RdBu')
                cbar=plt.colorbar(img,ax=ax[pnum])
                #ax[pnum].set_ylim(cut_end, cut_start)
                pnum+=1
    # plt.tight_layout()
    print(multi_vis.shape)

    #=====PLOT THE PHASE NOISE
    if plot == True:
        fig1.savefig(os.path.join(pulse_out,"phases_unfitted.png"))

        fig2, ax = plt.subplots()
        for bl in range(n_blines):
            plt.semilogy(median_filter(thermal_noise[bl,:],10),label=f'bl id {bl}')
        ax.set_title(f"Median-filt. phase noise, all baselines, int. time {T_SPECTRA*acclen:4.2f}s")
        ax.set_ylabel("$\sigma_\phi$ (rad)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0.05,2)
        ax.grid(True)
        fig2.savefig(os.path.join(pulse_out,"noise.png"))
        plt.close(fig2)
        
    noise_var = median_filter(thermal_noise[:,:],3,axes=1)**2
    print(noise_var.shape)
    weights = np.sqrt(np.size(noise_var)/noise_var/np.sum(1/noise_var))
    phase = np.angle(multi_vis[:, :, :])

    phase = np.unwrap(phase,axis=2)
    phase = np.unwrap(phase,axis=1)
    print('phase shape', phase.shape)

    #make baseline fastest moving axis for convenience
    data_matrix = phase.transpose(1,2,0).copy()
    noise_matrix = noise_var.T.copy()
    weight_matrix = weights.T.copy()
    print('data_matrix shape', data_matrix.shape)
    print('weight_matrix shape', weight_matrix.shape)
    print('noise matrix shape', noise_matrix.shape)

    # make frequencies order 1 for fitting purposes
    freqs_normalized = freqs.copy()/1e9

    # set up some global variables
    assert nchans == data_matrix.shape[1]
    ntime = data_matrix.shape[0]
    print("nant,nfreq,ntime,nbl", nant,nchans,ntime,n_blines)

    nant = len(ant_idxs)
    #MAKE INITIAL FIT
    taus_fitted = np.zeros((ntime,nant-1),dtype='float64')
    for tt in range(ntime):
        ydata = data_matrix[tt,:,:].ravel()
        weights = weight_matrix[tt,:]
        t1=time.time()
        tau_fit_params = least_squares(mt.func,np.ones(nant-1), args=(ydata, freqs_normalized, weights),
                                                            jac=mt.jac,
                                                            method='lm',
                                                            ftol=1e-06, 
                                                            xtol=1e-06, 
                                                            gtol=1e-06, 
                                                            loss='linear')
        t2=time.time()
        taus_fitted[tt,:] = tau_fit_params['x']
        #print(fit_params)
        print(f"done tt={tt}, time = {t2-t1:5.3f}")

    # unwrap the fitted taus into actual delay values
    labels=['M1-4', 'M1-5', 'M1-6', 'M1-7', 'M1-8']
    taus_unwrapped = np.unwrap(np.angle(np.exp(1j*taus_fitted.reshape(-1,nant-1)*2*np.pi*freqs_normalized.mean())),axis=0)/(2*np.pi*freqs_normalized.mean())

    # plot the relative delays of the other antenna with respect to the reference antenna

    if plot == True:
        fig3, ax3 = plt.subplots()
        ax3.plot(taus_fitted)
        fig3.savefig(os.path.join(pulse_out, 'fitted_taus.png'))

        fig4, ax4 = plt.subplots()
        ax4.plot(taus_unwrapped[:,:],label=labels)
        ax4.set_ylabel('Relative Delay (ns)')
        ax4.set_xlabel('Time (s)')
        ax4.legend()
        fig4.savefig(os.path.join(pulse_out, "taus_unwrapped.png"))


    fig5,ax5 = plt.subplots(5,3, constrained_layout=True)
    fig5.set_size_inches(10,15)
    ax5=np.ravel(ax5)

    pnum=0
    print(n_blines)
    x = np.arange(nchans)
    phased_vis = np.empty_like(multi_vis[:,:,:])
    for i in range(len(ant_idxs)):
            for j in range(i+1, len(ant_idxs)):
                ai = ant_idxs[i]
                aj = ant_idxs[j]
                if i==0:
                    tau1=0
                else:
                    tau1 = taus_unwrapped[:,i-1]
                tau2 = taus_unwrapped[:,j-1]
                rel_delay = tau1-tau2
                ax5[pnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {pnum})")
                phased_vis[pnum,:,:] = multi_vis[pnum,:,:]*np.exp(-2j*np.pi*freqs_normalized[None,:]*rel_delay[:,None])
                img=ax5[pnum].imshow(np.angle(phased_vis[pnum,:,:]),aspect='auto',interpolation='none',cmap='RdBu')
                cbar=plt.colorbar(img,ax=ax5[pnum])
                pnum+=1
    # plt.tight_layout()
    if plot == True:
        fig5.suptitle(f"stokes I (phase), int. time {T_SPECTRA*acclen:4.2f}s")
        fig5.savefig(os.path.join(pulse_out, 'fitted_phases.png'))

    #take average across time once aligned to boost SNR
    avg_multi_vis = np.mean(phased_vis,axis=1)
    #now have very clean phase ramp with super low noise
    avg_multi_vis = avg_multi_vis.T.copy()
    plt.plot(np.unwrap(np.angle(avg_multi_vis),axis=0))
    print('average multi vis shape', avg_multi_vis.shape)

    #get that noise by fitting a line and getting std of residuals
    noise_matrix_avg = np.zeros((1,n_blines),dtype='float64')
    for bl in range(n_blines):
        y = np.unwrap(np.angle(avg_multi_vis[:,bl]))
        m, c = np.polyfit(x,y, 1)
        residual = y - (m*x + c)
        noise_matrix_avg[0,bl] = np.std(residual)
    print(noise_matrix_avg)


    #=======FINAL FIT
    ntime2=1
    nant=6
    nbl=nant*(nant-1)//2

    data_matrix = np.unwrap(np.angle(avg_multi_vis),axis=0)
    data_matrix = data_matrix.reshape(ntime2,nchans,nbl)

    Ag = mt.get_grammian(nant)
    print('data matrix shape', data_matrix.shape)

    AtA2,Atd2  = mt.get_AtA_Atd(data_matrix,
                                Ag,
                                noise_matrix_avg**2,
                                freqs_normalized,
                                nant,
                                nchans,
                                ntime2,
                                fit_constant=True)

    AtA_inv2 = np.linalg.inv(AtA2)
    mfit2 = AtA_inv2 @ Atd2

    # Total parameters for tau: ntime * (nant-1)
    bs = nant - 1
    tau_linear2 = mfit2[:ntime2 * bs]
    phi_linear2 = mfit2[ntime2 * bs:]
    errs_all = np.sqrt(np.diag(AtA_inv2))

    taus_fitted_all = tau_linear2[:, np.newaxis] + taus_unwrapped.T
    taus_errs_all = errs_all[:bs]

    with h5py.File(os.path.join(finetiming_out, 'finetiming_dump1.h5'), 'a') as f:
        if fname_data not in f:
            grp = f.create_group(fname_data)
        else:
            grp = f[fname_data]
        if 'taus' in grp:
            del grp['taus']
        taus = grp.create_dataset('taus', data=taus_fitted_all)
        taus.attrs['starting_specnum'] = pulse_start_spec
        taus_errs = grp.create_dataset('errs', data = taus_errs_all)

print('done!')