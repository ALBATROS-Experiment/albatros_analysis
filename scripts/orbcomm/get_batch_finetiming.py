#system stuff
import os
import sys
sys.path.append(os.path.expanduser('~'))
#general
import numpy as np 
import numba as nb
import time
import importlib
import json
import argparse
import h5py
from matplotlib import pyplot as plt
from datetime import datetime as dt
#utils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.utils import finetiming_utils as futils
#helper and functions
from scipy.optimize import minimize,check_grad,least_squares
from scipy.ndimage import median_filter
from scipy.ndimage import binary_opening, binary_closing, label
from skyfield.api import load, wgs84
#etc
import figures as fgs

sys.path.append(os.path.expanduser('~'))

#=========================================================================
# SETUP
#=========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    #parser.add_argument("pulse_list", type = list)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    antpos=[]
    for i, (ant, details) in enumerate(config["antennas"].items()):
        antpos.append(details["coordinates"])
    print(antpos)

    batch_start_ts = config["correlation"]["start_timestamp"]
    osamp = config['correlation']['osamp']
    acclen = config['correlation']['new_acclen']

    T_SPECTRA = 4096/250e6 * osamp
    METEOR_ONLY = True
    ant_idxs = [0, 1, 2, 3, 4, 5, 6]
    antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}
    nant_used = len(ant_idxs)
    nblines = len(ant_idxs)*(len(ant_idxs)-1)//2
    print(T_SPECTRA)
    print(nblines)  

    path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
    path_fine_timing = os.path.join(path_batch, 'fine_timing')
    os.makedirs(path_fine_timing, exist_ok=True)

    #pulse list
    with open(os.path.join(path_batch, "data/pulses.json"), "r") as f:
        list_pulses = json.load(f)
    #discrepancy results
    with open(os.path.join(path_batch, 'timing_discrepancies/times_all.json'), "r") as f:
        dict_fits_discrep = json.load(f)
    #discrepancy cutting
    with open(os.path.join(path_batch, "data/cutting_discrep.json"), "r") as f:
        dict_cutting_discrep = json.load(f)
    #finetiming cutting, create it doesn't exist
    path_cutting_finetiming = os.path.join(path_batch, 'data/cutting_finetiming.json')
    if os.path.exists(path_cutting_finetiming):
        with open(path_cutting_finetiming, 'r') as f:
            dict_cutting_finetiming = json.load(f)
    else:
        dict_cutting_finetiming = {}
        
    #load in the discrepancy mapping
    UTC_per_spec = dict_fits_discrep['fit']['UTC_per_spec']
    UTC_offset = dict_fits_discrep['fit']['UTC_offset']

    #BATCH 2
    #UTC_per_spec = 1.638401491028474e-05
    #UTC_offset = 1753200128.4654782
    
    #=========================================================================
    #ITERATION
    #=========================================================================
    for idx_pulse in range(len(list_pulses)):
        #extract from json
        pulse = list_pulses[idx_pulse]
        pulse_start_ts = pulse['t_start']
        pulse_end_ts = pulse['t_end']
        satID = pulse['sat']
        chan_det = pulse['channel']
        print(f'\nStarting Pulse {idx_pulse} satID {satID}')
        print(pulse)
        print('detection channel', chan_det)
        #make debugplot dir
        path_debug = os.path.join(path_fine_timing, 'debugplots')
        os.makedirs(path_debug, exist_ok=True)
        path_pulse = os.path.join(path_debug, f'pulse_{pulse_start_ts}_{satID}')
        os.makedirs(path_pulse, exist_ok=True)
        #check if we're masking (i.e. if NOAA)
        assert satID in {59051, 57166, 28654,25338,33591}
        if satID in {59051, 57166}:
            masking = False
        if satID in {28654,25338,33591}:
            masking=True
            if METEOR_ONLY:
                continue
        print('masking:', masking)
        #get old channels for fname
        if chan_det%2 == 0:
            chans_old = np.arange(chan_det-2, chan_det+2) + 1834
        else:
            chans_old = np.arange(chan_det-1, chan_det+3) + 1834
        print(chans_old)
        #get fname
        fname_data = f"data_raw_osamp=64_start={pulse_start_ts}_end={pulse_end_ts}_chans={chans_old[0]}:{chans_old[-1]}.npy"
        #extract from discrep cutter (to get channels)
        cut_discrep = dict_cutting_discrep[fname_data]
        chan_new_start, chan_new_end  = cut_discrep["cut_chans"]
        chans_new = np.arange(chan_new_start, chan_new_end)
        nchans = len(chans_new)
        print('nchans:', nchans)
        print('new chans:', chans_new)
        #get frequencies
        freqs = 250e6 - (chans_new/64 + chans_old[0])*250e6/4096
        #load up data
        print('Loading Data')
        data1 = np.load(os.path.join(path_batch, 'data', fname_data), mmap_mode='r')
        data1 = data1[:, :, :, chans_new]
        nant = data1.shape[0]
        print('Uncut Data Shape:', data1.shape)
        print(nant)

        #check for overflow
        spec_pstart1 = dict_fits_discrep[fname_data]["start_specnum"]
        print('starting specnum', spec_pstart1)
        #correct for UTC offset
        pstart1 = UTC_per_spec*spec_pstart1 + UTC_offset
        print(pstart1)
        tle_path = outils.get_tle_file(pstart1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")


        if fname_data in dict_cutting_finetiming:
            print('Data already cut! Extracting info from scratch')
            cut = dict_cutting_finetiming[fname_data]
            spec_cut_start,spec_cut_end=cut['spec_cut_start'],cut['spec_cut_end']
            spec_pstart2 = cut['spec_start_corrected']
        else:
            #get visibilities without any cutting
            vis1 = futils.get_vis(data1,satID,freqs,pstart1,pulse_end_ts,antpos,ant_idxs,tle_path,T_SPECTRA,osamp,acclen)
            #get mask if needed
            if masking:
                mask1, fig_mask1 = get_mask(vis1, tol=1.7)
                fig_mask1.savefig(os.path.join(path_pulse, 'mask1.png'))
                plt.close(fig_mask1)
                print(mask1.shape)
            else:
                mask1 = None
            #get thermal noise to see where to cut
            thermal_noise1, fig_phases1 = futils.get_thermal_noise(vis1,ant_idxs,mask=mask1)
            print('Thermal noise 1 shape:', thermal_noise1.shape)
            #plot phases
            fig_phases1.savefig(os.path.join(path_pulse, 'phases1.png'))
            plt.close(fig_phases1)
            #plot thermal noise
            fig_thermal_noise1, ax = plt.subplots()
            for bl in range(nblines):
                ax.semilogy(median_filter(thermal_noise1[bl,:],10),label=f'bl id {bl}')
            ax.set_title(f"Median-filt. phase noise, all baselines, int. time {T_SPECTRA*acclen:4.2f}s")
            ax.set_ylabel("$\sigma_\phi$ (rad)")
            ax.set_xlabel("Time (s)")
            ax.set_ylim(0.05,2)
            ax.grid(True)
            fig_thermal_noise1.savefig(os.path.join(path_pulse, 'thermal_noise1.png'))
            plt.close(fig_thermal_noise1)
            #cut according to phase noise
            chunks_cut_start, chunks_cut_end = futils.find_lowest_noise(thermal_noise1)
            spec_cut_start, spec_cut_end = chunks_cut_start*acclen, chunks_cut_end*acclen
            spec_pstart2 = spec_pstart1 + spec_cut_start*64
            print(chunks_cut_start, chunks_cut_end)
            #save important stuff so we don't have to keep doing this
            dict_cutting_finetiming[fname_data] ={'spec_cut_start':spec_cut_start,
                                        'spec_cut_end': spec_cut_end,
                                        'spec_start_corrected': spec_pstart2,
                                        'new_chans':[chan_new_start, chan_new_end]}
            with open(path_cutting_finetiming, 'w') as f:
                json.dump(dict_cutting_finetiming, f, indent=4)

        pstart2 = UTC_per_spec*spec_pstart2 + UTC_offset
        data2 = data1[:,:,spec_cut_start:spec_cut_end,:]
        print('pstart2:', pstart2)
        print('data2 shape', data2.shape)
        vis2=futils.get_vis(data2,satID,freqs,pstart2,pulse_end_ts,antpos,ant_idxs,tle_path,T_SPECTRA,osamp,acclen)
        nblines, ntimes, nchans = vis2.shape
        print(vis2.shape)

        if masking:
            mask2, fig_mask2 = get_mask(vis2, tol=1.7)
            fig_mask2.savefig(os.path.join(path_pulse, 'mask2.png'))
            plt.close(fig_mask2)
        else:
            mask2 = None

        #get cut phases and thermal noise
        thermal_noise2, fig_phases2 = futils.get_thermal_noise(vis2, ant_idxs, mask=mask2)
        fig_phases2.savefig(os.path.join(path_pulse, 'phases2.png'))
        plt.close(fig_phases2)

        #plot cut thermal noise
        fig_thermal_noise2, ax = plt.subplots()
        for bl in range(nblines):
            ax.semilogy(median_filter(thermal_noise2[bl,:],10),label=f'bl id {bl}')
        ax.set_title(f"Median-filt. phase noise, all baselines, int. time {T_SPECTRA*acclen:4.2f}s")
        ax.set_ylabel("$\sigma_\phi$ (rad)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0.05,2)
        ax.grid(True)
        fig_thermal_noise2.savefig(os.path.join(path_pulse, 'thermal_noise2.png'))
        plt.close(fig_thermal_noise2)

        #=========================================================================
        #WEIGHTING
        noise_var = median_filter(thermal_noise2[:,:],3,axes=1)**2   #why a median filter of just 3??
        print('noise var shape', noise_var.shape)
        weights = np.sqrt(np.size(noise_var)/noise_var/np.sum(1/noise_var))
        angle = np.angle(vis2)
        fig_weights, ax = plt.subplots()
        if masking:
            weights = np.tile(weights[:, :, None], (1, 1, nchans))
            weights = np.where(mask2[None, :, :], 0, weights)
            weight_matrix = weights.transpose(1,2,0).copy() #do same for weights so it follows
            phase_unwrapped = angle

            for bl in range(nblines):
                #max is fine since weights either zero or depend on full channel thermal noise
                ax.semilogy(np.max(weights[bl,:, :], axis=1),label=f'bl id {bl}')
        else:
            phase_unwrapped = angle
            weight_matrix = weights.T.copy()
            fig_weights, ax = plt.subplots()
            for bl in range(nblines):
                ax.semilogy(weights[bl,:],label=f'bl id {bl}')
        ax.legend()
        fig_weights.savefig(os.path.join(path_pulse, 'weights.png'))
        plt.close(fig_weights)
            
        #make baseline fastest moving axis for convenience
        data_matrix = phase_unwrapped.transpose(1,2,0).copy()
        noise_matrix = noise_var.T.copy()
        print('data_matrix shape', data_matrix.shape)
        print('weight_matrix shape', weight_matrix.shape)
        print('noise matrix shape', noise_matrix.shape)

        # make frequencies order 1 for fitting purposes
        freqs_normalized = freqs.copy()/1e9

        # quick sanity check
        assert ntimes == data_matrix.shape[0]
        assert nchans == data_matrix.shape[1]
        assert nblines == data_matrix.shape[2]
        assert nant_used == len(ant_idxs) 
        print('ntimes', ntimes)
        print('nchans', nchans)
        print('nblines', nblines)
        print("nant used:", nant_used)

        # DETERMINE RAMP GUESSES

        avg_multi_vis0 = np.mean(vis2[:, 80:90, :], axis=1) #hard-coded for now
        print(avg_multi_vis0.shape)

        avg_multi_phase0 = np.unwrap(np.angle(avg_multi_vis0), axis = 1)
        fig, ax = plt.subplots()
        for i in range(nblines):
            ax.plot(avg_multi_phase0[i,:])
        ax.set_xlabel('Channel idx')
        ax.set_ylabel('Unwrapped Phase')
        fig.savefig(os.path.join(path_pulse, 'guess_ramps.png'))
        plt.close(fig)

        #get that noise by fitting a line and getting std of residuals
        noise_matrix_avg0 = np.zeros((1,nblines),dtype='float64')
        x = freqs
        for bl in range(nblines):
            y = avg_multi_phase0[bl,:]
            m, c = np.polyfit(x,y, 1)
            residual = y - (m*x + c)
            noise_matrix_avg0[0,bl] = np.std(residual)

        #can now just fit linearly using normal equations
        data_matrix0 = avg_multi_phase0.T.reshape(1,nchans,nblines)

        Ag = futils.get_grammian(nant)
        print(data_matrix0.shape)

        AtA2,Atd2 = futils.get_AtA_Atd(data_matrix0,Ag,noise_matrix_avg0**2,freqs_normalized,nant,nchans,1,fit_constant=True)

        AtA_inv2 = np.linalg.inv(AtA2)

        mfit2 = AtA_inv2 @ Atd2

        # Total parameters for tau: ntime * (nant-1)
        bs = nant - 1
        taus_guess0 = mfit2[:1 * bs]
        phi_guess0 = mfit2[1 * bs:]
        errs_guess0 = np.sqrt(np.diag(AtA_inv2))
        print(taus_guess0)
        print(phi_guess0)
        print(errs_guess0)

        #test to see
        antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}
        fig,ax = plt.subplots(7,3, constrained_layout=True)
        fig.set_size_inches(10,20)
        ax=np.ravel(ax)

        blnum=0
        x = np.arange(nchans)
        phased_vis = np.empty_like(vis2[:,:,:])
        for i in range(len(ant_idxs)):
                for j in range(i+1, len(ant_idxs)):
                    ai = ant_idxs[i]
                    aj = ant_idxs[j]
                    ax[blnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {blnum})")
                    if i==0:
                        tau1=0
                    else:
                        tau1 = taus_guess0[i-1]
                    tau2 = taus_guess0[j-1]
                    rel_delay = tau1-tau2
                    phase_bline = avg_multi_phase0[blnum, :]
                    phase_pred = np.unwrap(np.angle(np.exp(2j*np.pi*rel_delay*freqs_normalized)))
                    ax[blnum].plot(phase_bline - phase_bline[0], label='Data')
                    ax[blnum].plot(phase_pred - phase_pred[0], label='Coarse Fit')
                    ax[blnum].set_ylabel('Relative Phase')
                    ax[blnum].legend()
                    blnum+=1
        plt.tight_layout()
        fig.savefig(os.path.join(path_pulse, 'ramp_comparison.png'))

    
        #=========================================================================
        # FIT AND UNWRAP TAUS

        # array of fitted taus for each time, for each (non-reference) antenna
        taus_fitted = np.zeros((ntimes,nant_used-1),dtype='float64')
        # make some guess about fitted taus (for one sample time)
        taus_guess = taus_guess0
        for tt in range(ntimes):
            ydata = data_matrix[tt,:,:].ravel()
            if masking:
                weights = weight_matrix[tt,:,:]
            else:
                weights = weight_matrix[tt,:]
            t1=time.time()
            tau_fit_params = least_squares(func, taus_guess, args=(ydata, freqs_normalized, weights),
                                                                jac=jac,
                                                                method='lm',
                                                                ftol=1e-06, 
                                                                xtol=1e-06, 
                                                                gtol=1e-06, 
                                                                loss='linear')
            t2=time.time()
            taus_fitted[tt,:] = tau_fit_params['x']
            taus_guess = tau_fit_params['x']
            print(f"done tt={tt}, time = {t2-t1:5.3f}")
        fig, ax = plt.subplots(figsize=(8,5))
        labels=['M1-2', 'M1-4', 'M1-5', 'M1-6', 'M1-7', 'M1-8']
        for antidx in range(nant-1):
            ax.plot(taus_fitted[:,antidx] - taus_fitted[0,antidx], label=labels[antidx])
            ax.set_xlabel('Time Sample (~1 s)')
            ax.set_ylabel('Relative Drift (ns)')
        ax.legend()
        plt.tight_layout()
        plt.suptitle('Wrapped Taus')
        fig.savefig(os.path.join(path_pulse, 'taus_fitted_wrapped.png'))
        plt.close(fig)


        # DO WE NEED TO UNWRAP???
        labels=['M1-2', 'M1-4', 'M1-5', 'M1-6', 'M1-7', 'M1-8']
        taus_unwrapped = np.unwrap(np.angle(np.exp(1j*taus_fitted.reshape(-1,nant_used-1)*2*np.pi*freqs_normalized.mean())),axis=0)/(2*np.pi*freqs_normalized.mean())
        # plot the relative delays of the other antenna with respect to the reference antenna
        fig_taus_unwrapped, ax = plt.subplots()
        ax.plot(taus_unwrapped[:,:],label=labels)
        ax.set_ylabel('Relative Delay (ns)')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right')
        fig_taus_unwrapped.savefig(os.path.join(path_pulse, 'taus_fitted_unwrapped.png'))

        #=========================================================================
        #generate aligned phase plots for each baseline with fitted taus
        #in the meantime also phases the visibilities to align them

        antmap = {0:"MARS1", 1:"MARS2",2:"MARS4",3:"MARS5",4:"MARS6",5:"MARS7",6:"MARS8"}
        fig_phases2_aligned,ax = plt.subplots(7,3, constrained_layout=True)
        fig_phases2_aligned.set_size_inches(10,15)
        ax=np.ravel(ax)

        blnum=0
        print(nblines)
        fig_phases2_aligned.suptitle(f"stokes I (phase), int. time {T_SPECTRA*acclen:4.2f}s")
        x = np.arange(nchans)
        phased_vis = np.empty_like(vis2[:,:,:])
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
                ax[blnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {blnum})")
                phased_vis[blnum,:,:] = vis2[blnum,:,:]*np.exp(-2j*np.pi*freqs_normalized[None,:]*rel_delay[:,None])
                if masking:
                    phased_vis_masked = np.ma.masked_where(mask2, np.angle(phased_vis[blnum,:,:]))
                    img = ax[blnum].imshow(phased_vis_masked, aspect='auto', interpolation='none', cmap='RdBu')
                    cbar=plt.colorbar(img,ax=ax[blnum])
                else:
                    img=ax[blnum].imshow(np.angle(phased_vis[blnum,:,:]),aspect='auto',interpolation='none',cmap='RdBu')
                    cbar=plt.colorbar(img,ax=ax[blnum])
                blnum+=1
        fig_phases2_aligned.savefig(os.path.join(path_pulse,'phases2_aligned.png'))
        plt.close(fig_phases2_aligned)

        #=========================================================================
        #take average across time once aligned to boost SNR
        if masking:
            #introduce mask 3 or something to check if there are enough points ALONG TIME. 
            #if bad, just interpolate after the fact and assign high noise value
            phased_vis_masked = np.ma.array(phased_vis, mask=np.broadcast_to(mask2[None, :, :], phased_vis.shape))
            avg_multi_vis = phased_vis_masked.mean(axis=1)
        else:
            avg_multi_vis = np.mean(phased_vis, axis=1)

        #avg_multi_vis = np.mean(phased_vis,axis=1)
        #avg_multi_vis = np.median(phased_vis,axis=1)

        #now have very clean phase ramp with super low noise
        avg_multi_vis = avg_multi_vis.T.copy()
        print(avg_multi_vis.shape)
        fig_averaged_ramp, ax = plt.subplots()
        ax.plot(np.unwrap(np.angle(avg_multi_vis),axis=0))
        fig_averaged_ramp.savefig(os.path.join(path_pulse, 'averaged_ramp.png'))

        #get that noise by fitting a line and getting std of residuals
        noise_matrix_avg = np.zeros((1,nblines),dtype='float64')
        for bl in range(nblines):
            y = np.unwrap(np.angle(avg_multi_vis[:,bl]))
            m, c = np.polyfit(x,y, 1)
            residual = y - (m*x + c)
            noise_matrix_avg[0,bl] = np.std(residual)
        print(noise_matrix_avg)




        #=========================================================================
        ntime2=1
        #can now just fit linearly using normal equations
        data_matrix = np.unwrap(np.angle(avg_multi_vis),axis=0)
        data_matrix = data_matrix.reshape(ntime2,nchans,nblines)

        Ag = futils.get_grammian(nant_used)
        print(data_matrix.shape)

        AtA2,Atd2 = futils.get_AtA_Atd(data_matrix,
                                    Ag,
                                    noise_matrix_avg**2,
                                    freqs_normalized,
                                    nant_used,
                                    nchans,
                                    ntime2,
                                    fit_constant=True)


        AtA_inv2 = np.linalg.inv(AtA2)
        mfit2 = AtA_inv2 @ Atd2

        # Total parameters for tau: ntime * (nant-1)
        bs = nant_used - 1
        tau_linear2 = mfit2[:ntime2 * bs]
        phi_linear2 = mfit2[ntime2 * bs:]
        errs_all = np.sqrt(np.diag(AtA_inv2))

        taus_fitted_all = tau_linear2[:, np.newaxis] + taus_unwrapped.T
        taus_errs_all = errs_all[:bs]

        print(tau_linear2)
        print(errs_all)

        with h5py.File(os.path.join(path_fine_timing, f'timing_solution_15may.h5'), 'a') as f:
            if fname_data not in f:
                grp = f.create_group(fname_data)
            else:
                grp = f[fname_data]
            if 'taus' in grp:
                del grp['taus']
            taus = grp.create_dataset('taus', data=taus_fitted_all)
            if 'thermal_noise' in grp:
                del grp['thermal_noise']
            thermal_noise = grp.create_dataset('thermal_noise', data=thermal_noise2)
            if 'errs' in grp:
                del grp['errs']
            taus_errs = grp.create_dataset('errs', data = taus_errs_all)
            taus.attrs['starting_specnum'] = spec_pstart2
    print('done!')