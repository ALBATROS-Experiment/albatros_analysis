#system
import os
import sys
sys.path.insert(0, "/home/thomasb/")
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
from albatros_analysis.scripts.mapmaking import rfitools
#helpers and functions
from scipy.optimize import minimize,check_grad,least_squares
from scipy.ndimage import median_filter
from scipy.ndimage import binary_opening, binary_closing, label
from skyfield.api import load, wgs84
#etc

sys.path.append(os.path.expanduser('~'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument('-s', "--save_to_soln", action='store_true', help='Sets if results should be saved to the full timing_solution database.')
    parser.add_argument('-t', "--testing", type=str, default=None , help='name of test that is being run')

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    #path_config = "/home/thomasb/albatros_analysis/scripts/orbcomm/config/config_batch2.json"
    #path_config = "/home/thomasb/albatros_analysis/scripts/orbcomm/config/nov25_batch_1762272160.json"

    antpos=[]
    antmap = {}
    ant_names = []
    coarse_delays = []
    for i, (ant, details) in enumerate(config["antennas"].items()):
        antpos.append(details["coordinates"])
        antmap[i] = details['name']
        ant_names.append(details['name'])
        coarse_delays.append(details['clock_offset'])
    print('coords\n', antpos)
    print('antmap\n', antmap)

    batch_start_ts = config["correlation"]["start_timestamp"]
    batch_end_ts = config["correlation"]["end_timestamp"]
    osamp = config['correlation']['osamp']
    acclen = config['correlation']['new_acclen']
    print('osamp', osamp)
    print(acclen)

    T_SPECTRA = 4096/250e6 * osamp
    print('T_SPECTRA', T_SPECTRA)

    if args.testing is not None:
        path_batch = f'/scratch/thomasb/batch_{batch_start_ts}_testing/{args.testing}'
    else:
        path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
    print('path to batch', path_batch)
    path_finetiming = os.path.join(path_batch, 'fine_timing')
    os.makedirs(path_finetiming, exist_ok=True)
    print('path to finetiming', path_finetiming)

    path_pulses = os.path.join(path_batch, "data/pulses.json")
    with open(path_pulses, "r") as f:
        list_pulses = json.load(f)

    path_map = os.path.join(path_batch, 'timing_discrepancies/times_all.json')
    with open(path_map, "r") as f:
        map_dict = json.load(f)

    UTC_per_spec = map_dict['fit']['UTC_per_spec']
    print('UTC per spec', UTC_per_spec)
    UTC_offset = map_dict['fit']['UTC_offset']
    print('UTC offset', UTC_offset)

    nant = len(antmap)
    print('nant', nant)
    nblines = nant*(nant-1)//2
    print('nblines', nblines)
    ant_idxs = np.arange(nant)
    print('ant idxs', ant_idxs)
    window_size = int(2**13*3*5/acclen) #want around 2 minutes always
    print('window size', window_size)

    ncols = 3 #number of columns for data plotting
    triuidx = np.triu_indices(nant, k=1)
    #==================================================================
    # Iterate through pulses

    for idx_pulse in range(len(list_pulses)):
        #set to continue if you just want to save everything to timing solution folder
        #continue

        print(f'\n=================\nSTARTING PULSE {idx_pulse}')
        pulse = list_pulses[idx_pulse]
        print(pulse)

        pulse_start_ts, pulse_end_ts = pulse['t_start'], pulse['t_end']
        print('pulse times', pulse_start_ts, pulse_end_ts)
        satID = pulse['sat']
        print('pass satellite', satID)
        path_debug = os.path.join(path_finetiming, 'debugplots')
        os.makedirs(path_debug, exist_ok=True)
        path_pulse = os.path.join(path_debug, f'pulse_{pulse_start_ts}_{satID}')
        os.makedirs(path_pulse, exist_ok=True)
        print('path to pulse', path_pulse)
        chan_det = pulse['channel']
        print('detection channel', chan_det)
        pulse_start_spec = pulse['start_specnum']
        print('pulse start spectrum', pulse_start_spec)

        assert satID in {59051, 57166}
        if chan_det%2 == 0:
            chans_old = np.arange(chan_det-2, chan_det+2) + 1834
        else:
            chans_old = np.arange(chan_det-1, chan_det+3) + 1834
        print('old channel indices', chans_old)

        #double check if it's centered on the channel or not!
        new_chans = np.linspace(chans_old[0], chans_old[-1]+1, osamp*len(chans_old), endpoint=False) 
        freqs_all = 250e6 - new_chans*250e6/4096
        print('starting frequency', freqs_all[0])

        pstart1 = UTC_per_spec*pulse_start_spec + UTC_offset
        print('fitted pulse start time', pstart1)
        tle_path = outils.get_tle_file(pstart1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")

        #==================================================================
        # Load in data, compute visibilities

        fname_data = f"data_raw_osamp=64_start={pulse_start_ts}_end={pulse_end_ts}_chans={chans_old[0]}:{chans_old[-1]}.npy"
        print('\nLOADING IN DATA')
        data1 = np.load(os.path.join(path_batch, 'data', fname_data))
        print('size in GB', data1.nbytes/(1024**3))
        print('nants, npols, ntimes, nchans', data1.shape)
        print('\nCOMPUTING VISIBILITIES')
        vis1 = futils.get_vis(data1,satID,freqs_all,pstart1,pulse_end_ts,antpos,ant_idxs,tle_path,T_SPECTRA,acclen)
        fig_amp = futils.plot_amp(vis1, ant_idxs, antmap)
        fig_amp.savefig(os.path.join(path_pulse, 'amp.png'))
        plt.close(fig_amp)
        #==================================================================
        # Cut visibilities (signal frequencies and low noise times)

        chans_new_slice = futils.find_signal_channels(vis1, 'METEOR')
        vis1=vis1[:,:,chans_new_slice]

        freqs = freqs_all[chans_new_slice]
        freqs_normalized = freqs/1e9
        nchans = len(freqs)
        print('nchans', nchans)

        thermal_noise1, fig_phases1 = futils.get_thermal_noise(vis1, ant_idxs, antmap, title = f'Pre-Cut | Batch {batch_start_ts} | Int {T_SPECTRA*acclen:.2f} s | Pulse {pulse_start_ts}', mask=None, T_SPECTRA = T_SPECTRA, acclen=acclen)
        fig_phases1.savefig(os.path.join(path_pulse, 'phases1.png'))
        plt.close(fig_phases1)

        fig_noise1, ax = plt.subplots()
        for bl in range(nblines):
            ax.semilogy(median_filter(thermal_noise1[bl,:],10),label=f'bl id {bl}')
        plt.title(f"Median-filt. phase noise, all baselines, int. time {T_SPECTRA*acclen:4.2f}s")
        plt.ylabel(r"$\sigma_\phi$ (rad)")
        plt.xlabel("Time")
        plt.ylim(0.05,2)
        plt.grid(True)
        fig_noise1.savefig(os.path.join(path_pulse, 'thermal_noise1.png'))
        plt.close(fig_noise1)

        start_cut, end_cut = futils.find_lowest_noise(thermal_noise1, window_size = window_size)
        print('cut times', start_cut, end_cut)
        start_spectra, end_spectra = start_cut*acclen, end_cut*acclen
        print('cut spectra', start_spectra, end_spectra)
        pulse_start_spec_cut = pulse_start_spec+start_spectra*osamp
        pstart2 = UTC_per_spec*pulse_start_spec_cut + UTC_offset
        print('fitted start time after cut', pstart2)
        vis2 = vis1[:,start_cut:end_cut,:]
        print('cut vis shape', vis2.shape)
        ntimes = end_cut-start_cut

        #plot fully cut pulse, get thermal noise
        thermal_noise2, fig_phases2 = futils.get_thermal_noise(vis2, ant_idxs, antmap, title = f'Post-Cut | Batch {batch_start_ts} | Int {T_SPECTRA*acclen:.2f} s | Pulse {pulse_start_ts}',  mask=None, acclen = acclen)
        fig_phases2.savefig(os.path.join(path_pulse, 'phases2.png'))
        plt.close(fig_phases2)

        #plot the thermal noise (smoothed a bit)
        fig_noise2, ax = plt.subplots()
        for bl in range(nblines):
            ax.semilogy(median_filter(thermal_noise2[bl,:],10),label=f'bl id {bl}')
        plt.title(f"Median-filt. phase noise, all baselines, int. time {T_SPECTRA*acclen:4.2f}s")
        plt.ylabel("$\sigma_\phi$ (rad)")
        plt.xlabel("Time (s)")
        plt.ylim(0.05,2)
        plt.grid(True)
        fig_noise2.savefig(os.path.join(path_pulse, 'thermal_noise2.png'))
        plt.close(fig_noise2)

        #==================================================================
        # Determine the starting guess for peak tracking fit

        #find 5 time-sample window with lowest phase noise
        start_cut_guess, end_cut_guess = futils.find_lowest_noise(thermal_noise2, window_size = 5)
        print(f'guesses using timestamps from {start_cut_guess} to {end_cut_guess}')

        #get an initial guess for tau
        avg_multi_vis = np.mean(vis2[:, start_cut_guess:end_cut_guess, :], axis=1) #starting position is arbitrary for now. make lowest phase noise region?
        print(avg_multi_vis.shape)
        avg_multi_phase = np.unwrap(np.angle(avg_multi_vis), axis = 1)

        #get that noise by fitting a line and getting std of residuals
        noise_matrix_avg = np.zeros((1,nblines),dtype='float64')
        for bl in range(nblines):
            y = avg_multi_phase[bl,:]
            m, c = np.polyfit(freqs,y, 1)
            residual = y - (m*freqs + c)
            noise_matrix_avg[0,bl] = np.std(residual)
        # print(noise_matrix_avg)
        print(avg_multi_phase.shape)

        data_matrix = avg_multi_phase.T.reshape(1,nchans,nblines)
        print(data_matrix.shape)

        Ag = futils.get_grammian(nant)
        AtA2,Atd2 = futils.get_AtA_Atd(data_matrix,
                                Ag,
                                noise_matrix_avg**2,
                                freqs_normalized,
                                nant,
                                nchans,  
                                1,
                                fit_constant=True
                                )

        AtA_inv2 = np.linalg.inv(AtA2)
        mfit2 = AtA_inv2 @ Atd2

        # Total parameters for tau: ntime * (nant-1)
        bs = nant - 1
        taus_guess0 = mfit2[:1 * bs]
        phi_guess0 = mfit2[1 * bs:]
        errs_guess0 = np.sqrt(np.diag(AtA_inv2))
        print('initial guesses for taus:', taus_guess0)
        print('phis', phi_guess0)
        print('errors on phi, tau', errs_guess0)

        prediction = -2*np.pi*taus_guess0[0]*freqs_normalized
        phis_guess0 = np.mean(prediction - avg_multi_phase[0,:])
        print(phis_guess0)

        #plot the ramp guesses vs data
        title = 'Guess Ramps'
        nrows = int(np.ceil(nblines / ncols))
        fig_guesses, ax = plt.subplots(nrows, ncols,figsize=(10, 3*nrows),sharex=True)
        fig_guesses.subplots_adjust(left=0.07,right=0.93,bottom=0.05,top=0.92,wspace=0.18,hspace=0.20)
        fig_guesses.text(0.5, 0.97,title,ha="center",va="top",fontsize=14,fontweight="bold")
        ax = np.ravel(ax)

        blnum=0
        print(nblines)
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
                    phase_bline = avg_multi_phase[blnum, :]
                    phase_pred = np.unwrap(np.angle(np.exp(2j*np.pi*rel_delay*freqs_normalized)))
                    ax[blnum].plot(phase_bline - phase_bline[0], label='Data')
                    ax[blnum].plot(phase_pred - phase_pred[0], label='Coarse Fit')
                    ax[blnum].legend()
                    blnum+=1
        fig_guesses.savefig(os.path.join(path_pulse, 'ramps_guess.png'))
        plt.close(fig_guesses)

        # get noise and fit for each timestamp. plot the weights
        noise_var = median_filter(thermal_noise2[:,:],3,axes=1)**2   #why a median filter of just 3??
        print('noise var shape', noise_var.shape)

        weights = np.sqrt(np.size(noise_var)/noise_var/np.sum(1/noise_var))
        angle = np.angle(vis2)

        phase_unwrapped = angle
        weight_matrix = weights.T.copy()
        for bl in range(nblines):
            plt.semilogy(weights[bl,:],label=f'bl id {bl}')
        plt.legend()

        #make baseline fastest moving axis for convenience
        data_matrix = phase_unwrapped.transpose(1,2,0).copy()
        noise_matrix = noise_var.T.copy()
        print('data_matrix shape', data_matrix.shape)
        print('weight_matrix shape', weight_matrix.shape)
        print('noise matrix shape', noise_matrix.shape)

        # make frequencies order 1 for fitting purposes
        freqs_normalized = freqs.copy()/1e9

        # set up some global variables
        assert ntimes == data_matrix.shape[0]
        assert nchans == data_matrix.shape[1]
        assert nblines == data_matrix.shape[2]
        assert nant == len(ant_idxs) 
        print('ntimes', ntimes)
        print('nchans', nchans)
        print('nblines', nblines)
        #print("nant used:", nant_used)

        #check the cost surface to see if it makes sense
        cost1, cost2, fig_cost_guesses = futils.cost_surface(data_matrix, taus_guess0, freqs_normalized, weight_matrix, antmap, antidx=0, timeidx=0, N1=10001, N2=40)
        fig_cost_guesses.savefig(os.path.join(path_pulse, 'cost_surface_guesses.png'))
        plt.close(fig_cost_guesses)

        #==================================================================
        # Perform the peak tracking fit

        # array of fitted taus for each time, for each (non-reference) antenna
        taus_carrier = np.zeros((ntimes,nant-1),dtype='float64')
        taus_carrier_errs = np.zeros_like(taus_carrier)

        # make some guess about fitted taus (for one sample time)
        taus_guess = taus_guess0
        print('starting guess:', taus_guess0)

        for tt in range(ntimes):
            ydata = data_matrix[tt,:,:].ravel()
            weights = weight_matrix[tt,:]
            t1=time.time()

            tau_fit_params = least_squares(futils.func, taus_guess, args=(ydata, freqs_normalized, weights),
                                                                jac=futils.jac,
                                                                method='lm',
                                                                ftol=1e-06, 
                                                                xtol=1e-06, 
                                                                gtol=1e-06, 
                                                                loss='linear')

            t2=time.time()
            taus_carrier[tt,:] = tau_fit_params['x']
            taus_guess = tau_fit_params['x']
            #taus_carrier_errs[tt,:] = tau_fit_params['jac']
            
            print(f"done tt={tt}, time = {t2-t1:5.3f}")
            #print('new guess:', taus_guess)

        fig_tausz, ax = plt.subplots(figsize=(8,5))
        tausz_carrier = np.zeros_like(taus_carrier)
        for antidx in range(nant-1):
            ts = taus_carrier[:,antidx] - taus_carrier[0,antidx]
            tausz_carrier[:,antidx] = ts
            ax.plot(ts, label=antmap[antidx+1])
            ax.set_xlabel('Time Sample (~1 s)')
            ax.set_ylabel('Relative Drift (ns)')
        ax.legend()
        plt.suptitle(f'Zeroed Carrier Tracking Delays (ref={antmap[0]})')
        plt.tight_layout()
        fig_tausz.savefig(os.path.join(path_pulse, 'taus_zeroed.png'))
        plt.close(fig_tausz)

        #==================================================================
        # Align the phases and stack into one ramp per baseline
        
        title = f'Aligned Phases | Batch {batch_start_ts} | Int {T_SPECTRA*acclen:.2f} s | Pulse {pulse_start_ts}'
        nrows = int(np.ceil(nblines / ncols))
        fig_phases2_aligned, ax = plt.subplots(nrows, ncols,figsize=(10, 3*nrows),sharex=True,sharey=True)
        fig_phases2_aligned.subplots_adjust(left=0.07,right=0.93,bottom=0.05,top=0.92,wspace=0.08,hspace=0.20)
        ax = np.ravel(ax)
        fig_phases2_aligned.text(0.5, 0.97,title,ha="center",va="top",fontsize=14,fontweight="bold")

        blnum=0
        print(nblines)
        x = np.arange(nchans)
        phased_vis = np.empty_like(vis2[:,:,:])
        center_chan = nchans//2
        for i in range(len(ant_idxs)):
                for j in range(i+1, len(ant_idxs)):
                    ai = ant_idxs[i]
                    aj = ant_idxs[j]
                    if i==0:
                        tau1=0
                    else:
                        tau1 = tausz_carrier[:,i-1]
                    tau2 = tausz_carrier[:,j-1]
                    rel_delay = tau1-tau2
                    ax[blnum].set_title(f"{antmap[ai]}-{antmap[aj]} (id {blnum})")
                    #the actual data we will use
                    phased_vis[blnum,:,:] = vis2[blnum,:,:]*np.exp(-2j*np.pi*freqs_normalized[None,:]*rel_delay[:,None])
                    #plotting data centered at zero so it looks better
                    p = np.unwrap(np.angle(phased_vis[blnum]), axis=1)
                    pc = p - p[:, center_chan, None]
                    pc = (pc + np.pi) % (2 * np.pi) - np.pi
                    img=ax[blnum].imshow(pc,aspect='auto',interpolation='none',cmap='RdBu')
                    cbar=plt.colorbar(img,ax=ax[blnum])
                    blnum+=1
        fig_phases2_aligned.savefig(os.path.join(path_pulse, 'phases2_aligned.png'))
        plt.close(fig_phases2_aligned)

        #==================================================================
        #mask on noisy parts of aligned phases
        mask_all = np.zeros((nblines, ntimes), dtype = bool)
        for blid in range(nblines):
            mask_all[blid, :], _ = rfitools.time_mask(np.abs(phased_vis[blid,:,:]), 1, allow_positive=True)

        fig_masking, ax = plt.subplots(nblines, 3, figsize = (10, 3*nblines), sharey=True, sharex=True)
        for blid in range(nblines):
            ax[blid, 1].set_title(f'{antmap[triuidx[0][blid]]}-{antmap[triuidx[1][blid]]} | id {blid}')
            im = ax[blid, 0].imshow(np.abs(phased_vis[blid,:,:]), interpolation = 'none', aspect = 'auto')
            im = ax[blid, 1].imshow(np.angle(phased_vis[blid, :, :]),aspect='auto',interpolation='none',cmap='RdBu')
            im = ax[blid, 2].imshow(np.angle(phased_vis[blid, :, :])*(1-mask_all)[blid, :, None],aspect='auto',interpolation='none',cmap='RdBu')
        fig_masking.savefig(os.path.join(path_pulse, 'vis_masking.png'))
        plt.close(fig_masking)

        
        valid = ~mask_all
        #p = np.unwrap(np.angle(phased_vis), axis=2)
        #avg_unmasked = np.mean(p, axis=1)
        avg_vis_unmasked = np.mean(phased_vis, axis=1)
        avg_vis_masked = np.sum(phased_vis * valid[:, :, None], axis=1)/ np.sum(valid, axis=1)[:, None]
        avg_phase_unmasked = np.unwrap(np.angle(avg_vis_unmasked), axis = 1)
        avg_phase_masked = np.unwrap(np.angle(avg_vis_masked), axis = 1)

        nrows = int(np.ceil(nblines / ncols))
        fig_masked_ramps, ax = plt.subplots(nrows, ncols,figsize=(10, 3*nrows),sharex=True)
        fig_masked_ramps.subplots_adjust(left=0.07,right=0.93,bottom=0.05,top=0.92,wspace=0.18,hspace=0.20)
        ax = np.ravel(ax)
        fig_masked_ramps.text(0.5, 0.97,f'Ramps with/without Masking | Batch {batch_start_ts}',ha="center",va="top",fontsize=14,fontweight="bold")
        for blid in range(nblines):
            ax[blid].plot(avg_phase_unmasked[blid, :], label='Without')
            ax[blid].plot(avg_phase_masked[blid, :]+1, label='With+1')
            ax[blid].legend()
        fig_masked_ramps.savefig(os.path.join(path_pulse, 'ramps_masked.png'))
        plt.close(fig_masked_ramps)

        avg_multi_phase = avg_phase_masked.T.copy()
        print('averaged vis shape', avg_multi_phase.shape)

        noise_matrix_avg = np.zeros((1,nblines),dtype='float64')
        for bl in range(nblines):
            y = avg_multi_phase[:, bl]
            m, c = np.polyfit(freqs,y, 1)
            residual = y - (m*freqs + c)
            noise_matrix_avg[0,bl] = np.std(residual)
        print('averaged ramp noise', noise_matrix_avg)

        #==================================================================
        # Fit on stacked ramps 
        data_matrix_stacked = avg_multi_phase.reshape(1,nchans,nblines)
        print(data_matrix_stacked.shape)

        Ag = futils.get_grammian(nant)
        print(data_matrix_stacked.shape)

        AtA2,Atd2 = futils.get_AtA_Atd(data_matrix_stacked,
                                    Ag,
                                    noise_matrix_avg**2,
                                    freqs_normalized,
                                    nant,
                                    nchans,
                                    1, #ntimes
                                    fit_constant=True)

        AtA_inv2 = np.linalg.inv(AtA2)
        mfit2 = AtA_inv2 @ Atd2
        print(mfit2)

        # Total parameters for tau: ntime * (nant-1)
        bs = nant - 1
        tau0_carrier = mfit2[:1 * bs]
        phi0_carrier = mfit2[1 * bs:]
        errs_all = np.sqrt(np.diag(AtA_inv2))
        tau0_carrier_errs = errs_all[:nant-1]

        print('net offsets\n', tau0_carrier)
        print('errors all\n', errs_all)

        wts = np.ones([ntimes])
        threesigma = errs_all[0]*3
        #check that we are on the correct carrier peak
        N2 = max(20, threesigma/2)
        cost1, cost2, fig_cost_soln = futils.cost_surface(data_matrix, tau0_carrier, freqs_normalized, wts, antmap, antidx = 0, timeidx=0, N1=10001, N2=N2, err=threesigma)
        fig_cost_soln.savefig(os.path.join(path_pulse, 'cost_surface_soln.png'))
        plt.close(fig_cost_soln)

        taus = tausz_carrier.T + tau0_carrier[:, None]
        taus_errs_all = errs_all[:bs]
        print(taus)

        with h5py.File(os.path.join(path_finetiming, f'timing_solution.h5'), 'a') as f:
            if fname_data not in f:
                grp = f.create_group(fname_data)
            else:
                grp = f[fname_data]
            if 'taus' in grp:
                del grp['taus']
            taus = grp.create_dataset('taus', data=taus)
            if 'thermal_noise' in grp:
                del grp['thermal_noise']
            thermal_noise = grp.create_dataset('thermal_noise', data=thermal_noise2)
            if 'errs' in grp:
                del grp['errs']
            taus_errs = grp.create_dataset('errs', data = taus_errs_all)
            taus.attrs['starting_specnum'] = pulse_start_spec_cut


    #LOOK OVER THIS THING; HOW TO SAVE MORE EFFICIENTLY
    if args.save_to_soln:
        print('now saving the final thing into timing solution object')

        nvis = window_size
        ntimes = nvis*len(list_pulses)
        spectra = np.zeros(ntimes)
        taus = np.zeros((nant-1, ntimes))
        noise_all = np.zeros((nblines, ntimes))
        errs_tau = np.zeros((nant-1, ntimes))

        pass_ctr = 0
        int_spec = acclen*osamp #DOUBLE CHECK THIS
        with h5py.File(f'/scratch/thomasb/batch_{batch_start_ts}/fine_timing/timing_solution.h5', 'r') as f:
            for name, obj in f.items():
                #specnum stuff
                start_spec = obj['taus'].attrs['starting_specnum']
                #beware: we want the CENTRAL spectrum number, not the STARTING one, we accumulate through entire integration time
                s = np.arange(int(start_spec+int_spec/2), int(start_spec + (nvis+1/2)*int_spec), int_spec)
                spectra[pass_ctr*nvis: (pass_ctr+1)*nvis] = s
                #taus stuff
                taus_old = obj['taus'][:]
                print('taus shape', taus_old.shape)
                taus[:, pass_ctr*nvis: (pass_ctr+1)*nvis] = taus_old
                #noise and errors
                noise = obj['thermal_noise'][:]
                errs = obj['errs'][:]
                print('noise shape', noise.shape)
                print('errors shape', errs_tau.shape)
                errs_tau[:, pass_ctr*nvis:(pass_ctr+1)*nvis] = errs[:, None]
                noise_all[:, pass_ctr*nvis:(pass_ctr+1)*nvis] = noise
                pass_ctr += 1

        with h5py.File(f'/scratch/thomasb/timing_solution/batch_{batch_start_ts}.h5', "w") as f:
            f.create_dataset("spectra", data=spectra)
            f.create_dataset("taus", data=taus) #shape (nblines, ntimes)
        
        #by convention the reference antenna is always the lowest index that is present in the config file
        consensus_offset_dict = {}
        for ai in range(1, len(ant_names)):
            consensus_offset_dict[ant_names[ai]] = coarse_delays[ai]

        index_entry = {"start": batch_start_ts,
                        "end": batch_end_ts,
                        "ref_ant": ant_names[0],
                        "non_ref_ants": ant_names[1:],
                        "UTC_per_spec": UTC_per_spec,
                        "UTC_offset": UTC_offset,
                        "consensus_offsets": consensus_offset_dict
                    }
        #save new timing solution entry into whole dict
        print('saving into index file')
        with open('/scratch/thomasb/timing_solution/index.json', 'r') as f:
            index = json.load(f)
        index[f'batch_{batch_start_ts}'] = index_entry
        with open('/scratch/thomasb/timing_solution/index.json', 'w') as f:
            json.dump(index, f, indent=4)

