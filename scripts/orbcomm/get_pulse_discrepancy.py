#system
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
from matplotlib import pyplot as plt
from datetime import datetime as dt
from skyfield.api import load, wgs84
#utils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.utils import sat_utils as sutils
from albatros_analysis.src.utils import finetiming_utils as futils
#functions and helpers
from albatros_analysis.scripts.xcorr.fine_timing  import dump_upchan_baseband
from albatros_analysis.scripts.xcorr import helper as hxc
from scipy.optimize import minimize


def objective_times(time_offset,
                    t_start, t_end,
                    data_slice, ant_idxs,
                    ant_coords, freqs,
                    satID,
                    coherent = False,
                    plot=False,
                    bb_spectrum_T=4096/250e6, osamp=64, acclen = 1024,
                    ylims = None
                    ):

    """ 
    t_start needs to coincide with the first spectrum of the data.
    however, t_end will determine how much the data is cut by, so as long as it's
    BEFORE the last spectrum in the data, we're all good.

    this is done to make cutting garbage time at the end easier, and minimize interference with the data array
    """

    process_start_ts = time.time()
    #turn time_offset into float when passed in scipy.optimize.minimize
    if type(time_offset) == np.ndarray:
        time_offset = time_offset[0]

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.axhline(6.28, ls='--',c='black')
        ax.axhline(-6.28,ls='--', c='black')
        if ylims != None:
            plt.ylim(ylims[0], ylims[1])

    T_SPECTRA = bb_spectrum_T * osamp
    tle_path = outils.get_tle_file(t_start, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
    sats_objects = load.tle_file(tle_path)

    nspec_tot = data_slice.shape[2]
    nspec = int((t_end-t_start)/T_SPECTRA)
    print('nspec total', nspec_tot)
    print('nspec', nspec)

    assert nspec_tot >= nspec
    assert (nspec*bb_spectrum_T) < (t_end-t_start+1)

    many_chans = False
    if data_slice.ndim == 4:
        nchans = data_slice.shape[3]
        many_chans = True
        assert len(freqs) == nchans
        data_slice = data_slice[:, :, :nspec, :].copy()
    else:
        assert len(freqs) == 1
        data_slice = data_slice[:, :, :nspec].copy()

    chisq = 0
    sum_wt = 0
    for i in range(len(ant_idxs)):
        for j in range(i+1, len(ant_idxs)):
            ai = ant_idxs[i]
            aj = ant_idxs[j]
            a1_coords=ant_coords[ai]
            a2_coords=ant_coords[aj]
            dist = sutils.get_haversine_dist(a1_coords,a2_coords)
            sum_wt += dist**2

            dly = outils.get_sat_delay2(
                                a1_coords,
                                a2_coords,
                                sats_objects,
                                t_start+time_offset,
                                int(t_end-t_start)+1,
                                satID,
                                altaz=False
                            )
            delay = np.interp(
                np.arange(0, nspec) * T_SPECTRA, np.arange(0, int(t_end-t_start)+1), dly
            )
            if many_chans:
                spec2_phased = np.empty_like(data_slice[aj,0,:,:])
                spec2_phased = futils.apply_delay(data_slice[aj,0,:,:], spec2_phased, -delay, freqs)
                Vxx = futils.xcorr_avg(data_slice[ai,0,:,:], spec2_phased, acclen)
                spec2_phased = futils.apply_delay(data_slice[aj,1,:,:], spec2_phased, -delay, freqs)
                Vyy = futils.xcorr_avg(data_slice[ai,1,:,:],spec2_phased,acclen)
                V=(Vxx+Vyy)/2
                if plot:
                    ax.plot(np.unwrap(np.angle(V[:,8]))-np.angle(V[:,0])[0],label=f'{ai}-{aj}')
                    plt.legend()
                Vnew  =  np.exp(1j*np.angle(V))
                if coherent:
                    chisq -= dist**2*np.abs(np.mean(np.sum(Vnew, axis=1)))
                else:
                    for chan_idx in range(nchans):
                        chisq -= dist**2*np.abs(np.mean(Vnew[:,chan_idx]))**2

            else:
                spec2_phased = np.empty_like(data_slice[aj,0,:])
                spec2_phased = futils.apply_delay_1d(data_slice[aj,0,:], spec2_phased, -delay, freqs[0])
                Vxx = futils.xcorr_avg_1d(data_slice[ai,0,:],spec2_phased,acclen)
                spec2_phased = futils.apply_delay_1d(data_slice[aj,1,:], spec2_phased, -delay, freqs[0])
                Vyy = futils.xcorr_avg_1d(data_slice[ai,1,:],spec2_phased,acclen)
                V=(Vxx+Vyy)/2
                if plot:
                    ax.plot(np.unwrap(np.angle(V))-np.angle(V)[0],label=f'{ai}-{aj}')
                    plt.legend()
                Vnew  =  np.exp(1j*np.angle(V))
                chisq -= dist**2*np.abs(np.mean(Vnew))**2 
    chisq/=sum_wt
    print('time for objective compute:', time.time()-process_start_ts)
    print("offset", time_offset, "chisq", chisq)
    print('nchans', len(freqs))

    if plot:
        ax.set_xlabel(f"Visibility Chunk(~{np.round(T_SPECTRA*acclen, decimals=2)} s)")
        ax.set_ylabel("Unwrapped Phase (radians)")
        ax.set_title(f"Phase at Baselines (offset={np.round(time_offset, decimals=3)}, sat={satID})")
        ax.grid(True, alpha=0.3)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )
        plt.tight_layout()
        return chisq, fig
    
    return chisq


def cost_curve(offset_start, offset_end,
                ntimes,
                batch_start_ts,batch_end_ts,
                data_slice, ant_idxs,
                ant_coords, freqs,
                satID,
                coherent = False,
                bb_spectrum_T=4096/250e6, osamp=64,
                include_fig=False, vline = None):
    """ 
    Generates cost curve of single pulse given time offset intervals and steps.
    """

    offsets = np.linspace(offset_start, offset_end, ntimes)
    chisqs = np.zeros(len(offsets),dtype='float64')
    for i,offset in enumerate(offsets):
        print(f'\n----ITERATION {i}-----')
        chisqs[i]=objective_times(offset,
                            batch_start_ts,
                            batch_end_ts,
                            data_slice,
                            ant_idxs,
                            ant_coords,
                            freqs,
                            satID,
                            plot=False,
                            coherent = coherent,
                            bb_spectrum_T=bb_spectrum_T,
                            osamp=osamp)
    
    if include_fig:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        ax.plot(offsets,chisqs)
        ax.set_xlabel('Time Discrepancy (s)')
        ax.set_ylabel(r'$\chi ^2$')
        if vline:
            plt.axvline(x=vline, color='r', linestyle='--')
        return offsets, chisqs, fig

    else:
        return offsets, chisqs, None


def get_discrepancy(config_path, 
                    disk_path,
                    satID,
                    out_path, 
                    osamp=64,
                    plot=False,
                    coherent=False):
    print(f'Starting Discrepancy Fit for {disk_path}')
    with open(config_path, "r") as f:
        config = json.load(f)
    ant_coords, dir_parents = [], []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        dir_parents.append(details["path"])
        ant_coords.append(details["coordinates"])

    batch_start_ts = config["correlation"]["start_timestamp"]
    batch_end_ts = config["correlation"]["end_timestamp"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = config["correlation"]["osamp"]
    acclen = config["correlation"]["new_acclen"]

    print('batch start ts', batch_start_ts)
    print('batch end ts', batch_end_ts)

    data_all = np.load(disk_path, mmap_mode='r')
    print(data_all.shape)

    fname = os.path.basename(disk_path)
    parts = fname.split("_")
    pulse_start_ts_file = int(parts[3].split("=")[1])
    pulse_end_ts_file = int(parts[4].split("=")[1])
    chan_range = parts[5].split("=")[1].split(".")[0]
    old_chan_start, old_chan_end = map(int, chan_range.split(":"))
    print('pulse start file:', pulse_start_ts_file)
    print(old_chan_start, old_chan_end)

    #batch_path = os.path.join(out_path, f'full_timing_discrepancies_{batch_start_ts}')
    #os.makedirs(batch_path, exist_ok=True)

    #-----------------------SET GLOBAL VARIABLE STUFF--------------
    bb_spectrum_T = 4096/250e6
    T_SPECTRA = bb_spectrum_T * osamp
    nchans = data_all.shape[3]
    sat_freqs = 250e6 - ((np.arange(nchans)/osamp + old_chan_start)/(4096/250e6))  #put as chan_start instead of 1834?
    ant_idxs = [0, 1, 2, 3, 4, 5, 6]

    #get start spectrum from file (INDIRECT! SHOULD BE DONE WHEN COMPUTING DATA RIGHT AWAY)
    #still reliant on the function giving the same starting spectrum for each iteration

    # overflow_files = hxc.get_overflow_files(batch_start_ts, batch_end_ts, dir_parents[0])
    # overflow_ct = np.sum(overflow_files<pulse_start_ts_file)
    # files, idx = butils.get_init_info(pulse_start_ts_file, pulse_start_ts_file+100, dir_parents[0]) #ref ant
    # p = bdc.BasebandFileIterator(files,0,idx,1024,None,chanstart=1834,chanend=1852,type="float")
    # start_specnum = p.spec_num_start + 2**32*overflow_ct
    
    #print('pulse start specnum', start_specnum)

    #CUTTING DATA-------------------------------------

    cutting_path = os.path.join(out_path, 'data/cutting_discrep.json')
    with open(cutting_path, "r") as f:
        cutter = json.load(f)
    cut_spectra_start = cutter[fname]["cut_spectra_start"]
    cut_spectra_end = cutter[fname]["cut_spectra_end"]
    assert cut_spectra_end>0
    chans_new = cutter[fname]["cut_chans"]
    if len(chans_new)==1:
        chans_new = np.array(chans_new)
    else:
        chans_new = np.arange(chans_new[0], chans_new[1])

    #check this works to get starting spectrum of data
    path_pulse = os.path.join(out_path, 'data/pulses.json')
    with open(path_pulse, 'r') as f:
        list_pulses = json.load(f)
    pulse = next((p for p in list_pulses if p["t_start"] == pulse_start_ts_file), None)
    start_specnum = pulse["start_specnum"]

    print(chans_new)

    freqs = sat_freqs[chans_new]
    pulse_start_ts = pulse_start_ts_file + T_SPECTRA*cut_spectra_start
    pulse_end_ts =   pulse_start_ts_file +  T_SPECTRA*cut_spectra_end 

    print('starting data slice')
    data_slice_cut = data_all[:, :, cut_spectra_start:cut_spectra_end, chans_new]
    print('done data slice')

    #========================COMPUTE======================
    #UNFITTED INITIAL CHISQ (with phase plot)------------
    print('starting initial phase plot')
    chisq_unfitted, phases_fig_unfitted = objective_times(0,
                                        pulse_start_ts, pulse_end_ts,
                                        data_slice_cut, ant_idxs,
                                        ant_coords, freqs,
                                        satID,
                                        plot=True,
                                        bb_spectrum_T=bb_spectrum_T,
                                        osamp=osamp,
                                        coherent=coherent
                                        )

    #COST CURVE----------------------
    print('starting cost curve for guess')
    offsets, chisqs, cost_fig = cost_curve(-2, 2, 
                                        101, 
                                        pulse_start_ts, pulse_end_ts, 
                                        data_slice_cut, ant_idxs, 
                                        ant_coords, freqs, 
                                        satID, 
                                        coherent=coherent,
                                        bb_spectrum_T=bb_spectrum_T, 
                                        osamp=osamp,
                                        include_fig=True
                                        )
    offset_guess = offsets[np.argmin(chisqs)]

    #FIT FOR MINIMUM USING COST CURVE GUESS--------------
    print('starting fit')
    result = minimize(objective_times, 
                        offset_guess,
                        args=(pulse_start_ts, pulse_end_ts, 
                                data_slice_cut, ant_idxs, 
                                ant_coords, freqs, 
                                satID,
                                coherent,
                                False,
                                bb_spectrum_T,
                                osamp,
                                acclen), 
                        method='Nelder-Mead',
                        tol=1e-12)
    offset_fitted = result.x[0]

    #FITTED FINAL CHISQ (with phase plot)----------------------------------
    print('starting post-fit phase plot')
    chisq_fitted, phases_fig_fitted = objective_times(offset_fitted,
                                    pulse_start_ts, pulse_end_ts,
                                    data_slice_cut, ant_idxs,
                                    ant_coords, freqs,
                                    satID,
                                    coherent=coherent,
                                    plot=True,
                                    bb_spectrum_T=bb_spectrum_T,
                                    osamp=osamp
                                    )

    #============================MAKE AND SAVE PLOTS==========================
    if plot:
    #ZOOMED COST CURVE------------------------
        # print('starting zoomed cost curve')
        # offset_rounded = np.round(offset_fitted, decimals = 2)
        # print(offset_rounded)
        # _, _, cost_fig_zoomed = cost_curve(offset_rounded-0.2, 
        #                                     offset_rounded+0.2, 
        #                                     101, 
        #                                     pulse_start_ts, pulse_end_ts, 
        #                                     data_slice_cut, ant_idxs, 
        #                                     ant_coords, freqs, 
        #                                     satID,
        #                                     coherent=coherent,
        #                                     include_fig=True,
        #                                     bb_spectrum_T=bb_spectrum_T, 
        #                                     osamp=osamp,
        #                                     vline=offset_fitted
        #                                     )

        #MAKE DIR AND SAVE---------------------
        figpath = os.path.join(out_path, "timing_discrepancies/debugplots", f"pulse_{pulse_start_ts_file}_{satID}_c={coherent}")
        os.makedirs(figpath, exist_ok=True)
        plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
            "legend.fontsize": 14
        })
        phases_fig_unfitted.savefig(os.path.join(figpath, 'unfitted_phases.png'))
        plt.close(phases_fig_unfitted)
        cost_fig.savefig(os.path.join(figpath, 'cost_curve_initial.png'))
        plt.close(cost_fig)
        #cost_fig_zoomed.savefig(os.path.join(figpath, 'cost_curve_zoomed.png'))
        #plt.close(cost_fig_zoomed)
        phases_fig_fitted.savefig(os.path.join(figpath, 'fitted_phases_unzoomed.png'))
        plt.close(phases_fig_fitted)
        
    return {
        "offset_fitted": int(offset_fitted*1000),
        "chisq_fitted": int(chisq_fitted*1000),
        #"chisq_unfitted": chisq_unfitted,
        #"offset_guess": offset_guess,
        "start_specnum": int(start_specnum),
        "pulse_start_ts": int(pulse_start_ts_file)
    }

#(if you want to run it from terminal)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("disk_path", type=str)
    parser.add_argument("satID", type=int)
    parser.add_argument("-o", "--out_path", type=str, default="/scratch/thomasb")
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-c', '--coherent', action='store_true')
    args = parser.parse_args()

    results = get_discrepancy(args.config_path, 
                            args.disk_path,
                            args.satID, 
                            args.out_path, 
                            plot=args.plot,
                            coherent=args.coherent)
    print("Results:", results)
