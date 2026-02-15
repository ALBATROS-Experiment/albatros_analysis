import os
import sys
sys.path.append(os.path.expanduser('~'))
import numpy as np 
from matplotlib import pyplot as plt
from datetime import datetime as dt
import figures as fgs
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.scripts.xcorr.fine_timing  import dump_upchan_baseband
import numba as nb
import time
import importlib
import json
from scipy.optimize import minimize
from skyfield.api import load, wgs84
import cupy
import fitting_helper as fh
import argparse

def objective_times(time_offset,
                    t_start, t_end,
                    data_slice, ant_idxs,
                    ant_coords, freq,
                    satID,
                    plot=False,
                    bb_spectrum_T=4096/250e6, osamp=64, acclen = 1024,
                    ylims = None):

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
    data_slice = data_slice[:, :, :nspec].copy()

    assert nspec_tot > nspec
    assert (nspec*bb_spectrum_T) < (t_end-t_start+1)

    chisq = 0
    sum_wt = 0
    for i in range(len(ant_idxs)):
        for j in range(i+1, len(ant_idxs)):
            ai = ant_idxs[i]
            aj = ant_idxs[j]
            a1_coords=ant_coords[ai]
            a2_coords=ant_coords[aj]
            dist = fh.haversine(a1_coords,a2_coords)
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
            spec2_phased = np.empty_like(data_slice[aj,0,:])
            spec2_phased = fh.apply_delay_1d(data_slice[aj,0,:], spec2_phased, -delay, freq)
            Vxx = fh.xcorr_avg_1d(data_slice[ai,0,:],spec2_phased,acclen)
            spec2_phased = fh.apply_delay_1d(data_slice[aj,1,:], spec2_phased, -delay, freq)
            Vyy = fh.xcorr_avg_1d(data_slice[ai,1,:],spec2_phased,acclen)
            V=(Vxx+Vyy)/2


            if plot:
                ax.plot(np.unwrap(np.angle(V))-np.angle(V)[0],label=f'{ai}-{aj}')
                plt.legend()

            
            Vnew = np.exp(1j*np.angle(V))
            chisq-= dist**2*np.abs(np.mean(Vnew))**2 
    chisq/=sum_wt
    print('time for objective compute:', time.time()-process_start_ts)
    print("offset", time_offset, "chisq", chisq)

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
                ant_coords, freq,
                satID,
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
                            freq,
                            satID,
                            plot=False,
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


def run_from_config(config_path, 
                    disk_path,
                    pulse_start_ts, pulse_end_ts, 
                    satID,
                    osamp=64,
                    output_path="/scratch/thomasb", 
                    plot=False):

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

    print('batch start ts', batch_start_ts)
    print('batch end ts', batch_end_ts)

    data_all = np.load(disk_path, mmap_mode='r')
    print(data_all.shape)

    #-----------------------SET GLOBAL VARIABLE STUFF--------------
    acclen = 1024
    bb_spectrum_T = 4096/250e6
    T_SPECTRA = bb_spectrum_T * osamp
    nchans = data_all.shape[3]
    sat_freqs = 250e6 - ((np.arange(nchans)/osamp + 1834)/(4096/250e6))  #put as chan_start instead of 1834?
    ant_idxs = [0, 2, 3, 4, 5, 6]

    #IMPROVE LATER====
    start_spectrum = fh.get_start_specnum(pulse_start_ts, dir_parents[0])

    #--------------------------PULSE DEPENDENT STUFF------------
    ####
    #how to pick the channel and how to cut this thing properly???

    cut_spectra_start = 60000
    cut_seconds_end = 150
    chan_new = 170

    freq = sat_freqs[chan_new]
    pulse_start_ts = pulse_start_ts + T_SPECTRA* cut_spectra_start
    pulse_end_ts = pulse_end_ts - cut_seconds_end

    print('starting data slice')
    data_slice_cut = data_all[:, :, cut_spectra_start:, chan_new]
    print('done data slice')


    #========================COMPUTE======================
    #UNFITTED INITIAL CHISQ (with phase plot)------------
    print('starting initial phase plot')
    chisq_unfitted, phases_fig_unfitted = objective_times(0,
                                        pulse_start_ts, pulse_end_ts,
                                        data_slice_cut, ant_idxs,
                                        ant_coords, freq,
                                        satID,
                                        plot=True,
                                        bb_spectrum_T=bb_spectrum_T,
                                        osamp=osamp,
                                        ylims=[-50, 70]
                                        )

    #COST CURVE----------------------
    print('starting cost curve for guess')
    offsets, chisqs, cost_fig = cost_curve(-2, 2, 
                                        101, 
                                        pulse_start_ts, pulse_end_ts, 
                                        data_slice_cut, ant_idxs, 
                                        ant_coords, freq, 
                                        satID, 
                                        bb_spectrum_T=bb_spectrum_T, 
                                        osamp=osamp,
                                        include_fig=True)
    offset_guess = offsets[np.argmin(chisqs)]


    #FIT FOR MINIMUM USING COST CURVE GUESS--------------
    print('starting fit')
    result = minimize(objective_times, 
                        offset_guess,
                        args=(pulse_start_ts, pulse_end_ts, 
                                data_slice_cut, ant_idxs, 
                                ant_coords, freq, 
                                satID,
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
                                    ant_coords, freq,
                                    satID,
                                    plot=True,
                                    bb_spectrum_T=bb_spectrum_T,
                                    osamp=osamp
                                    )

    #============================MAKE AND SAVE PLOTS==========================
    if plot:
    #ZOOMED COST CURVE------------------------
        print('starting zoomed cost curve')
        offset_rounded = np.round(offset_fitted, decimals = 2)
        print(offset_rounded)
        _, _, cost_fig_zoomed = cost_curve(offset_rounded-0.2, 
                                            offset_rounded+0.2, 
                                            101, 
                                            pulse_start_ts, pulse_end_ts, 
                                            data_slice_cut, ant_idxs, 
                                            ant_coords, freq, 
                                            satID,
                                            include_fig=True,
                                            bb_spectrum_T=bb_spectrum_T, 
                                            osamp=osamp,
                                            vline=offset_fitted
                                            )

        #MAKE DIR AND SAVE---------------------
        figpath = os.path.join(module_out_path, f"timing_discrepancies_{batch_start_ts}_{satID}")
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
        cost_fig.savefig(os.path.join(figpath, 'cost_curve_initial.png'))
        cost_fig_zoomed.savefig(os.path.join(figpath, 'cost_curve_zoomed.png'))
        phases_fig_fitted_unzoomed.savefig(os.path.join(figpath, 'fitted_phases_unzoomed.png'))
        phases_fig_fitted_zoomed.savefig(os.path.join(figpath, 'fitted_phases_zoomed.png'))

    return {
        "offset_fitted": offset_fitted,
        "chisq_fitted": chisq_fitted,
        "chisq_unfitted": chisq_unfitted,
        "offset_guess": offset_guess,
        "start_spectrum": start_spectrum
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("disk_path", type=str)
    parser.add_argument("satID", type=int)
    parser.add_argument("-o", "--output_path", type=str, default="/scratch/thomasb")
    args = parser.parse_args()

    results = run_from_config(args.config_path, args.disk_path, args.satID, output_path=args.output_path)
    print("Results:", results)
