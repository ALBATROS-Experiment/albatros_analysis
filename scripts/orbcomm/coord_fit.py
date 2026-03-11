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
import numba as nb
import time
import importlib
import json
from scipy.optimize import minimize
from skyfield.api import load, wgs84
import cupy
import fitting_helper as fh

def objective_coords(
    pos_offset,
    time_offset,
    t_start,
    t_end,
    data_slice,
    ant_idxs,
    fit_ant_idx, 
    ant_coords,
    freq,
    satID,
    plot=False,
    bb_spectrum_T=4096/250e6,
    osamp=64,
    acclen = 1024
    ):

    """
    Gives chi squared given some antenna coordinates and a timing discrepancy

    t_start needs to coincide with the first spectrum of the data.
    however, t_end will determine how much the data is cut by, so as long as it's
    BEFORE the last spectrum in the data, we're all good.

    this is done to make cutting garbage time at the end easier, and minimize interference with the data array
    """
    print(pos_offset)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        plt.axhline(6.28, ls='--',c='black')
        plt.axhline(-6.28,ls='--', c='black')
        plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
            "legend.fontsize": 14
        })

    process_start_ts = time.time()

    a1_coords = ant_coords[fit_ant_idx]
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

    for j in range(len(ant_idxs)):
        nonfit_ant_idx = ant_idxs[j]
        a2_coords = ant_coords[nonfit_ant_idx]
        dist = fh.haversine(a1_coords, a2_coords)
        sum_wt += dist


        dly = outils.get_sat_delay2(
                            [a1_coords[0]+1e-4*pos_offset[0], a1_coords[1]+1e-4*pos_offset[1], a1_coords[2]+pos_offset[2]],
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

        spec2_phased = np.empty_like(data_slice[nonfit_ant_idx, 0, :])
        spec2_phased = fh.apply_delay_1d(data_slice[nonfit_ant_idx, 0, :], spec2_phased, -delay, freq)
        Vxx = fh.xcorr_avg_1d(data_slice[fit_ant_idx,0,:], spec2_phased, acclen)
        spec2_phased = fh.apply_delay_1d(data_slice[nonfit_ant_idx, 1, :], spec2_phased, -delay, freq)
        Vyy = fh.xcorr_avg_1d(data_slice[fit_ant_idx,1,:], spec2_phased, acclen)
        V=(Vxx+Vyy)/2


        if plot:
            ax.plot(np.unwrap(np.angle(V))-np.angle(V)[0],label=f'{fit_ant_idx}-{nonfit_ant_idx}')
            plt.legend()

        Vnew = np.exp(1j*np.angle(V))
        chisq-= dist * np.abs(np.mean(Vnew))**2 #cut it off where snr drops
    chisq/=sum_wt
    print('time for objective compute:', time.time()-process_start_ts)

    if plot:
        ax.set_xlabel(f"Visibility Chunk(~{np.round(T_SPECTRA*acclen, decimals=2)} s)")
        ax.set_ylabel("Unwrapped Phase (radians)")
        ax.set_title(f"Phase at Baslines (offset={time_offset}, sat={satID})")
        ax.grid(True, alpha=0.3)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )

    plt.tight_layout()
    print("chisq", chisq)
    return chisq



def objective_coords_all(pos_offset, 
                        data,
                        pulses,
                        cutter,
                        ant_idxs, 
                        fit_ant_idx, 
                        ant_coords,
                        bb_spectrum_T=4096/250e6,
                        osamp=64,
                        acclen = 1024):
    assert len(data) == len(pulses)
    assert len(data) == len(cutter)
    chisq = 0
    for i, pulse in enumerate(pulses):
        pulse_start_ts = pulse['t_start']
        pulse_start_ts = pulse['t_end']
        satID = pulse['sat']
        fname = f"data_raw_osamp={osamp}_start={pulse_start_ts}_end={pulse_end_ts}_chans={compute_chans[0]}:{compute_chans[-1]}.npy"
        
        chisq += objective_coords(
                        pos_offset,
                        info[2],
                        info[0],
                        info[1],
                        data[i],
                        ant_idxs,
                        fit_ant_idx, 
                        ant_coords,
                        info[4],
                        info[3],
                        plot=False,
                        bb_spectrum_T=bb_spectrum_T,
                        osamp=osamp,
                        acclen = acclen
                        )
    return chisq


def cost_curve(
    lats,
    lons,
    alt_offset,
    time_offset,
    t_start,
    t_end,
    data_slice,
    ant_idxs,
    fit_ant_idx,
    ant_coords,
    freq,
    satID,
    bb_spectrum_T=4096/250e6,
    osamp=64,
    include_fig=False
):

    assert len(lons) == len(lats)
    nstep = len(lons)
    #lats = np.linspace(lats[0], lats[1], nstep)
    #lons = np.linspace(lons[0], lons[1], nstep)
    chisqs = np.zeros((nstep, nstep) ,dtype='float64')

    for i, lon in enumerate(lats):
        for j, lat in enumerate(lons):
            print(f'\n----ITERATION {j, i}-----')
            chisqs[i, j]=objective_coords(
                                [lat, lon, alt_offset],
                                time_offset,
                                t_start,
                                t_end,
                                data_slice,
                                ant_idxs,
                                fit_ant_idx, 
                                ant_coords,
                                freq,
                                satID,
                                plot=False
            )
    
    if include_fig:
        fig, ax = plt.subplots()
        im = ax.imshow(
            chisqs,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            extent=[lats[0], lats[-1], lons[0], lons[-1]]  
        )
        plt.rcParams.update({
                    "font.size": 16,
                    "axes.labelsize": 18,
                    "axes.titlesize": 20,
                    "xtick.labelsize": 14,
                    "ytick.labelsize": 14,
                    "figure.titlesize": 22,
                    "legend.fontsize": 14
                })
        ax.set_xlabel('Lat Offset ()')
        ax.set_ylabel('Lon Offset ()')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\chi^2$')
        ax.plot(
            0, 0,
            marker='x',
            color='red',
            markersize=10,
            markeredgewidth=2)
        plt.tight_layout()
        return chisqs, fig
    else:
        return chisqs



def cost_curve_all(lats,
                    lons,
                    alt_offset,
                    data,
                    metadata,
                    ant_idxs,
                    fit_ant_idx,
                    ant_coords,
                    bb_spectrum_T=4096/250e6,
                    osamp=64,
                    include_fig=False
                ):

    assert len(lons) == len(lats)
    nstep = len(lons)
    #lats = np.linspace(lats[0], lats[1], nstep)
    #lons = np.linspace(lons[0], lons[1], nstep)
    chisqs = np.zeros((nstep, nstep) ,dtype='float64')

    for i, lon in enumerate(lats):
        for j, lat in enumerate(lons):
            print(f'\n----ITERATION {j, i}-----')
            chisqs[i, j]=objective_coords_all(
                                    [lat, lon, alt_offset],
                                    data,
                                    metadata,
                                    ant_idxs, 
                                    fit_ant_idx, 
                                    ant_coords,
                                    bb_spectrum_T=bb_spectrum_T,
                                    osamp=osamp,
                                    acclen = acclen
                                    )
    
    if include_fig:
        fig, ax = plt.subplots()
        im = ax.imshow(
            chisqs,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            extent=[lats[0], lats[-1], lons[0], lons[-1]]  
        )
        plt.rcParams.update({
                    "font.size": 16,
                    "axes.labelsize": 18,
                    "axes.titlesize": 20,
                    "xtick.labelsize": 14,
                    "ytick.labelsize": 14,
                    "figure.titlesize": 22,
                    "legend.fontsize": 14
                })
        ax.set_xlabel('Lat Offset ()')
        ax.set_ylabel('Lon Offset ()')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\chi^2$')
        ax.plot(
            0, 0,
            marker='x',
            color='red',
            markersize=10,
            markeredgewidth=2)
        plt.tight_layout()
        return chisqs, fig
    else:
        return chisqs

acclen=1024
bb_spectrum_T = 4096/250e6
osamp = 64
T_SPECTRA = bb_spectrum_T * osamp

nstep = 51
lats = np.linspace(-5, 5, nstep)
lons = np.linspace(-5, 5, nstep)

ant_idxs = ant_idxs = [0, 2, 3, 4, 5, 6]

config_path = '/home/thomasb/albatros_analysis/scripts/orbcomm/config/config_batch2.json'

with open(config_path, "r") as f:
    config = json.load(f)
ant_coords = []
for i, (ant, details) in enumerate(config["antennas"].items()):
    ant_coords.append(details["coordinates"])
batch_start_ts = config["correlation"]["start_timestamp"]
batch_end_ts = config["correlation"]["end_timestamp"]
chanstart = config["frequency"]["start_channel"]
chanend = config["frequency"]["end_channel"]
channels_old = np.arange(chanstart, chanend)

batch_path = f'/scratch/thomasb/full_timing_discrepancies_{batch_start_ts}'
pulse_fname = 'pulses2.json'
cutter_fname = 'cutting.json'
times_fname = 'times.json'

with open(os.path.join(batch_path, pulse_fname), "r") as f:
    pulses = json.load(f)

with open(os.path.join(batch_path, cutter_fname), "r") as f:
    cutter = json.load(f)

with open(os.path.join(batch_path, times_fname), "r") as f:
    time_map = json.load(f)

fitting_path = os.path.join(batch_path, 'coord_fitting')
os.makedirs(fitting_path, exist_ok=True)

chisq_tot = np.zeros((nstep, nstep))

for pulse in pulses:
    pulse_start_ts_file = pulse['t_start']
    pulse_end_ts_file = pulse['t_end']
    satID = pulse['sat']

    det_chan = channels_old[pulse['channel']] #if want to compute with fewer channels
    if det_chan%2 == 0:
        compute_chans = np.array([det_chan-2, det_chan-1, det_chan, det_chan+1])
    else:
        compute_chans = np.array([det_chan-1, det_chan, det_chan+1, det_chan+2])
    print('old compute chans', compute_chans)
    
    fname = f"data_raw_osamp={osamp}_start={pulse_start_ts_file}_end={pulse_end_ts_file}_chans={compute_chans[0]}:{compute_chans[-1]}.npy"
    cuts = cutter[fname]
    new_chan = cuts["new_channel"]
    spectra_start = cuts["spectra_start"]
    spectra_end = cuts["spectra_end"]
    disk_path = os.path.join(f'/scratch/thomasb/full_timing_discrepancies_{batch_start_ts}/{fname}')
    data = np.load(disk_path)
    data_cut = data[:, :, spectra_start:-spectra_end, new_chan]

    nchans = data.shape[3]
    sat_freqs = 250e6 - ((np.arange(nchans)/osamp + compute_chans[0])/(4096/250e6))
    freq = sat_freqs[new_chan]

    pulse_start_ts = pulse_start_ts_file + T_SPECTRA*spectra_start
    pulse_end_ts =   pulse_end_ts_file -   T_SPECTRA*spectra_end -2

    time_offset = time_map[fname]['offset_fitted']/1000
    print('TIME OFFSET', time_offset)

    chisqs, cost_fig = cost_curve(lats,
                                lons,
                                0,              #alt offset
                                time_offset,    #time offset
                                pulse_start_ts, #tstart
                                pulse_end_ts,   #tend
                                data_cut,       #data slice
                                ant_idxs,       #ant indices
                                1,              #fit ant idx
                                ant_coords,    
                                freq,  
                                satID, 
                                bb_spectrum_T=bb_spectrum_T,
                                osamp=osamp,
                                include_fig = True)
    chisq_tot += chisqs
    cost_fig.savefig(os.path.join(fitting_path, f'coordfit_plot_{pulse_start_ts_file}.png'))
    plt.close(cost_fig)
    del(data)
    del(data_cut)
    del(chisqs)

fig, ax = plt.subplots()
im = ax.imshow(
    chisq_tot,
    origin='lower',
    aspect='auto',
    cmap='viridis',
    extent=[lats[0], lats[-1], lons[0], lons[-1]]  
)
plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
            "legend.fontsize": 14
        })
ax.set_xlabel('Lat Offset ()')
ax.set_ylabel('Lon Offset ()')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\chi^2$')
ax.plot(
    0, 0,
    marker='x',
    color='red',
    markersize=10,
    markeredgewidth=2)
plt.tight_layout()
plt.savefig(os.path.join(fitting_path, f'all_coordfit_plot.png'))