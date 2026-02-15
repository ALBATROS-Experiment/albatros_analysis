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
                        metadata,
                        ant_idxs, 
                        fit_ant_idx, 
                        ant_coords,
                        bb_spectrum_T=4096/250e6,
                        osamp=64,
                        acclen = 1024):
    assert len(data) == len(metadata)
    chisq = 0
    for i, info in enumerate(metadata):
        # print(f'entry {i}')
        # print(pos_offset)
        # print(info[2])
        # print(info[0])
        # print(info[1])
        # print(data[i])
        # print(ant_idxs)
        # print(fit_ant_idx) 
        # print(ant_coords)
        # print(info[4])
        # print(info[3])
        # sys.exit()
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

config_path = '/home/thomasb/albatros_analysis/scripts/xcorr/config/config_bright_test.json'
disk_path = '/scratch/thomasb/raw_1753216650:1753217000_bit=1_ant=7_pol=2_cha=1834:1852_tim=650_upx=64_acc=512_ipfb=0.4_complex64_regular_20260110T114008.npy'

out_path = '/scratch/thomasb'
fig_path = os.path.join(out_path, f'coord_fitting')
os.makedirs(fig_path, exist_ok=True)

with open(config_path, "r") as f:
    config = json.load(f)
ant_coords = []
for i, (ant, details) in enumerate(config["antennas"].items()):
    ant_coords.append(details["coordinates"])
batch_start_ts = config["correlation"]["start_timestamp"]
batch_end_ts = config["correlation"]["end_timestamp"]

acclen=1024
bb_spectrum_T = 4096/250e6
osamp = 64
T_SPECTRA = bb_spectrum_T * osamp

satID = 57166
chan_new = 170

data_all = np.load(disk_path, mmap_mode='r')
data = data_all[:, :, 60000:, chan_new]  #find way to properly cut data off the end to save space later

nchans = data_all.shape[3]
sat_freqs = 250e6 - ((np.arange(nchans)/osamp + 1834)/(4096/250e6))



metadata_list = [[1753216712.91456, 1753216850, -1.265, 57166, sat_freqs[chan_new]]]
data_list = [data]


nstep = 51
lats = np.linspace(-5, 5, nstep)
lons = np.linspace(-5, 5, nstep)

ant_idxs = ant_idxs = [0, 2, 3, 4, 5, 6]

print(data.shape)


chisqs, cost_fig = cost_curve_all(lats,
                                lons,
                                0,
                                data_list,
                                metadata_list,
                                ant_idxs,
                                1,
                                ant_coords,
                                bb_spectrum_T=bb_spectrum_T,
                                osamp=osamp,
                                include_fig = True)

cost_fig.savefig(os.path.join(fig_path, 'test_coord_fitting.png'))







