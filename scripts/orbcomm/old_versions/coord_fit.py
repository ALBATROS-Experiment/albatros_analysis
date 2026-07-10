import os
import sys
sys.path.append(os.path.expanduser('~'))
#general
import numpy as np 
import numba as nb
import json
import importlib
import time
from matplotlib import pyplot as plt
from datetime import datetime as dt
#scipy
from scipy.optimize import minimize
#skyfield/astropy
from skyfield.api import load, wgs84
#in-house
import figures as fgs
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
import helper_discrepancies as hd

#=================================================
#fitting functions
#=================================================

def objective_coords(
    pos_offset,
    time_offset,
    t_start,
    t_end,
    data_slice,
    ant_idxs,
    fit_ant_idx, 
    ant_coords,
    freqs,
    satID,
    plot=False,
    bb_spectrum_T=4096/250e6,
    osamp=64,
    acclen = 1024
    ):

    """
    Gives chi squared given some antenna coordinates and a timing discrepancy
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
    nspec = data_slice.shape[2]
    print(data_slice.shape)
    print('nspec', nspec)

    many_chans = False
    if data_slice.ndim == 4:
        nchans = data_slice.shape[3]
        many_chans = True
        assert len(freqs) == nchans
    else:
        assert len(freqs) == 1
    
    chisq = 0
    sum_wt = 0

    for j in range(len(ant_idxs)):
        nonfit_ant_idx = ant_idxs[j]
        a2_coords = ant_coords[nonfit_ant_idx]
        dist = hd.haversine(a1_coords, a2_coords)
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
        if many_chans:
            spec2_phased = np.empty_like(data_slice[nonfit_ant_idx,0,:,:])
            spec2_phased = hd.apply_delay(data_slice[nonfit_ant_idx,0,:,:], spec2_phased, -delay, freqs)
            Vxx = hd.xcorr_avg(data_slice[fit_ant_idx,0,:,:], spec2_phased, acclen)
            spec2_phased = hd.apply_delay(data_slice[nonfit_ant_idx,1,:,:], spec2_phased, -delay, freqs)
            Vyy = hd.xcorr_avg(data_slice[fit_ant_idx,1,:,:],spec2_phased,acclen)
            V=(Vxx+Vyy)/2
            if plot:
                ax.plot(np.unwrap(np.angle(V[:,0]))-np.angle(V[:,0])[0],label=f'{ai}-{aj}')
                plt.legend()
            Vnew  =  np.exp(1j*np.angle(V))
            for chan_idx in range(nchans):
                chisq -= dist*np.abs(np.mean(Vnew[:,chan_idx]))**2

        else:
            spec2_phased = np.empty_like(data_slice[nonfit_ant_idx,0,:])
            spec2_phased = hd.apply_delay_1d(data_slice[nonfit_ant_idx,0,:], spec2_phased, -delay, freqs[0])
            Vxx = hd.xcorr_avg_1d(data_slice[fit_ant_idx,0,:],spec2_phased,acclen)
            spec2_phased = hd.apply_delay_1d(data_slice[nonfit_ant_idx,1,:], spec2_phased, -delay, freqs[0])
            Vyy = hd.xcorr_avg_1d(data_slice[fit_ant_idx,1,:],spec2_phased,acclen)
            V=(Vxx+Vyy)/2
            if plot:
                ax.plot(np.unwrap(np.angle(V))-np.angle(V)[0],label=f'{ai}-{aj}')
                plt.legend()
            Vnew  =  np.exp(1j*np.angle(V))
            chisq -= dist*np.abs(np.mean(Vnew))**2 
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
                        times_all,
                        data_all,
                        ant_idxs,
                        fit_ant_idx, 
                        ant_coords,
                        freqs_all,
                        satIDs_all,
                        plot=False,
                        bb_spectrum_T=4096/250e6,
                        osamp=64,
                        acclen = 1024
                        ):
    npulses = len(data_all)
    assert len(times_all)==npulses
    assert len(freqs_all)==npulses
    assert len(satIDs_all)==npulses
    chisq_tot = 0
    for i in range(npulses):
        print(f'\nGetting chisq for PULSE {i}')
        chisq_tot += objective_coords(pos_offset,
                                0,#time offset
                                times_all[i][0],
                                times_all[i][1],
                                data_all[i],
                                ant_idxs,
                                fit_ant_idx,
                                ant_coords,
                                freqs_all[i],
                                satIDs_all[i],
                                plot=False,
                                bb_spectrum_T=bb_spectrum_T,
                                osamp=osamp,
                                acclen = acclen
                                )
    return chisq_tot

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

#=================================================
#set up global stuff
#=================================================

acclen=1024
bb_spectrum_T = 4096/250e6
osamp = 64
T_SPECTRA = bb_spectrum_T * osamp

nstep = 51
lats = np.linspace(-5, 5, nstep)
lons = np.linspace(-5, 5, nstep)

ant_idxs = [0, 2, 3, 4, 5, 6]

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

#=================================================
#paths and co
#=================================================

#path for batch
path_batch = f'/scratch/thomasb/batch_{batch_start_ts}'
#path for cutting info
path_cutting = os.path.join(path_batch, 'data/cutting_discrep.json')
with open(path_cutting, "r") as f:
    cutter = json.load(f)
#path for pulse into
path_pulses = os.path.join(path_batch, 'data/pulses.json')
with open(path_pulses, "r") as f:
    pulses = json.load(f)
#path for discrepancy information (mainly starting spectrum)
path_discreps = os.path.join(path_batch, 'timing_discrepancies/times_all_incoherent.json')
with open(path_discreps, "r") as f:
    discreps = json.load(f)
#make new path for where we want to save coordinate fitting info
fitting_path = os.path.join(path_batch, 'coord_fitting')
os.makedirs(fitting_path, exist_ok=True)
#timing discrepancy corrections
#BATCH 2
UTC_per_spec = 1.638400946109685e-05
UTC_offset = 1753200128.469018


#==========================================================
#get all the required data, frequencies, satIDs, and times
#==========================================================

data_all, times_all, freqs_all, satIDs_all = [],[],[],[]
for pnum, pulse in enumerate(pulses):
    print(f'\nExtracting Data for PULSE {pnum}')
    ts_pulse_start, ts_pulse_end = pulse['t_start'], pulse['t_end']
    satID = pulse['sat']
    print('satID', satID)
    #determine baseband channels in data
    det_chan = channels_old[pulse['channel']]
    if det_chan%2 == 0:
        chans_old = np.array([det_chan-2, det_chan-1, det_chan, det_chan+1])
    else:
        chans_old = np.array([det_chan-1, det_chan, det_chan+1, det_chan+2])
    #determine the raw data filename (standardized naming method)
    fname = f"data_raw_osamp={osamp}_start={ts_pulse_start}_end={ts_pulse_end}_chans={chans_old[0]}:{chans_old[-1]}.npy"
    print('fname:', fname)
    #extract information from cutting json
    cut_pulse = cutter[fname]
    chan_new_start,chan_new_end = cut_pulse["new_chans"]
    chans_new = np.arange(chan_new_start, chan_new_end)
    spec_cut_start,spec_cut_end = cut_pulse["spectra_start"],cut_pulse["spectra_end"]
    print('spectra cut:', spec_cut_start, spec_cut_end)
    #open the data and cut right away
    disk_path = os.path.join(path_batch, 'data', fname)
    data = np.load(disk_path, mmap_mode='r')
    data_cut = data[:, :, spec_cut_start:spec_cut_end, chans_new]  #check slicing convention
    print('Data shape:', data_cut.shape)
    nant, npol, ntimes, nchans = data_cut.shape
    #get frequencies
    freqs = 250e6 - (chans_new/osamp+chans_old[0])/(4096/250e6)
    #get discrepancy-fitted pulse starting time
    discrep_pulse = discreps[fname]
    spec_pulse_start = discrep_pulse["start_spectrum"] + spec_cut_start*osamp
    ts_pulse_start_fitted = UTC_per_spec*spec_pulse_start + UTC_offset
    print('fitted starting timestamp', ts_pulse_start_fitted)

    data_all.append(data_cut)
    times_all.append([ts_pulse_start_fitted, ts_pulse_end])
    freqs_all.append(freqs)
    satIDs_all.append(satID)


#==========================================================
#get a chisq to test
#==========================================================
chisq = objective_coords_all([0,0,0],
                        times_all,
                        data_all,
                        ant_idxs,
                        1, 
                        ant_coords,
                        freqs_all,
                        satIDs_all,
                        plot=False,
                        bb_spectrum_T=4096/250e6,
                        osamp=64,
                        acclen = 1024)
print(chisq)
sys.exit()






#         #get cost curve for pulse
#         chisqs, cost_fig = cost_curve(lats,
#                                     lons,
#                                     0,              #alt offset
#                                     0,    #time offset
#                                     ts_pulse_start_fitted, #tstart
#                                     ts_pulse_end,   #tend
#                                     data_cut,       #data slice
#                                     ant_idxs,       #ant indices
#                                     1,              #fit ant idx
#                                     ant_coords,    
#                                     freqs,  
#                                     satID, 
#                                     bb_spectrum_T=bb_spectrum_T,
#                                     osamp=osamp,
#                                     include_fig = True)
#     #add cost to total
#     chisq_tot += chisqs
#     #save individual pulse cost surface figure
#     cost_fig.savefig(os.path.join(fitting_path, f'cost_surface_pulse_{pulse_start_ts_file}.png'))
#     plt.close(cost_fig)
#     del(data)
#     del(data_cut)
#     del(chisqs)
#     break

# #make plot with summed cost surface
# fig, ax = plt.subplots()
# im = ax.imshow(
#     chisq_tot,
#     origin='lower',
#     aspect='auto',
#     cmap='viridis',
#     extent=[lats[0], lats[-1], lons[0], lons[-1]]  
# )
# plt.rcParams.update({
#             "font.size": 16,
#             "axes.labelsize": 18,
#             "axes.titlesize": 20,
#             "xtick.labelsize": 14,
#             "ytick.labelsize": 14,
#             "figure.titlesize": 22,
#             "legend.fontsize": 14
#         })
# ax.set_xlabel('Lat Offset ()')
# ax.set_ylabel('Lon Offset ()')
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(r'$\chi^2$')
# ax.plot(
#     0, 0,
#     marker='x',
#     color='red',
#     markersize=10,
#     markeredgewidth=2)
# plt.tight_layout()
# plt.savefig(os.path.join(fitting_path, f'all_coordfit_plot2.png'))