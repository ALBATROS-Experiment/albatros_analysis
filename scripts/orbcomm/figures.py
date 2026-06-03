import os
import sys
from sys import path
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np 
from matplotlib import pyplot as plt
from src.utils import orbcomm_utils as outils
from scripts.orbcomm import sat_utils as su
import cupy as cp

def make_ampfig(vis, v_acclen=10000, T_SPECTRA=4096/250e6):
    vis_amp = np.abs(vis)
    chan_width = 1/T_SPECTRA
    chunk_length = v_acclen*T_SPECTRA
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.rcParams.update({
                "font.size": 16,
                "axes.labelsize": 16,
                "axes.titlesize": 20,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "figure.titlesize": 22
            })

    im = ax.imshow(vis_amp.T, aspect='auto', cmap='plasma', interpolation='none')
    ax.set_xlabel(f"Channel Index (~{int(chan_width)} Hz)")
    ax.set_ylabel(f"Visibility Chunk (~{np.round(chunk_length, decimals=2)} s)")
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label("Amplitude", fontsize = 12)

    return fig



def make_cxcorr_plot(data):
    data_cpu = cp.asnumpy(data)
    fig,ax=plt.subplots(6,3)
    fig.set_size_inches(10,12)
    ax=ax.flatten()
    for chan in range(18):
        data_chan = np.abs(data_cpu[chan,:])
        peak_idx=np.argmax(data_chan)
        ax[chan].set_title(f"{peak_idx}")
        ax[chan].plot(data_chan)
    plt.tight_layout()
    return fig

def zoomed_cxcorr_plot(data, chan_small_idx, N2 = 200, x = 'spectra', T_SPECTRA = 4096/250e6):
    assert (x == 'spectra') or (x == 'secs')
    data_cpu = cp.asnumpy(data)
    fig, ax=plt.subplots(1,2)
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22
        })
    fig.set_size_inches(10,5)
    ax=ax.flatten()
    data_chan = np.abs(data_cpu[chan_small_idx,:])
    nspec1 = len(data_chan)/2
    x1 = np.arange(-nspec1, nspec1)
    x2 = np.arange(-N2, N2)
    peak_idx = np.argmax(data_chan)
    offset = peak_idx-100000
    if x == 'secs':
        x1 = x1 * T_SPECTRA
        x2 = x2 * T_SPECTRA * 1000
        offset = np.round(offset * T_SPECTRA, decimals=3)
    peak_amp = data_chan[peak_idx]
    noise_amp = su.median_abs_deviation(data_chan)
    snr = peak_amp/noise_amp

    #left plot (no zoom)
    ax[0].plot(x1, data_chan)
    ax[0].set_title(f'Full CXCORR')
    if x == 'secs':
        ax[0].set_xlabel(r"Offset (s)")
    else:
        ax[0].set_xlabel(r"Spectrum Offset ($\sim16$ $\mu$s)")
    ax[0].set_ylabel("Amplitude")
    interval1 = 0.5
    ticks1 = np.arange(-3*interval1, 4*interval1, interval1)
    ax[0].set_xticks(ticks1)
    ax[0].grid(True)

    #data information
    stats_text = f"SNR: {snr:.0f}\nMAD: {noise_amp:.4f}\nOffset: {offset} s"
    box_xloc = 0.65
    if offset>0:
        box_xloc = 0.05
    ax[0].text(
        box_xloc, 0.95, stats_text,
        transform=ax[0].transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    #right plot (with zoom)
    data_chan_zoomed = data_chan[peak_idx - N2: peak_idx + N2]
    ax[1].plot(x2, data_chan_zoomed)
    ax[1].set_title('Zoomed on Peak')
    ax[1].grid(True)
    interval2 = 1
    ticks2 = np.arange(-3*interval2, 4*interval2, interval2)
    ax[1].set_xticks(ticks2)
    if x == 'secs':
        ax[1].set_xlabel(r"Offset (ms)")
    else:
        ax[1].set_xlabel(r"Spectrum Offset ($\sim16$ $\mu$s)")
    plt.tight_layout()

    return fig



def make_snr_plot(data, temp_satmap):
    snrfig, snrax = plt.subplots()
    for i in range(len(temp_satmap)):
        snrax.plot(data[i, :], label=f"{temp_satmap[i]}")
    snrax.set_xlabel("Channels")
    snrax.set_ylabel("SNR")
    snrax.legend()
    return snrfig

def makeplot_fringes_phase(coords, 
                           times,
                           chan_big_idx, 
                           chanlist, 
                           vis_angle, 
                           phase, 
                           satmap,
                           sats_present,
                           v_acclen,
                           T_SPECTRA = 4096/250e6):
    ''' 
    make plot of angle fringes and phase, side by side
    '''
    chan_small_idx = np.where(chanlist == chan_big_idx)[0][0]
    chunk_length = v_acclen*T_SPECTRA
    plt.rcParams.update({
                "font.size": 16,
                "axes.labelsize": 16,
                "axes.titlesize": 20,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "figure.titlesize": 22
            })
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.4)
    #fig.suptitle(f'Pulse {times[0]} Channel {chan_big_idx}/{chan_small_idx}')

    im = ax[0].imshow(vis_angle, aspect='auto', cmap='RdBu', interpolation='none')
    ax[0].set_title('Wrapped Phases')
    ax[0].set_xlabel("Channel Index (~60 kHz)")
    ax[0].set_ylabel(f"Visibility Chunk (~{np.round(chunk_length, decimals=2)} s)")
    cbar = fig.colorbar(im, ax=ax[0], orientation='vertical')
    cbar.set_label("Wrapped Phase (radians)", fontsize = 12)

    ax[1].plot(phase, linewidth = 5, alpha=0.5, label='Detected phase')
    ax[1].set_title(f'Unwrapped Phase (Chan. {chan_small_idx})')
    ax[1].set_xlabel(f"Visibility Chunk (~{np.round(chunk_length, decimals=2)} s)")
    ax[1].set_ylabel("Unwrapped Phase (radians)")
    for sat in sats_present:
        satID = satmap[sat]
        print("getting prediction for", satID)
        pred_phase = outils.pred(coords[0], coords[1], times[0], times[1], chan_big_idx, int(satID), v_acclen=v_acclen)[:len(phase)]
        #print('MAX PHASE DIFFERENCE:', np.max(np.diff(np.abs(pred_phase))))
        ax[1].plot(pred_phase, linestyle='--', color = 'red', linewidth=1, label=f'Satellite {satID}')
    ax[1].legend()

    return fig

def makeplot_fringes_phase_upsampled(data_angle, 
                                     chan_new, 
                                     start_time, 
                                     end_time,
                                     coords1,
                                     coords2, 
                                     acclen=512, 
                                     osamp=64, 
                                     satID = 57166):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    #fig.suptitle(f'METEOR M2-3 on {name1}-{name2} Channel {chan_old}')
    chan_old = 1834 + chan_new/osamp
    chunk_len = (acclen*4096*osamp)/250e6
    print('old_channel:', chan_old)

    im = ax[0].imshow(data_angle, aspect='auto', cmap='RdBu', interpolation='none')
    ax[0].set_xlabel(f"Channel idx (~{int(250e6/(4096*osamp))} Hz)")
    ax[0].set_ylabel(f"Visibility Chunk (~{np.round(chunk_len, decimals=2)} s)")
    cbar = fig.colorbar(im, ax=ax[0], orientation='vertical')
    cbar.set_label("Phase (radians)")

    phase = np.unwrap(data_angle[:, chan_new] - data_angle[0, chan_new])
    pred_phase = outils.pred(coords1, 
                             coords2, 
                             start_time, 
                             end_time, 
                             chan_old, 
                             int(satID), 
                             v_acclen=acclen, 
                             T_SPECTRA = (4096*osamp)/250e6)[:len(phase)]

    ax[1].plot(phase, label='Detected')
    ax[1].plot(pred_phase, label=f'Predicted {satID}')
    ax[1].set_xlabel(f"Visibility Chunk (~{np.round(chunk_len, decimals=2)} s)")
    ax[1].set_ylabel("Phase (radians)")

    print('MAX PHASE DIFFERENCE:', np.max(np.diff(np.abs(pred_phase))))
    ax[1].legend()
    plt.tight_layout
    return fig


def makeplot_fringes_phase2(coords1, 
                            coords2,
                            t1, 
                            t2,
                            chan_big_idx, 
                            p_vis1,
                            p_vis2,
                            phase1,
                            phase2,
                            satID,
                            v_acclen,
                            T_SPECTRA = 4096/250e6,
                            suptitle = 'Fringes and Phases'):

    print(p_vis1.shape)
    chunk_length = v_acclen * T_SPECTRA
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
    fig.suptitle(suptitle)
    
    im1 = ax[0,0].imshow(p_vis1, aspect='auto', cmap='RdBu', interpolation='none')
    ax[0,0].set_xlabel("channel idx (~60 kHz interval)")
    ax[0,0].set_ylabel("chunk number (~0.5 s interval)")
    cbar = fig.colorbar(im1, ax=ax[0,0], orientation='vertical')
    cbar.set_label("phase (radians)")
    
    ax[0,1].plot(phase1, label='detected phase')
    ax[0,1].set_ylabel("phase (radians)")
    print("getting prediction for", satID)
    pred_phase1 = outils.pred(coords1[0], 
                              coords1[1], 
                              t1, 
                              t2, 
                              chan_big_idx, 
                              int(satID), 
                              v_acclen=v_acclen)[:len(phase1)]
    ax[0,1].plot(pred_phase1, label=f'sat {satID}')
    ax[0,1].legend()

    im2 = ax[1,0].imshow(p_vis2, aspect='auto', cmap='RdBu', interpolation='none')
    ax[1,0].set_xlabel("channel idx (~60 kHz interval)")
    ax[1,0].set_ylabel("chunk number (~0.5 s interval)")
    cbar = fig.colorbar(im2, ax=ax[1,0], orientation='vertical')
    cbar.set_label("phase (radians)")

    ax[1,1].plot(phase2, label='detected phase')
    ax[1,1].set_xlabel(f"chunk number (~{np.round(chunk_length, decimals=2)} s interval)")
    ax[1,1].set_ylabel("phase (radians)")
    print("getting prediction for", satID)
    pred_phase2 = outils.pred(coords2[0], 
                              coords2[1], 
                              t1, 
                              t2, 
                              chan_big_idx, 
                              int(satID), 
                              v_acclen=v_acclen)[:len(phase2)]
    ax[1,1].plot(pred_phase2, label=f'sat {satID}')
    ax[1,1].legend()

    return fig

    
    
def makeplot_cxcorr_phase(cxcorr, 
                          coords, 
                          global_start_t, 
                          rel_start_t, 
                          chan_idx_small, 
                          satID, 
                          phase, 
                          T_SPECTRA=4096/250e6, 
                          c_acclen=10**6, 
                          v_acclen=5000, 
                          dN=10**5):
    """
    Plots overall cross-correlation and unwrapped phase

    Parameters
    ----------
    cxcorr : array
        coarse xcorr data (in complex form)
    coords : list of lists
        [[lat, lon, alt], [lat, lon, alt]] of both antenna (lower index antenna first)
    global_start_t : int
        unix starting timestamp of pulsedata file
    rel_start_t : int
        unix starting timestamp of pulse after global_start_t
    chan_idx_small : int
        index of channel in chanlist
    satID : int
        ID of satellite in the pulse
    phase : array
        measured unwrapped phase data

    T_SPECTRA : float, optional
        time for a single spectrum
    c_acclen : int, optional
        coarse cross correlation accumulation length
    v_acclen : int, optional
        visibility computation accumulation length
    dN : int, optional
        window of spectrum offsets we plot around the center

    Returns
    -------
    fig : figure
        final figure
    """

    #cxcorr parameter setup
    center = c_acclen
    chunk_length = c_acclen * T_SPECTRA
    spectra = np.arange(-dN, dN)
    pulse_start_t = rel_start_t + global_start_t

    #phase parameter setup
    chanlist = np.arange(1834, 1852)
    chan_idx_big = chanlist[chan_idx_small]
    #check for bug in frequency it prints!!
    chan_mhz = np.round(outils.chan2freq(chan_idx_big)/(10**6), decimals = 2)

    #times setup
    time_in_secs = np.round(np.arange(len(phase)) * chunk_length).astype(int)
    t_tick_spacing = 20 
    t_tick_vals = np.arange(0, time_in_secs[-1] + t_tick_spacing, t_tick_spacing)
    t_tick_idxs = np.searchsorted(time_in_secs, t_tick_vals)
    t_tick_idxs = t_tick_idxs[t_tick_idxs < len(time_in_secs)]
    T_spec_ms = int(np.round(T_SPECTRA * 10 **6))

    #phase stuff
    ampdata = np.abs(cxcorr[chan_idx_small,center-dN:center+dN])
    peak_idx = np.argmax(ampdata)
    pred = outils.pred(coords[0], coords[1], pulse_start_t, pulse_start_t+1000, chan_idx_big, satID, v_acclen=v_acclen)[:len(phase)]
    

    #plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22
        })

    #left
    fig.suptitle(f"Sat {satID} at {chan_mhz}MHz")
    ax[0].plot(spectra, ampdata, label=f"Peak at {peak_idx - dN}")
    ax[0].set_xticklabels([])
    ax[0].set_title(f"Coarse x-corr")
    ax[0].legend(loc='upper right', fontsize=12)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel(f"Spectrum Shift ({T_spec_ms}" + r'$\mu$s units)')

    #right
    ax[1].set_ylabel("Phase (rads)")
    ax[1].set_xlabel(f"Time (seconds)")
    ax[1].set_title(f"Unwrapped Phase")
    ax[1].plot(pred, label = 'Prediction', color='orange')
    ax[1].plot(phase, label='Measurement', linestyle='--', c='blue')
    ax[1].legend(fontsize=12)
    ax[1].set_xticks(t_tick_idxs)
    ax[1].set_xticklabels([])
    #ax[0,1].annotate(f"MARS1-\nMARS7\n{bl1_dist}m", xy=(1.05, 0.5), xycoords='axes fraction', rotation=0, va='center', ha='left', fontsize=15)

    plt.tight_layout()
    return fig


def makeplot_cxcorr_phase2(cxcorr1, 
                           cxcorr2, 
                           coords1, 
                           coords2, 
                           global_start_t, 
                           rel_start_t, 
                           chan_idx_small, 
                           sat_ID, 
                           phase1, 
                           phase2, 
                           T_SPECTRA=4096/250e6, 
                           c_acclen=10**6, 
                           v_acclen=5000):
    ''' 
    Overall cxcorr and phase plot for two pulses. Can be different baselines or same baseline.
    '''
    
    #parameter setup
    center = c_acclen
    dN = 10**5
    chunk_length = c_acclen * T_SPECTRA
    spectra = np.arange(-dN, dN)
    pulse_start_t = rel_start_t + global_start_t

    chanlist = np.arange(1834, 1852)
    chan_idx_big = chanlist[chan_idx_small]
    chan_mhz = np.round(outils.chan2freq(chan_idx_big)/(10**6), decimals = 2)

    #times setup
    time_in_secs = np.round(np.arange(len(phase1)) * chunk_length).astype(int)
    t_tick_spacing = 20 
    t_tick_vals = np.arange(0, time_in_secs[-1] + t_tick_spacing, t_tick_spacing)
    t_tick_idxs = np.searchsorted(time_in_secs, t_tick_vals)
    t_tick_idxs = t_tick_idxs[t_tick_idxs < len(time_in_secs)]
    T_spec_ms = int(np.round(T_SPECTRA * 10 **6))

    #data setup
    ampdata_1 = np.abs(cxcorr1[chan_idx_small,center-dN:center+dN])
    peak_1 = np.argmax(ampdata_1)
    ampdata_2 = np.abs(cxcorr2[chan_idx_small,center-dN:center+dN])
    peak_2 = np.argmax(ampdata_2)

    pred1 = outils.pred(coords1[0], coords1[1], pulse_start_t, pulse_start_t+1000, chan_idx_big, sat_ID, v_acclen=v_acclen)[:len(phase1)]
    pred2 = outils.pred(coords2[0], coords2[1], pulse_start_t, pulse_start_t+1000, chan_idx_big, sat_ID, v_acclen=v_acclen)[:len(phase2)]

    #plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22
        })

    #top left
    fig.suptitle(f"METEOR M2-3 at {chan_mhz}MHz")
    ax[0,0].plot(spectra, ampdata_1, label=f"Peak at {peak_1 - dN}")
    ax[0,0].set_xticklabels([])
    ax[0,0].set_title(f"Coarse x-corr")
    ax[0,0].legend(loc='upper right', fontsize=12)
    ax[0,0].set_ylabel("Amplitude")

    #top right
    ax[0,1].set_ylabel("Phase (rads)")
    ax[0,1].set_title(f"Unwrapped Phase")
    ax[0,1].plot(pred1, label = 'Prediction', color='orange')
    ax[0,1].plot(phase1, label='Measurement', linestyle='--', c='blue')
    ax[0,1].legend(fontsize=12)
    ax[0,1].set_xticks(t_tick_idxs)
    ax[0,1].set_xticklabels([])
    #ax[0,1].annotate(f"MARS1-\nMARS7\n{bl1_dist}m", xy=(1.05, 0.5), xycoords='axes fraction', rotation=0, va='center', ha='left', fontsize=15)

    #bottom left
    ax[1,0].plot(spectra, ampdata_2, label=f"Peak at {peak_2 - dN}")
    ax[1,0].tick_params(axis='x', labelsize=12)
    ax[1,0].set_xlabel(f"Spectrum Shift ({T_spec_ms}" + r'$\mu$s units)')
    ax[1,0].legend(loc='upper right', fontsize=12)
    ax[1,0].set_ylabel("Amplitude")

    #bottom right
    ax[1,1].set_xlabel(f"Time (seconds)")
    ax[1,1].set_ylabel("Phase (rads)")
    ax[1,1].set_xticks(t_tick_idxs)
    ax[1,1].set_xticklabels([time_in_secs[i] for i in t_tick_idxs])
    ax[1,1].plot(pred2, label='Prediction', color='orange')
    ax[1,1].plot(phase2, label='Measurement', linestyle = '--', color='blue')
    ax[1,1].legend(fontsize=12)
    #ax[1, 1].annotate(f"MARS1-\nMARS4\n{bl2_dist}m", xy=(1.05, 0.5), xycoords='axes fraction', rotation=0, va='center', ha='left', fontsize=15)

    plt.tight_layout()
    return fig


def snr_vs_coords(snr_arr,
                  latitudes,
                  longitudes,
                  title = None,
                  marker_coords=None,  
                  marker_labels=None,   
                  out_path=None):
    ''' 
    marker coords: list of (lat, lon) or array shape (N, 2)
    marker labels: optional labels for each marker
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({"font.size": 16,
                         "axes.labelsize": 18,
                         "axes.titlesize": 20,
                         "xtick.labelsize": 14,
                         "ytick.labelsize": 14,
                         "figure.titlesize": 22})

    im = ax.imshow(snr_arr,
                   extent=[longitudes[0], longitudes[-1], latitudes[0], latitudes[-1]],
                   origin='lower',
                   aspect='auto',
                   cmap='viridis')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('SNR')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    if title:
        ax.set_title(title)

    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    ax.set_xticks(np.round(np.linspace(longitudes[0], longitudes[-1], 7), 4))
    ax.set_yticks(np.round(np.linspace(latitudes[0], latitudes[-1], 7), 4))
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    ax.grid(color='white', linestyle='--', alpha=0.4)

    if marker_coords is not None:
        marker_coords = np.array(marker_coords)
        ax.scatter(marker_coords[:, 1],  # longitude (x)
                   marker_coords[:, 0],  # latitude (y)
                   color='red',
                   marker='x',
                   s=80,
                   label='Marker' if marker_labels is None else None)
        if marker_labels is not None:
            for (lat, lon), label in zip(marker_coords, marker_labels):
                ax.text(lon, lat, f" {label}", color='red', fontsize=9, va='center')

        if marker_labels is None:
            ax.legend(loc='upper right')

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_phase_residuals(phase,
                         coords,
                         times,
                         channel,
                         satID,
                         T_SPECTRA = 4096/250e6,
                         v_acclen = 10000):

    chunk_length = v_acclen *T_SPECTRA
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({"font.size": 16,
                         "axes.labelsize": 18,
                         "axes.titlesize": 20,
                         "xtick.labelsize": 14,
                         "ytick.labelsize": 14,
                         "figure.titlesize": 22})
    
    pred = outils.pred(coords[0], 
                       coords[1], 
                       times[0], 
                       times[1], 
                       channel, 
                       satID, 
                       T_SPECTRA=T_SPECTRA, 
                       v_acclen=v_acclen)
    
    ax.set_xlabel(f'Visibility Chunks (~{np.round(chunk_length, decimals=2)} s)')
    ax.set_ylabel('Phase Residual (radians)')
                       
    assert len(phase) == len(pred)
    
    plt.plot(phase - pred)

    return fig

