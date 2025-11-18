import os
import sys
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np
import cupy as cp
import helper as hp_f
import figures as fgs
import json
import argparse
from src.utils import orbcomm_utils as outils
from src.utils import orbcomm_utils_gpu as outils_g
from src.utils import baseband_utils as butils
from src.correlations import baseband_data_classes as bdc
from scripts.orbcomm import sat_utils as su
from scripts.orbcomm import sat_utils_gpu as sug
from scripts.xcorr import helper as hp_x
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Config file containing all required data.",)
    parser.add_argument(
        "-o", "--output_path", type=str, default="/scratch/thomasb", help="Output directory for debug and pulses")
    args = parser.parse_args()

    out_path = args.output_path
    T_SPECTRA = 4096/250e6
    v_acclen = 1000  #accumulation length for visibility sanity-checks
    bline_ants = ['Antenna 1', 'Antenna 6']
    file_save_names = ['MARS1', 'MARS6']
    sat = 57166
    chan_big_idx = 1837
    pulse_rel_start_t = 16545
    pulse_rel_end_t = 16810
    buffer_end = 100

    #EXTRACT INFO FROM CONFIG FILE-------------------------------------
    dir_parents, coords, ant_names, clock_offsets = [], [], [], []
    with open(args.config_file, "r") as f:
        config = json.load(f)
        for i, (ant, details) in enumerate(config["antennas"].items()):
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
            ant_names.append(details["name"])
            clock_offsets.append(details['clock_offset'])
        global_start_t = config["correlation"]["start_timestamp"]
        global_end_t = config["correlation"]["end_timestamp"]

    #SET UP VARIABLES-------------------------------------
    t1 = pulse_rel_start_t + global_start_t 
    t2 = pulse_rel_end_t + global_start_t - buffer_end

    tle_path = outils.get_tle_file(t1, "/project/rrg-sievers/mohanagr/OCOMM_TLES")

    ant1_idx, ant2_idx = ant_names.index(bline_ants[0]), ant_names.index(bline_ants[1])
    ant1_path, ant2_path = dir_parents[ant1_idx], dir_parents[ant2_idx]
    ant1_coords, ant2_coords  = coords[ant1_idx], coords[ant2_idx]
    ant1_offset, ant2_offset = clock_offsets[ant1_idx], clock_offsets[ant2_idx]
    spec_offset = ant2_offset - ant1_offset

    print('ant indices:', ant1_idx, ant2_idx)
    print('ant paths:', ant1_path, ant2_path)
    print('ant coords:', ant1_coords, ant2_coords)
    print('clock offsets (wrt MARS1):', ant1_offset, ant2_offset)
    print('relative clock offset (ant2 - ant1):', spec_offset)

    #GET BLOCKS-------------------------------------
    blk_nspec = 3*10**6
    spectrum_indices = np.arange(blk_nspec)
    T_BLOCK = blk_nspec * T_SPECTRA
    nblks = int(np.floor((t2-t1)/T_BLOCK))
    print('BLOCK PERIOD IN SECONDS:', T_BLOCK)
    print('BLOCK COUNT IN PULSE:', nblks)

    #GET FILES AND IDXS------------------------------------
    ant1_files, ant1_idx, ant2_files, ant2_idx = hp_x.get_init_info_2ant(t1, 
                                                                         t2, 
                                                                         spec_offset, 
                                                                         ant1_path, 
                                                                         ant2_path)
    
    channels = np.asarray(bdc.get_header(ant1_files[0])["channels"],dtype='int64')
    chanstart = np.where(channels == 1834)[0][0]
    chanend = np.where(channels == 1852)[0][0]
    nchans = chanend - chanstart
    chanlist = np.arange(1834, 1852)
    chan_s_idx = np.where(chanlist == chan_big_idx)[0]
    print('chanstart, chanend:', chanstart, chanend)
    print('small channel index', chan_s_idx)


    #GET DELAYS-----------------------------
    #start by getting delays for the whole pulse
    #want continuous delays spanning all blocks exactly
    niter = t2-t1 + 1  #+1 to avoid edge effects
    freq = 250e6 * (1 - chan_big_idx / 4096)
    delays = np.zeros(nblks*blk_nspec)
    d = outils.get_sat_delay(ant1_coords,
                             ant2_coords,
                             tle_path,
                             t1,
                             niter,
                             sat)
    delays_all = np.interp(np.arange(0, nblks*blk_nspec) * T_SPECTRA, 
                                np.arange(0, niter), 
                                d)
    delays_all = cp.asarray(delays_all)


    #GET DATA----------------------------------------
    ant1_blks, ant2_blks = [], []

    ant1 = bdc.BasebandFileIterator(ant1_files,
                                    0,
                                    ant1_idx,
                                    blk_nspec,
                                    nchunks=nblks,
                                    chanstart=chanstart,
                                    chanend=chanend,
                                    type = 'float')

    ant2 = bdc.BasebandFileIterator(ant2_files,
                                    0,
                                    ant2_idx,
                                    blk_nspec,
                                    nchunks=nblks,
                                    chanstart=chanstart,
                                    chanend=chanend,
                                    type='float')

    for i, (chunk1,chunk2) in enumerate(zip(ant1,ant2)):
        ant1_data = cp.asarray(chunk1['pol0'][:, chan_s_idx].copy().ravel())
        ant2_data = cp.asarray(chunk2['pol0'][:, chan_s_idx].copy().ravel())
        assert len(ant1_data) == blk_nspec
        assert len(ant2_data) == blk_nspec
        ant1_blks.append(ant1_data)
        ant2_blks.append(ant2_data)

    #PICK ONE BLOCK FOR NOW------------------
    k = 2
    ant1_blk, ant2_blk = ant1_blks[k], ant2_blks[k]
    delays_blk = delays_all[k*blk_nspec:(k+1)*blk_nspec]
    pulse_output = os.path.join(out_path, f'finetiming_{pulse_rel_start_t}_nspec{int(blk_nspec/10e6)}M_acclen{int(v_acclen/1000)}_kval{k}')
    os.makedirs(pulse_output, exist_ok=True)

    print('ant1_blk shape', ant1_blk.shape)
    print('ant2_blk shape', ant2_blk.shape)

    #APPLY GEO DELAY TO BLOCK
    phase = cp.exp(-2j * cp.pi * freq * delays_blk)
    ant2_block_delayed = ant2_blk * phase
    
    #GET ALPHA 1
    xc = ant1_blk*cp.conj(ant2_block_delayed)
    xc = xc.get()
    xc_fft = np.fft.fftshift(np.abs(np.fft.fft(xc)))
    fft_len = len(xc_fft)
    fft_argmax = np.argmax(xc_fft)
    alpha1 = -(fft_argmax-fft_len/2)/fft_len /chan_big_idx
    fft_fig = fgs.make_alpha_approximator_plot(xc_fft)
    fft_fig.savefig(os.path.join(pulse_output, 'xc_fft.jpg'))

    print('XC SHAPE:', xc.shape)
    print('XC_FFT SHAPE:', xc_fft.shape)
    print('FFT ARGMAX:', fft_argmax)
    print('ALPHA 1:', alpha1)

    #MAKE SOME VISIBILITIES
    blk_vis = hp_f.average_rows(xc, nblock = v_acclen).ravel()
    blk_angle = np.angle(blk_vis)
    blk_phase = np.unwrap(blk_angle) - blk_angle[0]

    #FIT FOR NEW ALPHA
    alpha2 = hp_f.lmsolver(xc,alpha1,chan_big_idx)
    xc_new = xc*np.exp(1j*2*np.pi*chan_big_idx*spectrum_indices*alpha2)
    print('alpha2', alpha2)

    #PLOT VIS VS LINEAR
    vis_idxs = np.arange(0, blk_nspec, v_acclen)
    alpha1_linear = -2*np.pi*chan_big_idx*vis_idxs*alpha1
    alpha2_linear = -2*np.pi*chan_big_idx*vis_idxs*alpha2
    visfig = fgs.plot_vis_alpha(blk_phase, alpha1_linear, alpha2_linear, v_acclen = v_acclen)
    visfig.savefig(os.path.join(pulse_output, 'phase_block.jpg'))

    #PLOT OF AMPLITUDES
    trialfig = fgs.plot_around_guess(xc, 
                                     alpha1, 
                                     alpha2, 
                                     chan_big_idx, 
                                     50, 
                                     1*10**(-11))
    trialfig.savefig(os.path.join(pulse_output, 'alpha_trials.jpg'))
    
    #COMPARE AMPLITUDES
    xc_amp_1 = np.abs(np.mean(xc))**2
    xc_amp_2 = np.abs(np.mean(xc_new))**2
    print('old amp', xc_amp_1)
    print('new_amp', xc_amp_2)


    #MAKE VISIBILITIES FOR WHOLE PULSE
    ant1_tot = cp.empty(blk_nspec*nblks, dtype=cp.complex64)
    ant2_tot = cp.empty(blk_nspec*nblks, dtype=cp.complex64)

    for k in range(len(ant1_blks)):
        ant1_tot[k*blk_nspec:(k+1)*blk_nspec] = ant1_blks[k]
        ant2_tot[k*blk_nspec:(k+1)*blk_nspec] = ant2_blks[k]

    phase_tot = cp.exp(-2j * cp.pi * freq * delays_all)
    ant2_tot_delayed = ant2_tot * phase_tot
    tot_xc = ant1_tot * cp.conj(ant2_tot_delayed)
    tot_vis = hp_f.average_rows(tot_xc, nblock = v_acclen).ravel()
    tot_angle = np.angle(tot_vis)
    tot_phase = np.unwrap(tot_angle) - tot_angle[0]
    tot_phase = tot_phase.get()
    visfig_tot = fgs.plot_vis_all(tot_phase, v_acclen = v_acclen)
    visfig_tot.savefig(os.path.join(pulse_output, 'full vis.jpg'))



