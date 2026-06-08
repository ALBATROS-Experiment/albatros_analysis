import os
import sys
import time
import gc
from os import path

sys.path.insert(0, "/home/thomasb/")

from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_g
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.scripts.orbcomm import sat_utils_gpu as sug

import cupy as cp
import numpy as np
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.scripts.xcorr import helper as hxc
from albatros_analysis.scripts.xcorr import helper_gpu as hxc_g


import argparse
import json
from matplotlib import pyplot as plt
import sat_utils as su
import sat_utils_gpu as sug
import figures as fgs
import importlib

#batch_start_ts = 1753200150
batch_start_ts = 1762037710
batch_end_ts = 1762037710 + 30000
dN = 100000
v_acclen = 10000
c_acclen = 3000000
antnum_ref = 1
antnum = 4
pnum = 0
indent = 0 #49.152*2
DO_CX = True
DO_VIS = False
DO_PHASE = True
RUSSIANS_ONLY = True

T_SPECTRA = 4096/250e6
T_SCAN = 5 #seconds between each satellite risen scan -- look for sat rise/set every 5 sec.
altitude_cutoff = 10  #cutoff when looking for satellites
satlist = [28654,25338,33591,57166,59051,44387]

path_debug = f'/scratch/thomasb/satdet_debugging/{batch_start_ts}'
os.makedirs(path_debug, exist_ok = True)

#with open(f'config/nov25_batch_{batch_start_ts}_refALB2.json', "r") as f:
#with open(f'config/config_batch2.json', "r") as f:
#with open(f'config/config_batch2_testing.json', "r") as f:
    
    # config = json.load(f)
    # dir_parents, coords, ant_names = [], [], []

    # print("\nAntenna Details:")
    # for i, (ant, details) in enumerate(config["antennas"].items()):
    #     #print(ant, details)
    #     coords.append(details['coordinates'])
    #     dir_parents.append(details["path"])
    #     ant_names.append(details["name"])
    # batch_start_ts = config["correlation"]["start_timestamp"]
    # batch_end_ts = config["correlation"]["end_timestamp"]
    # c_acclen = config["correlation"]["coarse_acclen"]


ant_names = np.array(['Antenna 1', 'Antenna 2', 'Antenna 3', 'Antenna 4','Antenna 5','Antenna 6','Antenna 7','Antenna 8'])
ant_paths = np.array([
    '/project/rrg-sievers/albatros/mars/202507/mars1/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars2/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars3/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars4/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars5',
    '/project/rrg-sievers/albatros/mars/202507/mars6/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars7/baseband',
    '/project/rrg-sievers/albatros/mars/202507/mars8/baseband'
])

ant_coords = [
            [79.41716147,	-90.76723869,	187.9577],	#MARS1
            [79.41719805,	-90.75873919,	183.0684],	#MARS2		
            [79.41540905,	-90.77299123,	180.4816],	#MARS3			
            [79.38845641,	-91.01920296,	25.1938],	#MARS4
            [79.41830257,	-90.66739545,	59.6242],	#MARS5			
            [79.39798424,	-90.79984241,	41.699],	#MARS6			
            [79.41147412,	-90.69526613,	31.6314],	#MARS7				
            [79.44375769,	-90.71820263,	414.9131]]	#MARS8


print('batch start time', batch_start_ts)
print("\nAntenna Coordinates:", ant_coords)
print("Coarse acclen", c_acclen)

array_time =  batch_end_ts - batch_start_ts
coords_ref, path_ref = ant_coords[antnum_ref], ant_paths[antnum_ref]  #(ref = Reference Ant, nref = Non-Reference Ant)
tle_path = outils.get_tle_file(batch_start_ts, "/project/rrg-sievers/mohanagr/OCOMM_TLES")
overflow_files_ref = hxc.get_overflow_files(batch_start_ts, batch_end_ts, ant_paths[antnum_ref])

satmap = {} #maps sat IDs (e.g. 33591) to its index in satlist (e.g. 2), without collisions
assert min(satlist) > len(satlist)
for i, sat_ID in enumerate(satlist):
    satmap[i] = sat_ID
    satmap[sat_ID] = i
print('Satmap', satmap)


#====================================================================================================
#RISEN SATS AND PASSES

nrows = int((array_time)/T_SCAN)
arr = np.zeros((nrows, len(satlist)), dtype="int64")
rsats = outils.get_risen_sats(tle_path, coords_ref, batch_start_ts, dt=T_SCAN, niter=nrows, good=satlist, altitude_cutoff=altitude_cutoff)
num_sats_risen = [len(x) for x in rsats]
for i, row in enumerate(rsats):
    for sat_ID, satele, sataz in row:
        arr[i,satmap[sat_ID]] = 1

passes = outils.get_simul_pulses(arr) #beware the function is named pulses
npasses = len(passes)
print("Number of Passes:", npasses, '\n')


#====================================================================================================
#PICK THE ANTENNA
print(f"{ant_names[antnum]}")
path_ant = os.path.join(path_debug, f'{ant_names[antnum]}')
os.makedirs(path_ant, exist_ok=True)

path_nref, coords_nref = ant_paths[antnum], ant_coords[antnum]
overflow_files_nref = hxc.get_overflow_files(batch_start_ts, batch_end_ts, ant_paths[antnum])

print('path ref', path_ref)
print('path non-ref', path_nref)
print('\n')
print('coords ref', coords_ref)
print('coords non-ref', coords_nref)

#====================================================================================================
#PICK THE PULSE
print(f'=============STARTING PNUM {pnum}============')
(pstart, pend), sats_present = passes[pnum]
pstart, pend = pstart*T_SCAN, pend*T_SCAN  #go from T_SCAN indices to times in s

#split everything into chunks
chunk_length_secs = c_acclen * T_SPECTRA
chunk_times = np.arange(pstart, pend, chunk_length_secs)
nchunks = len(chunk_times)-1

#make temporary satmap for the pulse
temp_satmap = [] 
temp_satmap.append("Uncorrected")
russian_present = False
for i, satidx in enumerate(sats_present):
    s = satmap[satidx]
    if s in {59051, 57166}:
        russian_present = True
    temp_satmap.append(s)

# if RUSSIANS_ONLY:
#     if not russian_present:
#         continue
    
path_pulse = os.path.join(path_ant, f'pulse{pnum}')
os.makedirs(path_pulse, exist_ok=True) 


#get things in unix time
#pass_start_unix, pass_end_unix = batch_start_ts + pstart, batch_start_ts + pend
pass_start_unix = batch_start_ts + pstart + indent
pass_end_unix = pass_start_unix + 800
tle_path = outils.get_tle_file(pass_start_unix, "/project/rrg-sievers/mohanagr/OCOMM_TLES") #use most up-to-date tle file

print('TLE path:', tle_path)
print('pstart', pstart)
print('pend', pend)
print("pass length:", pend-pstart)
print('unix pstart:', pass_start_unix)
print('chunk times', chunk_times)


if DO_CX:
    #====================================================================================================
    #ANTENNA OBJECT STUFF
    try:
        files_ref, idx_ref = butils.get_init_info(pass_start_unix, pass_end_unix, path_ref)
        files_nref, idx_nref = butils.get_init_info(pass_start_unix, pass_end_unix, path_nref)
    except Exception as e:
        print(e)
        print(f"WARNING: skipping pass {pstart} to {pend}. MISTAKE IN FILE CHECKER!!")
        sys.exit()

    #print('files ref"\n', files_ref)
    #print('files nref"\n', files_nref)

    print('idxs ref ant:', idx_ref)
    print('idxs nonref ant:', idx_nref)

    # Set up the number of channels we look through
    print("Setting Antenna as BFI Objects", '\n')
    channels = np.asarray(bdc.get_header(files_ref[0])["channels"],dtype='int64')
    chanstart = np.where(channels == 1834)[0][0]
    chanend = np.where(channels == 1852)[0][0]
    nchans = chanend - chanstart

    nchans = chanend - chanstart
    ref = bdc.BasebandFileIterator(
        files_ref,
        0,
        idx_ref,
        c_acclen,
        nchunks,
        chanstart=chanstart,
        chanend=chanend,
        type="float",
    )
    nref = bdc.BasebandFileIterator(
        files_nref,
        0,
        idx_nref,
        c_acclen,
        nchunks,
        chanstart=chanstart,
        chanend=chanend,
        type="float",
    )

    print('ref: acclen', ref.acclen, 'nchunks', ref.nchunks)
    print('nref: acclen', nref.acclen, 'nchunks', nref.nchunks)

    overflow_ct_ref = np.sum(overflow_files_ref<pass_start_unix)
    overflow_ct_nref = np.sum(overflow_files_nref<pass_start_unix)
    print('Overflows ref, nref:', overflow_ct_ref, overflow_ct_nref)
    start_specnum_ref = ref.spec_num_start + overflow_ct_ref*(2**32)
    start_specnum_nref = nref.spec_num_start + overflow_ct_nref*(2**32)
    specnum_offset = start_specnum_ref - start_specnum_nref
    print('Initial Specnum Offset:', specnum_offset)


    #====================================================================================================
    #ITERATE OVER CHUNKS

    #set these once up here and adjust via chunkidx
    ref_specnum_start = ref.spec_num_start
    nref_specnum_start = nref.spec_num_start

    print('starting specnum ref:', ref_specnum_start)
    print('starting specnum nref:', nref_specnum_start)

    for chunkidx, (chunk_ref, chunk_nref) in enumerate(zip(ref, nref)):
        print(f"===========CHUNK {chunkidx}=========")

        data_ref = cp.zeros((c_acclen, nchans), dtype="complex64") 
        data_nref = cp.zeros((c_acclen, nchans), dtype="complex64")

        # print(f'\n----- Chunk {chunkidx}/{nchunks-1} ------')
        chunk_start_unix = chunk_times[chunkidx] + batch_start_ts + indent
        chunk_end_unix = chunk_times[chunkidx+1] + batch_start_ts + indent
        print('chunk start', chunk_start_unix, 'aka', chunk_times[chunkidx])
        print('chunk end', chunk_end_unix, 'aka', chunk_times[chunkidx+1])

        # print('CHUNK IDX', chunkidx)
        # print('NCHUNKS', nchunks)
        # if chunkidx == nchunks:
        #     print('offbyone error!!')
        #     sys.exit()

        perc_missing_ref = (1 - len(chunk_ref["specnums"]) / c_acclen) * 100
        perc_missing_nref = (1 - len(chunk_nref["specnums"]) / c_acclen) * 100
        if perc_missing_ref > 10 or perc_missing_nref > 10:
            print(f'big problem! plenty of data missing {perc_missing_ref} and {perc_missing_nref}. abort!')
            if chunkidx == 0: #logic here is that maybe after a reboot the data is corrupted or something: will let chunk 1 free
                sys.exit()
                #continue
            #sys.exit()  #otherwise we will need to check what's going on. may be a sign of something ystemically wrong in data

        #ref_spec_to_use = chunk_ref['specnums'] - (ref_specnum_start+c_acclen*chunkidx)
        #nref_spec_to_use = chunk_nref['specnums'] - (nref_specnum_start+c_acclen*chunkidx)
        print(chunk_ref['specnums'][0])
        print(chunk_ref['specnums'][0] - ref_specnum_start)
        print(chunk_ref['specnums'] - (ref_specnum_start+c_acclen*chunkidx))
        bdc.make_continuous_gpu(chunk_ref['pol0'],
                                #ref_spec_to_use,
                                chunk_ref['specnums'] - (ref_specnum_start),
                                np.arange(nchans),
                                c_acclen,
                                nchans=nchans, 
                                out=data_ref)

        bdc.make_continuous_gpu(chunk_nref['pol0'],
                                #nref_spec_to_use,
                                chunk_nref['specnums'] - (nref_specnum_start),
                                np.arange(nchans),
                                c_acclen,
                                nchans=nchans, 
                                out=data_nref)

        print('ref: data shape', data_ref.shape, 'percentage missing', perc_missing_ref)
        print('ref: data shape', data_nref.shape, 'percentage missing', perc_missing_nref)
        #break
        #continue

    #sys.exit()

        cx = []
        freqs = 250e6 * (1 - cp.arange(1834, 1852) / 4096)
        data_nref_delayed = cp.zeros((c_acclen, nchans), dtype="complex64")

        #niter = int(chunk_end_unix - chunk_start_unix) + 2
        niter = int(pass_end_unix - pass_start_unix) + 2

        print('number of iterations for a chunk:', niter)

        delays = np.zeros((c_acclen, len(sats_present)))
        for i, satidx in enumerate(sats_present):
            print('Getting delays for sat:', satmap[satidx])
            #d = outils.get_sat_delay(coords_ref, coords_nref, tle_path, chunk_start_unix, niter, satmap[satidx])
            d = outils.get_sat_delay(coords_ref, coords_nref, tle_path, pass_start_unix, niter, satmap[satidx])
            delays[:, i] = np.interp(np.arange(0, c_acclen) * T_SPECTRA, np.arange(0, niter), d)
        delays = cp.asarray(delays)
        print(delays.shape)
        print('delays', delays[0])

        print("chunk", chunkidx)
        print("delay first sample", delays[0,0])
        print("delay last sample", delays[-1,0])
        #continue
        #====================================================================================================
        #GET CXCORR and DETECTIONS

        #uncorrected
        print('getting uncorrected cxcorr')
        cx.append(outils_g.coarse_xcorr(data_ref, data_nref, dN))
        #beamformed
        for i, satidx in enumerate(sats_present):
            print("getting cxcorr of sat:", satmap[satidx])
            outils_g.apply_delay(data_nref, delays[:,i], freqs, out=data_nref_delayed)
            cx.append(outils_g.coarse_xcorr(data_ref, data_nref_delayed, dN))
        #snr
        snr_arr = np.zeros((len(sats_present) + 1, nchans), dtype="float64")  #for each chan for each sat (plus uncorrected)
        for i in range(len(sats_present) + 1):
            print('CX GPU SHAPE', cx[i].shape)
            print('nchans', len(cx[i]))
            snr_arr[i, :] = cp.asnumpy(cp.max(cp.abs(cx[i]), axis=1) / outils_g.median_abs_deviation(cp.abs(cx[i]),axis=1))
        #detections
        cx_cpu = []
        for i, cxcorr in enumerate(cx):
            cx_cpu.append(cxcorr.get())
            fig = fgs.make_cxcorr_plot(cxcorr, title=f'SAT {temp_satmap[i]}')
            fig.savefig(os.path.join(path_pulse, f'c{chunkidx}_cxcorr_{temp_satmap[i]}.png'))

        detected_sats, detected_peaks, detected_snrs, rel_ratios = su.get_detections(cx_cpu, snr_arr, temp_satmap)
        fig = fgs.make_snr_plot(snr_arr, temp_satmap)
        fig.savefig(os.path.join(path_pulse, f'c{chunkidx}_snrs.png'))

        print('\noverall snr array:\n', np.array(snr_arr, dtype=int))
        print('detected sats:', detected_sats)
        print('detected peaks:', detected_peaks)
        print('detection snrs:', detected_snrs)
        print('rel ratios:', rel_ratios)



#====================================================================================================
#GET VISIBILITIES FOR WHOLE PULSE

#for batch 2
# mars2    -1884333
# mars4    -2427320
# mars5    -1895548
# mars6
# mars7    -2123503

#for nov25 batch 1 (tentative)
# mars4    -148997 or -148998
# mars7    -613585

if DO_VIS:
    vis, channels = sug.get_vis_gpu(pass_start_unix,
                                    pass_end_unix, 
                                    [path_ref, path_nref],
                                    [0, -613585],
                                    T_SPECTRA = T_SPECTRA,
                                    v_acclen = v_acclen)
    pol00 = vis[0, 2, :, :].T
    fig, ax = plt.subplots(1, 2, figsize=(16,5))
    img=ax[0].imshow(np.angle(pol00),aspect='auto',interpolation='none',cmap='RdBu')
    cbar=plt.colorbar(img,ax=ax[0])
    ax[0].set_ylabel(f'Visibilities (~{int(v_acclen*T_SPECTRA*1e3)} ms)')
    ax[0].set_xlabel(f'Channel (~{int(250e6/(4096))} Hz)')
    ax[0].set_title('Phase')

    img=ax[1].imshow(np.abs(pol00),aspect='auto', interpolation='none')
    cbar=plt.colorbar(img,ax=ax[1])
    ax[1].set_xlabel(f'Channel (~{int(250e6/(4096))} Hz)')
    ax[1].set_title('Amp')
    fig.savefig(os.path.join(path_pulse, f'amp_vis.png'))


if DO_PHASE:
    #plot delays
    niter = pend-pstart+1
    chunk_length = T_SPECTRA * v_acclen
    pulse_len_chunks = int(np.ceil((pass_end_unix - pass_start_unix)/chunk_length))
    nsats = len(sats_present)

    fig, ax = plt.subplots(2, nsats, figsize=(8*nsats, 12), squeeze=False)
    for i, satidx in enumerate(sats_present):
        d = outils.get_sat_delay(coords_ref, coords_nref, tle_path, pass_start_unix, niter, satmap[satidx])
        delays = np.interp(np.arange(0, pulse_len_chunks) * chunk_length, np.arange(0, niter), d)
        ax[0, i].plot(delays)
        phase = np.unwrap(np.angle(np.exp(2j*np.pi*delays*137e6)))
        ax[1, i].plot(phase)
        ax[1, i].set_xlabel('Chunks')
    ax[0, 0].set_ylabel('Delay')
    ax[1,0].set_ylabel('Phase')
    fig.savefig(os.path.join(path_pulse, f'all_phases.png'))


print(pass_start_unix)
print(pass_end_unix)
print(path_ref)
print(path_nref)
print(coords_ref)
print(coords_nref)
print(sats_present)