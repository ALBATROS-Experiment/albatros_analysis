import os
import sys
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np 
import numba as nb
import time
from scipy import linalg
from scipy import stats
from matplotlib import pyplot as plt
from datetime import datetime as dt
from src.correlations import baseband_data_classes as bdc
from src.utils import baseband_utils as butils
from src.utils import orbcomm_utils as outils
import json


#define some basic functions here, like xcorr and etc

@nb.njit()
def get_common_rows(specnum0,specnum1,idxstart0,idxstart1):
    nrows0,nrows1=specnum0.shape[0],specnum1.shape[0]
    maxrows=min(nrows0,nrows1)
    rownums0=np.empty(maxrows,dtype='int64')
    rownums0[:]=-1
    rownums1=rownums0.copy()
    rowidx=rownums0.copy()
    i=0;j=0;row_count=0;
    while i<nrows0 and j<nrows1:
        if (specnum0[i]-idxstart0)==(specnum1[j]-idxstart1):
            rownums0[row_count]=i
            rownums1[row_count]=j
            rowidx[row_count]=specnum0[i]-idxstart0
            i+=1
            j+=1
            row_count+=1
        elif (specnum0[i]-idxstart0)>(specnum1[j]-idxstart1):
            j+=1
        else:
            i+=1
    return row_count,rownums0,rownums1,rowidx

@nb.njit(parallel=True)
def avg_xcorr_4bit_2ant_float(pol0,pol1,specnum0,specnum1,idxstart0,idxstart1,delay=None,freqs=None):
    row_count,rownums0,rownums1,rowidx=get_common_rows(specnum0,specnum1,idxstart0,idxstart1)
    ncols=pol0.shape[1]
#     print("ncols",ncols)
    assert pol0.shape[1]==pol1.shape[1]
    xcorr=np.zeros((row_count,ncols),dtype='complex64') # in the dev_gen_phases branch
    if delay is not None:
        for i in nb.prange(row_count):
            for j in range(ncols):
                xcorr[i,j] = pol0[rownums0[i],j]*np.conj(pol1[rownums1[i],j]*np.exp(2j*np.pi*delay[rowidx[i]]*freqs[j]))
    else:
        for i in nb.prange(row_count):
            xcorr[i,:] = pol0[rownums0[i],:]*np.conj(pol1[rownums1[i],:])
    return xcorr

def get_coarse_xcorr(f1, f2, Npfb=4096):
    if len(f1.shape) == 1:
        f1 = f1.reshape(-1, 1)
    if len(f2.shape) == 1:
        f2 = f2.reshape(-1, 1)
    chans = f1.shape[1]
    Nsmall = f1.shape[0]
    wt = np.zeros(2 * Nsmall)
    wt[:Nsmall] = 1
    n_avg = np.fft.irfft(np.fft.rfft(wt) * np.conj(np.fft.rfft(wt)))
#     print(n_avg)
#     n_avg[Nsmall] = np.nan
#     print(n_avg[Nsmall-10:Nsmall+10])
    n_avg = np.tile(n_avg, chans).reshape(chans, 2*Nsmall)
#     print(n_avg.shape)
    bigf1 = np.vstack([f1, np.zeros(f1.shape, dtype=f1.dtype)])
    bigf2 = np.vstack([f2, np.zeros(f2.shape, dtype=f2.dtype)])
    bigf1 = bigf1.T.copy()
    bigf2 = bigf2.T.copy()
    bigf1f = np.fft.fft(bigf1,axis=1)
    bigf2f = np.fft.fft(bigf2,axis=1)
    xx = bigf1f * np.conj(bigf2f)
    xcorr = np.fft.ifft(xx,axis=1)
    xcorr = xcorr / n_avg
    xcorr[:,Nsmall] = np.nan
    return xcorr


#----------SETUP FROM CONFIG FILE-------------

T_SPECTRA = 4096/250e6
visibility_window = 1000
DEBUG = False

with open("config_test.json", "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    # unpack information from the json file
    # Call get_starting_index for all antennas except reference
    print('\n', "Antenna Details:")
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ant, details)
        coords.append(details['coordinates'])
        dir_parents.append(details["path"])
    global_start_time = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    c_acclen = config["correlation"]["coarse_accumulation_length"]
    v_acclen = config["correlation"]["visibility_accumulation_length"]

print("Antenna Paths:", dir_parents, '\n')
print("Antenna Coordinates:", coords, '\n')
print("Visibility Accumulation Length", v_acclen, '\n')
print("Coarse Accumulation Length:", c_acclen, '\n')


C_T_ACCLEN = c_acclen* T_SPECTRA
V_T_ACCLEN = v_acclen* T_SPECTRA

c_nchunks = int((visibility_window)/C_T_ACCLEN)
v_nchunks = int((visibility_window)/V_T_ACCLEN)

tle_path = outils.get_tle_file(global_start_time, "/project/s/sievers/mohanagr/OCOMM_TLES")


#------------IMPORT INFO FROM SATELLITE DETECTION------------

# filtering process, get list filtered_pulses
# note that the channels are probably wrong. to verify (maybe using my coarse correlator script)

filtered_pulses =  [
                    [[715, 1110], [28654], [1836]], 
                    [[4950, 5270], [59051], [1836]], 
                    [[7000, 7145], [28654], [1836]], 
                    [[8615, 8970], [44387], [1836]], 
                    [[10975, 11440], [59051], [1836]], 
                    [[14630, 15110], [44387], [1836]]
                    ]


#--------------SET UP PHASE PREDICTOR--------------

# just use the phase_predictor function I made


def phase_predictor(fit_coords, pulse_idx):
    
    #depends if I want to keep ant1 as reference in general
    ref_coords = coords[0]

    relative_start_time = filtered_pulses[pulse_idx][0][0]
    pulse_duration_sec = filtered_pulses[pulse_idx][0][1] - filtered_pulses[pulse_idx][0][0]
    pulse_duration_chunks = int( pulse_duration_sec / (T_SPECTRA * v_acclen) )

    time_start = global_start_time + relative_start_time
    sat_ID = filtered_pulses[pulse_idx][1][0]

    pulse_channel_idx = filtered_pulses[pulse_idx][2][0]
    pulse_freq = outils.chan2freq(pulse_channel_idx, alias=True)

    # 'd' has one entry per second
    d = outils.get_sat_delay(ref_coords, fit_coords, tle_path, time_start, visibility_window+1, sat_ID)
    # 'delay' has one entry per chunk (~0.5s) 
    delay = np.interp(np.arange(0, v_nchunks) * v_acclen * T_SPECTRA, np.arange(0, int(visibility_window)+1), d)
    #thus 'pred' has one entry for each chunk
    pred = (-delay[:pulse_duration_chunks]+ delay[0]) *  2*np.pi * pulse_freq

    return pred

plt.plot(phase_predictor(coords[1], 1))

if DEBUG == True:
    plt.plot(phase_predictor(coords[1], 1))
    plt.savefig(f"predicted_phase_antenna{antidx}_pulse{pulseidx}.png")



#---------------GET OBSERVED PHASES-----------------

pulse_idx = 1

pulse_duration_sec = filtered_pulses[pulse_idx][0][1] - filtered_pulses[pulse_idx][0][0]
pulse_duration_chunks = int( pulse_duration_sec / (T_SPECTRA * v_acclen) )

relative_start_time = filtered_pulses[pulse_idx][0][0]
t_start = global_start_time + relative_start_time
t_end = t_start + visibility_window

files_a1, idx1 = butils.get_init_info(t_start, t_end, dir_parents[0])
files_a2, idx2 = butils.get_init_info(t_start, t_end, dir_parents[1])

print('initial offset:', idx1-idx2)

idx_correction = 109993-100000
if idx_correction>0:
    idx1+=idx_correction
else:
    idx2+=np.abs(idx_correction)

print('final offset:', idx1-idx2)

channels = bdc.get_header(files_a1[0])["channels"].astype('int64')
chanstart = np.where(channels == 1834)[0][0] 
chanend = np.where(channels == 1852)[0][0]
nchans=chanend-chanstart

ant1 = bdc.BasebandFileIterator(files_a1, 0, idx1, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')
ant2 = bdc.BasebandFileIterator(files_a2, 0, idx2, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')


m1=ant1.spec_num_start
m2=ant2.spec_num_start

visibility_phased = np.zeros((v_nchunks,len(ant1.channel_idxs)), dtype='complex64')
st=time.time()
print(f"--------- Processing Pulse Idx {pulse_idx} ---------")
for i, (chunk1,chunk2) in enumerate(zip(ant1,ant2)):
        xcorr = avg_xcorr_4bit_2ant_float(
            chunk1['pol0'], 
            chunk2['pol0'],
            chunk1['specnums'],
            chunk2['specnums'],
            m1+i*v_acclen,
            m2+i*v_acclen)
        visibility_phased[i,:] = np.sum(xcorr,axis=0)/v_acclen
        print("CHUNK", i, " has ", xcorr.shape[0], " rows")
print("Time taken final:", time.time()-st)
visibility_phased = np.ma.masked_invalid(visibility_phased)
vis_phase = np.angle(visibility_phased)


if DEBUG == TRUE:
    plt.imshow(vis_phase[:,:], aspect='auto',cmap='RdBu',interpolation="none")
    plt.savefig(f"observed_phase_antenna{antidx}_pulse_{pulseidx}.png")



#----------------FIT IT---------------
