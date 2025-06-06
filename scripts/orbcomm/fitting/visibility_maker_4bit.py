import os
import sys
from os import path
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
import argparse
from scipy.optimize import least_squares
import coord_helper as ch
import h5py


''' 
Very simple script that saves visibility data into a h5 file.
The visibility is done in the way that I have computed it for a while now

To check/debug:
- spectrum number consistent for everything
- saves properly
- indices of pulses and data are set up correctly (maybe add an index in saving?)
- careful with the channel indices:
    - might be hard to know which channel corresponds to what: they all have different present channels.
    - might need to save the info of what channels are present as an extra attribute. that way I know exactly
      which channel I'm working with

'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Config file containing all required data.",
    )

    #later make this configurable for multiple, ig.
    parser.add_argument(
        "baseline", type=int, help="Which baseline we are getting visibilities for"
    )

    parser.add_argument(
        "-o", "--output_path", type=str, default="/project/s/sievers/thomasb", help="Output directory for data and debug"
    )

    parser.add_argument(
        "-dg", "--debug", action="store_true", help="debug option that spits out a ton of plots to see stuff"
    )

    args = parser.parse_args()


with open(f"{args.config_file}", "r") as f:
    config = json.load(f)
    dir_parents = []
    coords = []
    # unpack information from the json file
    # Call get_starting_index for all antennas except reference
    print('\n', "Antenna Details:")
    for i, (ant, details) in enumerate(config["antennas"].items()):
        if (i == 0) or (i ==args.baseline):
            coords.append(details['coordinates'])
            dir_parents.append(details["path"])
    global_start_time = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    v_acclen = config["correlation"]["vis_acclen"]
    visibility_window = config["correlation"]["visibility_window"]
    T_SPECTRA = config["correlation"]["point_PFB"] / config["correlation"]["sample_rate"]

print("Antenna Paths:", dir_parents, '\n')
print("Antenna Coordinates:", coords, '\n')
print("Visibility Accumulation Length", v_acclen, '\n')


V_T_ACCLEN = v_acclen* T_SPECTRA
v_nchunks = int((visibility_window)/V_T_ACCLEN)

tle_path = outils.get_tle_file(global_start_time, "/project/s/sievers/mohanagr/OCOMM_TLES")

ref_coords = coords[0]
ref_path = dir_parents[0]

a2_coords = coords[1]
a2_path = dir_parents[1]

#for only visibilities I don't think I need this
#context = [global_start_time, visibility_window, [T_SPECTRA, v_acclen, v_nchunks], ref_coords, tle_path]


with open(f"./pulsedata/pulsedata_{global_start_time}_1748962214.json", "r") as f:
        pulsedata = json.load(f)
        
        info = []
        offsets = []
        for pulse_idx, details in enumerate(pulsedata[f"{global_start_time}"]["antenna 1"]):
            #print(f"\n--------pulse idx {pulse_idx}---------")
            start_time = details["start"] * 5
            end_time = details["end"] * 5
            rel = False

            #For now, disregard multiple sat passes. For v2.
            if len(details["sats_present"]) > 1:
                continue

            for satID, values in details["sats_present"].items():
                
                for i in range(len(values)):
                    pulse_info = []
                    #print(values[i])
                    chan = values[i][0]
                    corr_offset = values[i][1]
                    if values[i][2][0] > 0.9:
                        rel = True
                    pulse_info.append(start_time)
                    pulse_info.append(end_time)
                    pulse_info.append(int(satID))
                    pulse_info.append(chan)
                    pulse_info.append(global_start_time)
                    pulse_info.append(tle_path)
                    pulse_info.append(corr_offset)

                    info.append(pulse_info)

            if rel==True:
                offsets.append(details["timestream_offset"])
                
print(info)
print(offsets)
specnumoffset = int(stats.mode(offsets)[0])
print(specnumoffset)
print(len(info))

#note that I'm just writing absolutely everything into the h5 file. no pulse filtering is done here. 
#also notice that in principle, the info list should be modular: can get visibilities purely from these. 


#the one thing that I might add to metadata (but doesn't seem necessary) is overall specnum offset. Could be a move. For later

observed_data = []
time_total = time.time()


for pulse_idx in range(len(info)):

    print(f"---------STARTING PULSE {pulse_idx}---------")

    #--------times-----
    relative_start_time = info[pulse_idx][0]
    relative_end_time = info[pulse_idx][1]
    relative_specnumoffset = info[pulse_idx][6]
    global_start_time = info[pulse_idx][4]

    pulse_duration_secs = relative_end_time - relative_start_time
    pulse_duration_chunks = int((pulse_duration_secs)/ (T_SPECTRA * v_acclen)  )

    t_start = global_start_time + relative_start_time
    t_end = t_start + visibility_window

    #----get initialized information----
    files_ref, idx_ref = butils.get_init_info(t_start, t_end, ref_path)
    files_a2, idx2 = butils.get_init_info(t_start, t_end, a2_path)

    #------get corrected offsets-----
    idx_correction = relative_specnumoffset - 100000
    if idx_correction>0:
        idx_ref = idx_ref + idx_correction
    else:
        idx2 = idx2 + np.abs(idx_correction)
    #print("Corrected Starting Indices:", idx1_v, idx2_v)

    #-------set up channels-------
    channels = bdc.get_header(files_ref[0])["channels"].astype('int64')
    chanstart = np.where(channels == 1834)[0][0] 
    chanend = np.where(channels == 1852)[0][0]
    nchans=chanend-chanstart

    chan_bigidx = info[pulse_idx][3] #index in total list , e.g. 1836 (for predicted phases)
    chanmap = channels[chanstart:chanend].astype(int)  #list of all the channels between 1834 and 1852
    chan_smallidx = np.where(chanmap == chan_bigidx)[0][0] #index in small list, e.g. 2 (for data selection)

    #--------open object----------
    ant_ref = bdc.BasebandFileIterator(files_ref, 0, idx_ref, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')
    ant2 = bdc.BasebandFileIterator(files_a2, 0, idx2, v_acclen, nchunks= v_nchunks, chanstart = chanstart, chanend = chanend, type='float')

    #--------get visibilities-----
    m_ref = ant_ref.spec_num_start
    m2 = m_ref + specnumoffset
    print(specnumoffset)
    print(m_ref - ant2.spec_num_start)

    visibility_phased = np.zeros((v_nchunks,len(ant_ref.channel_idxs)), dtype='complex64')
    time_pulse=time.time()
    #print(f"--------- Processing Pulse Idx {pulse_idx} ---------")
    for i, (chunk1,chunk2) in enumerate(zip(ant_ref,ant2)):
            xcorr = ch.avg_xcorr_4bit_2ant_float(
                chunk1['pol0'], 
                chunk2['pol0'],
                chunk1['specnums'],
                chunk2['specnums'],
                m_ref+i*v_acclen,
                m2+i*v_acclen)
            visibility_phased[i,:] = np.sum(xcorr,axis=0)/v_acclen
            #print("CHUNK", i, " has ", xcorr.shape[0], " rows")
    print(f"DONE PULSE {pulse_idx}. TIME:", time.time()-time_pulse)
    visibility_phased = np.ma.masked_invalid(visibility_phased)
    #vis_phase = np.angle(visibility_phased)
    #obs = np.unwrap(vis_phase[0:pulse_duration_chunks, chan_smallidx])
    observed_data.append(visibility_phased)
    break



#perhaps save this in order? somehow make sure that there is a consistent index to it all?
#maybe include a man-made index to keep track of them all, maybe included in the name of each dataset?
#also, save this to a specific location on project or scratch or something.

#ALSO ALSO save the WRAPPED visibility stuff, not the unwrapped. otherwise it's a huge pain

with h5py.File(f'{args.output_path}/vis_all_{global_start_time}.h5', 'w') as f:
    for pulse_idx, observed in enumerate(observed_data):
        pulse_array = f.create_dataset(f'{start_time}_{chan}', data=observed)

        pulse_array.attrs['start_time'] = info[pulse_idx][0]
        pulse_array.attrs['end_time'] = info[pulse_idx][1]
        pulse_array.attrs['satID'] = info[pulse_idx][2]
        pulse_array.attrs['chan'] = info[pulse_idx][3]
        pulse_array.attrs['global_start_time'] = info[pulse_idx][4]
        pulse_array.attrs['tle_path'] = info[pulse_idx][5]

    f.attrs["vis_window"] = visibility_window
    f.attrs["v_acclen"] = v_acclen
    f.attrs["T_SPECTRA"] = V_T_ACCLEN
    f.attrs["v_nchunks"] = v_nchunks
    f.attrs["ref_coords"] = ref_coords #necessary?




