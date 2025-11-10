import json
import os
import sys
from os import path
sys.path.insert(0, "/home/thomasb")
import numpy as np
from albatros_analysis.src.utils import orbcomm_utils as outils
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def unix_to_lst(unix_t, coords):
    loc = EarthLocation(lon=coords[1]*u.deg, lat=coords[0]*u.deg, height=coords[2]*u.m)
    t_obj = Time(unix_t, format="unix", location=loc)
    lst_time = t_obj.sidereal_time("mean").degree % 360
    return lst_time

    
def print_no_sat_times(data):
    time_set = set()
    for timestamp, all_data in data.items():
        for antname, antenna_data in all_data.items():
            for pulse in antenna_data['pulse_data']:
                start = pulse["start"]
                end = pulse["end"]
                time_set.add((start, end))
        
    times = list(time_set)
    sorted_times = sorted(times, key=lambda x : x[0])

    gaps = []
    lengths = []
    for i in range(1, len(sorted_times)):
        start = sorted_times[i-1][1]
        end = sorted_times[i][0]
        gaps.append((start, end))
        lengths.append(end-start)

    #print('all sat gaps:')
    #for gap in gaps:
    #    print(gap)
    print('mean seconds with no sat risen', int(np.mean(lengths)))
    print('in minutes:', int(np.mean(lengths)/60))
    print('longest in minutes:', int(max(lengths)/60))



def get_shared_pulses(
        config_data,
        pulse_data,
        satlist = [28654,25338,33591,57166,59051,44387],
        T_SCAN = 5,
        alt_cutoff = 15):
    
    ''' 
    Gets the pulses that are shared in all baselines, that all see the same satellite.


    Parameters
    ----------


    Returns
    -------

    
    
    '''

    ref_coords = config_data["antennas"]["ALB1"]["coordinates"]
    print(ref_coords)
    global_start_t = config_data["correlation"]["start_timestamp"]
    global_end_t = config_data["correlation"]["end_timestamp"]
    array_time =  global_end_t - global_start_t
    tle_path = outils.get_tle_file(global_start_t, "/project/rrg-sievers/mohanagr/OCOMM_TLES")

    satmap = {} 
    assert min(satlist) > len(satlist)
    for i, sat_ID in enumerate(satlist):
        satmap[i] = sat_ID
        satmap[sat_ID] = i
    print(satmap)

    #RISEN SATS
    nrows = int((array_time)/T_SCAN)
    arr = np.zeros((nrows, len(satlist)), dtype="int64")
    rsats = outils.get_risen_sats(tle_path, 
                                  ref_coords, 
                                  global_start_t, 
                                  dt=T_SCAN, 
                                  niter=nrows, 
                                  good=satlist, 
                                  altitude_cutoff=alt_cutoff)
    for i, row in enumerate(rsats):
        for sat_ID, satele, sataz in row:
            arr[i,satmap[sat_ID]] = 1

    #PASSES
    p_5s = outils.get_simul_pulses(arr)
    passes = []
    for p in p_5s:
        passes.append([[p[0][0]*5, p[0][1]*5], p[1]])
    npasses = len(passes)
    for passs in passes:
        print(passs)
    print("PASSES DETECTED:",'\n', passes, '\n')
    print("Number of Passes:", npasses, '\n')

    bline_data_all = pulse_data[f'{global_start_t}'] #good check to see if I have the right file

    shared_passes = []
    for satpass in passes:
        pass_start = satpass[0][0]
        print('STARTING PASS', pass_start)
        temp_dict = {}
        temp_dict[f'{pass_start}'] = {}
        sat_set = set() #could make this all possible satellites, not just in rsats?

        for sat_idx in satpass[1]:
            sat_set.add(satmap[sat_idx])

        for antenna, bline_data in bline_data_all.items():
            bline_data = bline_data["pulse_data"]
            found = False #start looking for this pass in this bline

            for pulse in bline_data:
                pulse_start = pulse['start']
                if pass_start<pulse_start:
                    print(f'looked past pass start in {antenna}!')
                    print("(will tell you it didn't see the NEXT pass)")
                    found = False
                    break

                if pulse["start"] == pass_start:
                    print(f'found {pulse_start} pass in {antenna}')
                    found = True
                    sats = set(int(sat) for sat in pulse['sats_present'].keys())
                    print(f'we see {sats} in {pulse_start} on {antenna}')
                    sat_set &= sats #check if all antenna share same sat
                    temp_dict[f'{pass_start}'][f'{antenna}'] = pulse #will only want specific information, not whole pulse
                    break

            if not found:
                print(f'{pulse_start} not found in {antenna}.')
                print('SKIPPING ENTIRE PASS\n')
                break
        
        if sat_set and found:
            print(f'pass seen everywhere, using {sat_set}\n')
            shared_passes.append(temp_dict)

    print('number of shared passes', len(shared_passes))
    return shared_passes

def get_sim_passes_lst(json_paths, coords, same_sat = False, fixed_sats = None, fixed_ant = None):
    """
    get the simultaneous passes across pulse_data files for same lst times

    Parameters
    ----------
    json_paths : list of str
        paths to the json pulse_data files. 
    location : astropy.coordinates.EarthLocation
        where the telescope is

    Returns
    -------
    sim_passes: list of dicts
        list of all the passes which happen in two or more files at same time
    """

    # STEP 1: extract all pulses into a dictionary
    interval_bounds = []  
    pulse_dict = {} 

    for path in json_paths:
        data = load_json(path)

        for interval_start_str, antenna_data in data.items():
            interval_start = int(interval_start_str)
            interval_end = interval_start
            for ant_key, ant_dict in antenna_data.items():
                if fixed_ant:
                    if ant_key != fixed_ant:
                        continue
                for pulse in ant_dict["pulse_data"]:
                    unix_start = int(interval_start+ pulse["start"])
                    unix_end = int(interval_start + pulse["end"])
                    lst_start = unix_to_lst(unix_start, coords)
                    lst_end = unix_to_lst(unix_end, coords)

                    if unix_end >interval_end:
                        interval_end = int(unix_end)

                    for sat_id in pulse["sats_present"]:
                        snr = pulse['sats_present'][sat_id][0][2]
                        if unix_start not in pulse_dict:
                            pulse_dict[unix_start] = {
                                "lst": (lst_start, lst_end),
                                "unix": (unix_start, unix_end),
                                "antenna": set(),
                                "sats": set(),
                                "SNR": set(),
                                "file": str(path),
                            }
                        pulse_dict[unix_start]["antenna"].add(ant_key)
                        pulse_dict[unix_start]["sats"].add(int(sat_id))
                        pulse_dict[unix_start]['SNR'].add(snr)

            interval_bounds.append((interval_start, interval_end, path))


    # STEP 2: CHECK NO FILES OVERLAP
    sorted_bounds = sorted(interval_bounds, key=lambda x: x[0])
    for i in range(1, len(sorted_bounds)):
        prev_end = sorted_bounds[i-1][1]
        curr_start = sorted_bounds[i][0]
        if curr_start < prev_end:
            raise ValueError(f"Intervals overlap: {sorted_bounds[i-1]} and {sorted_bounds[i]}")


    #STEP 3: CONVERT PULSE DICTIONARY TO LIST AND SORT
    pulse_entries = []
    for entry in pulse_dict.values():
        entry["antenna"] = list(entry["antenna"])
        entry["sats"] = list(entry["sats"])
        pulse_entries.append(entry)
        
    pulse_entries.sort(key=lambda x: x['lst'][0])  # Sort by LST start

    #STEP 4: GET OVERLAPPING PULSES
    sim_pulses = []
    for i in range(len(pulse_entries)):
        e1 = pulse_entries[i]
        lst1_start, lst1_end = e1['lst']
        group = [e1]

        for j in range(i+1, len(pulse_entries)):
            e2 = pulse_entries[j]
            lst2_start, lst2_end = e2['lst']

            if lst2_start >= lst1_end:
                break  # can't overlap any more, go to next

            sats1, sats2 = set(e1["sats"]), set(e2["sats"])

            if same_sat:
                if not sats1 & sats2:  # no common satellite
                    continue

            if fixed_sats:
                f_sats = set(fixed_sats)
                if not (f_sats & sats1) or not (f_sats & sats2):
                    continue

            if lst2_start < lst1_end and lst2_end > lst1_start:
                group.append(e2)

        if len(group) >= 2:
            sim_pulses.append(group)

    return sim_pulses




 #GET DIFF IN SNR

