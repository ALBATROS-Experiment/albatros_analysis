import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os import path
import sys
import helper
sys.path.insert(0,path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
import json
from pyuvdata import UVData
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u


#set this up so we can run it from other files as well as the terminal
def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        default="config3.json", 
        help="Path to config file"
        )

    parser.add_argument(
        "-o", 
        '--outdir', 
        type=str, 
        default="/project/sievers/s/thomasb", 
        help="Output file path"
        )

    return parser.parse_args()



#the functionality of the module hides in a function, with the only argument being the config file
def run_script(config_path, outdir):
    with open(config_path, "r") as f:
        config = json.load(f)

    #determine ref antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )

    #extract antenna information
    ant_names = []
    ant_coords = []
    dir_parents = []
    spec_offsets = []
    for i, (ant, details) in enumerate(config["antennas"].items()):
        ant_names.append(details["name"])
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])
        #coordinates in ITRS
        LLAcoords = details["coordinates"]
        location = EarthLocation(lat=LLAcoords[0]*u.deg, lon=LLAcoords[1]*u.deg, height=LLAcoords[2]*u.m)
        ant_coords.append(location.itrs.cartesian.xyz.to(u.m).value)
        
    print('ant coords', ant_coords)    
    #extract rest of information
    init_t = config["correlation"]["start_timestamp"]
    end_t  = config["correlation"]["end_timestamp"]
    acclen = config["correlation"]["accumulation_length"]
    t_acclen = acclen*4096/250e6
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)

    #compute data
    pols,rowcounts,channels = helper.get_avg_fast_tb(idxs, files, acclen, nchunks, chanstart, chanend)

    #create UVdata object and assign basic constants
    uv = UVData()
    uv.Ntimes, uv.Nbls, uv.Npols, uv.Nfreqs = pols.shape
    uv.Nblts = uv.Ntimes * uv.Nbls
    uv.Nants_data = len(config["antennas"].items())
    uv.Nants_telescope = uv.Nants_data #assume all ants have data
    uv.antenna_names = ant_names
    uv.antenna_numbers = np.arange(uv.Nants_data, dtype=int)
    uv.antenna_positions = np.array(ant_coords, dtype=float)
    uv.telescope_location = uv.antenna_positions[0]
    print('uv tel loc', uv.telescope_location)

    #PRIMARY DATA
    nt = uv.Ntimes
    na = uv.Nants_data
    v_l = []
    f_l = []
    a1_l = []
    a2_l = []
    c_l = []
    t_l = []
    ns_l = []

    idx = 0
    for i in range(na):
        for j in range(i+1,na):
            data = pols.data[:,idx,:,:].transpose(0,2,1)
            flags = pols.mask[:,idx,:,:].transpose(0,2,1)
            nsamples = rowcounts[:,idx]
            v_l.append(data)
            f_l.append(flags)
            ns_l.append(nsamples)

            a1_l.extend([i] * nt)
            a2_l.extend([j] * nt)

            code = int(i*2048 + j + 2**(16))
            c_l.extend([code] * nt)

            unix_times = np.linspace(init_t, init_t + (uv.Ntimes - 1) * t_acclen, uv.Ntimes)
            times = Time(unix_times, format='unix', scale='utc')
            jtimes = times.jd
            t_l.append(jtimes) # we do stuff in julian times

            print((i,j), idx)
            idx += 1

    uv.data_array = np.concatenate(v_l)
    uv.flag_array = np.concatenate(f_l)
    uv.ant_1_array = np.array(a1_l)
    uv.ant_2_array = np.array(a2_l)
    uv.baseline_array = np.array(c_l)
    uv.time_array = np.concatenate(t_l)
    uv.set_lsts_from_time_array() #automatically sets LST list from other info

    #nsamples and integration time
    nsample_array = np.concatenate(ns_l) #BEWARE for nsamples: same for each channel if pol, bline kept constant.
    uv.nsample_array = np.tile(nsample_array[:, np.newaxis, :], (1, uv.Nfreqs, 1))
    uv.nsample_array = uv.nsample_array.astype(float)
    uv.integration_time = np.full(uv.Nblts, t_acclen)

    print("saved data. shape:", uv.data_array.shape)


    #frequency stuff
    assert uv.Nfreqs == uv.data_array.shape[1]  #safety check
    width = 250e6 / 4096
    freqs = np.zeros(uv.Nfreqs)
    for i in range(uv.Nfreqs):
        freqs[i] = outils.chan2freq(channels[i])
    uv.freq_array = freqs
    uv.channel_width = np.full(uv.Nfreqs, width)

    #spectral window stuff
    uv.Nspws = 1
    uv.spw_array = np.array([0])
    uv.flex_spw_id_array = np.zeros((uv.Nfreqs), dtype=int)

    #ADD WHICH EXACT TYPE OF POLARIZATION WE ARE WORKING WITH
    uv.polarization_array = np.arange(1,3, dtype = int)  # e.g., XX

    #metadata
    uv.telescope_name = 'FakeTelescope'
    uv.instrument = 'FakeInstrument'
    uv.history = 'History'
    uv.object_name = 'Object Name'
    uv.vis_units = 'uncalib'

    #bare minimum phase center content. will not worry about this for now
    uv.Nphase = 1
    uv.phase_center_catalog = {
        0: {
            "cat_name": "zenith",        
            "cat_type": "sidereal",      
            "cat_lon": 0.0,             
            "cat_lat": 0.0,             
            "cat_frame": "icrs",  
        }
    }
    uv.phase_center_id_array = np.zeros(uv.Nblts, dtype=int)
    uv.phase_center_app_ra = np.zeros(uv.Nblts)
    uv.phase_center_app_dec = np.zeros(uv.Nblts)
    uv.phase_center_frame_pa = np.zeros(uv.Nblts)
    uv.uvw_array = np.zeros((uv.Nblts, 3))
    uv.set_uvws_from_antenna_positions()


    # save the data
    uv.write_uvh5('/project/s/sievers/thomasb/example_output.uvh5', clobber=True)

    print("done")


#allows us to run this from the terminal
if __name__ == "__main__":
    args = arguments()
    run_script(args.config, args.outdir)

