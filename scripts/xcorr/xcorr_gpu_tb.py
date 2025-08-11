import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os import path
import sys
import helper as hp
import helper_gpu as hpg
sys.path.insert(0,path.expanduser("~"))
import json
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time
from pyuvdata import UVData


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



def run_script(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["offset"],
    )
    ant_names = []
    ant_coords = []
    dir_parents = []
    spec_offsets = []
    for i, (ant, details) in enumerate(config["antennas"].items()):
        ant_names.append(details["name"])
        dir_parents.append(details["path"])
        spec_offsets.append(details["offset"])
        LLAcoords = details["coordinates"]
        location = EarthLocation(lat=LLAcoords[0]*u.deg, lon=LLAcoords[1]*u.deg, height=LLAcoords[2]*u.m)
        ant_coords.append(location.itrs.cartesian.xyz.to(u.m).value)
    nants = len(ant_names)

    osamp = 64
    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    pfb_size = config["correlation"]["accumulation_length"]
    #t_acclen = pfb_size*4096/250e6
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size/osamp))
    idxs, files = hp.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)

    print("total time (in s) of visibility computation", end_t - init_t)
    print("final idxs", idxs)
    print("PFB SIZE:", pfb_size)
    print("OVERSAMPLING", osamp)

    t1=time.time()
    pols,missing_fraction,channels=hpg.repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize=16,filt_thresh=0.45)
    t2=time.time()
    print("Total time taken", t2-t1)

    uv = UVData()
    uv.Nants_telescope, uv.Nants_data = nants, nants
    _, _, uv.Nfreqs, uv.Ntimes = pols.shape
    uv.Nbls = ((nants) * (nants-1))/2
    uv.Nblts = uv.Ntimes * uv.Nbls
    uv.Nants_telescope = uv.Nants_data #assume all ants have data
    uv.antenna_names = ant_names
    uv.antenna_numbers = np.arange(uv.Nants_data, dtype=int)
    uv.antenna_positions = np.array(ant_coords, dtype=float)
    uv.telescope_location = uv.antenna_positions[0]
    print('uv tel loc', uv.telescope_location)



    #PRIMARY DATA
    nt = uv.Ntimes
    na = uv.Nants_data
    visl = []
    freql = []
    a1l = []
    a2l = []
    codel = []
    timel = []
    nsl = []

    #fix this to work for the GPU function
    idx = 0
    for i in range(na):
        for j in range(i+1,na):
            #data = pols.data[:,idx,:,:].transpose(0,2,1)
            #flags = pols.mask[:,idx,:,:].transpose(0,2,1)
            #nsamples = rowcounts[:,idx]
            #v_l.append(data)
            #f_l.append(flags)
            #ns_l.append(nsamples)

            #a1_l.extend([i] * nt)
            #a2_l.extend([j] * nt)

            #code = int(i*2048 + j + 2**(16))
            #c_l.extend([code] * nt)

            #unix_times = np.linspace(init_t, init_t + (uv.Ntimes - 1) * t_acclen, uv.Ntimes)
            #times = Time(unix_times, format='unix', scale='utc')
            #jtimes = times.jd
            #t_l.append(jtimes) # we do stuff in julian times

            #print((i,j), idx)
            #idx += 1

    uv.data_array = np.concatenate(visl)
    uv.flag_array = np.concatenate(freql)
    uv.ant_1_array = np.array(a1l)
    uv.ant_2_array = np.array(a2l)
    uv.baseline_array = np.array(codel)
    uv.time_array = np.concatenate(timel)
    uv.set_lsts_from_time_array() #automatically sets LST list from other info

    #nsamples and integration time
    nsample_array = np.concatenate(nsl) #BEWARE for nsamples: same for each channel if pol, bline kept constant.
    uv.nsample_array = np.tile(nsample_array[:, np.newaxis, :], (1, uv.Nfreqs, 1))
    uv.nsample_array = uv.nsample_array.astype(float)
    uv.integration_time = np.full(uv.Nblts, t_acclen)

    print("saved data. shape:", uv.data_array.shape)






if __name__ == "__main__":
    args = arguments()
    pols, missing_fraction, channels = run_script(args.config)
    fname = f"xcorr_all_ant_4bit_{str(init_t)}_{str(pfb_size)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,missing_fraction=missing_fraction,chans=channels)

