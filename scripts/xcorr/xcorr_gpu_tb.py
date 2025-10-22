import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os import path
import sys
import helper
from pyuvdata import UVData
sys.path.insert(0,path.expanduser("~"))
import json
from astropy.time import Time
from astropy.coordinates import EarthLocation


def get_ant_pol_idxs(antpol_idx, nants, npols):
   '''
   get antenna and polarization index from certain datapoint's antpol index
   assumes antpol index uses antenna-major indexing (i.e. polarization is cycling through)
   '''
   ant_idx = antpol_idx // npols
   pol_idx = antpol_idx % npols
   return ant_idx, pol_idx

def get_poltot_idx(pol1_idx, pol2_idx):
    """
    turn (pol1, pol2) into single polarization index (two pol case)
    uses polarization_array = [-5, -6, -7, -8] which maps to [XX, YY, XY, YX] in convention
    """
    #XX case
    if pol1_idx == 0 and pol2_idx == 0:
        return 0 
    #YY case
    elif pol1_idx == 1 and pol2_idx == 1:
        return 1 
    #XY case
    elif pol1_idx == 0 and pol2_idx == 1:
        return 2 
    #YX case
    elif pol1_idx == 1 and pol2_idx == 0:
        return 3
    else:
        raise ValueError(f"wrong polarization indices: {pol1_idx} and {pol2_idx}")

def get_bline_idx(ant1_idx, ant2_idx, nants, auto=True):
   """
   turn (ant1, ant2) pair into single bline index
   assumes given visibility matrix is antenna-major. i.e. ant1<=ant2


   auto tells you if you allow auto-correlations (ant1 = ant2)
   """
   if ant1_idx > ant2_idx:
       raise ValueError("we need ant1 <= ant2 by convention")
   if not auto and ant1_idx == ant2_idx:
       raise ValueError("turned off auto-correlation but recieved ant1 == ant2")
   if auto:
       bline_idx = ant1_idx * nants - (ant1_idx * (ant1_idx - 1)) // 2 + (ant2_idx - ant1_idx)
   else:
       bline_idx = ant1_idx * (nants - 1) - (ant1_idx * (ant1_idx - 1)) // 2 + (ant2_idx - ant1_idx - 1)
   return bline_idx

def get_nbls(nants, auto=True):
   '''
   literally just compute baseline count depending on if we count auto-correlations or not
   '''
   if auto:
       nbls= (nants)*(nants+1)/2
   else:
       nbls=(nants)*(nants-1)/2
   return int(nbls)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default=".",
        help="Output plot directory [default: .]",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )
    ant_names = []
    ant_coords = []
    dir_parents = []
    spec_offsets = []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ref_ant, ant, details)
        ant_names.append(details["name"])
        ant_coords.append(details["coordinates"])
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])

    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = config["correlation"]["osamp"]
    pfb_size = config["correlation"]["pfb_size"]
    new_acclen = config["correlation"]["new_acclen"]
    cutsize = 16
    print("pfbsize",pfb_size)
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size))
    channels = np.arange(chanstart, chanend)
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)
    print("nchunks", nchunks)
    print("loaded files", files)
    print("IPFB ROWS", pfb_size, "OSAMP", osamp)
    filt_thresh = 0.2
    # t_acclen = acclen*4096/250e6
    # sys.exit()
    if osamp > 1:
        t1=time.time()
        pols,new_channels=helper.repfb_xcorr_avg(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,cutsize=16,filt_thresh=filt_thresh)
        t2=time.time()
    else:
        t1=time.time()
        pols,new_channels=helper.xcorr_avg(idxs,files,pfb_size,nchunks,channels)
        t2=time.time()
    print("Total time taken", t2-t1)



    #initialize UVData object
    uv = UVData()

    #set antenna metadata
    nants = len(dir_parents)
    uv.Nants_data, uv.Nants_telescope = nants, nants
    uv.antenna_numbers = np.arange(nants)
    uv.antenna_names = ant_names
    uv.antenna_positions = np.array([
        EarthLocation(lat=lat, lon=lon, height=height).geocentric.xyz.value
        for lat, lon, height in ant_coords
    ])

    #set telescope location (from lat/lon/height to ECEF)
    tel_lat, tel_lon, tel_height = dir_parents[0]
    tel_location = EarthLocation(lat=tel_lat, lon=tel_lon, height=tel_height)
    uv.telescope_location = tel_location.geocentric.xyz.value

    #set frequency stuff
    uv.Nfreqs = len(new_channels)
    uv.freq_array = np.array([new_channels]) #set to channel indices for now
    uv.channel_width = (250*10**6)/(4096*osamp)  #NEED TO CHECK

    #set polarization stuff
    uv.Npols = 4
    nantpols = nants * uv.Npols
    uv.polarization_array = np.array([-5, -6, -7, -8]) #standard convention for XX, YY, XY, YX

    #times/baselines
    uv.Nbls = get_nbls(uv.Nants_data, auto=True)
    uv.Ntimes = len(pols[0, 0, 0, :])
    uv.Nblts = uv.Ntimes * uv.Nbls
    
    #time stuff
    uv.integration_time = new_acclen*osamp*4096/(250*10**6)  #NEED TO CHECK
    times = Time(init_t + np.arange(uv.Ntimes) * uv.integration_time, format='unix', scale='utc')
    uv.time_array = np.repeat(times.jd, uv.Nbls)
    uv.lst_array = np.repeat(times.sidereal_time('mean', longitude=tel_location.lon).radian, uv.Nbls)
    uv.set_uvws_from_antenna_positions()

    #temporary placeholders
    uv.spw_array = np.array([0])  # just one spectral window for now
    uv.flag_array = np.zeros((uv.Nblts, uv.Nfreqs, uv.Npols), dtype=bool)
    uv.nsample_array = np.ones((uv.Nblts, uv.Nfreqs, uv.Npols), dtype=np.float32)

    #random metadata
    uv.telescope_name = "ALBATROS"
    uv.instrument = "ALBATROS Pipeline (-MA, -TB)"
    uv.history = "in process"
    
    #make pyuv variables
    ant_1_array = np.zeros(uv.Nblts, dtype=int)
    ant_2_array = np.zeros(uv.Nblts, dtype=int)
    vis_data = np.zeros((uv.Nblts, uv.Nfreqs, uv.Npols), dtype=np.complex64)

    populate_time = time.time()
    #populate UVH5 data arrays
    for chan_idx in range(uv.Nfreqs):
        for chunk_idx in range(uv.Ntimes):
            for i in range(nantpols):
                ant1_idx, pol1_idx = get_ant_pol_idxs(i, nants, uv.Npols)
                for j in range(i, nantpols):
                    ant2_idx, pol2_idx = get_ant_pol_idxs(j, nants, uv.Npols)
                    pol_tot_idx = get_poltot_idx(pol1_idx, pol2_idx, uv.Npols)
                    bline_idx = get_bline_idx(ant1_idx, ant2_idx, nants, auto=True)
                    blt_idx = chunk_idx*uv.Nbls + bline_idx
                    ant_1_array[blt_idx] = ant1_idx
                    ant_2_array[blt_idx] = ant2_idx

                    vis_data[blt_idx, chan_idx, pol_tot_idx] = pols[i, j, chan_idx, chunk_idx]

    #set data arrays
    uv.ant_1_array = ant_1_array
    uv.ant_2_array = ant_2_array
    uv.data_array = vis_data

    print(f"time to populate data array: {time.time()- populate_time}")
    print(f"data shape: {uv.data_array}")
    print(f"Nblts: {uv.Nblts}, Nfreqs: {uv.Nfreqs}, Npols: {uv.Npols}")

    #write to uvh5 file
    writing_time = time.time()
    outname = f"xcorr_{init_t}_{new_acclen}_{osamp}_{nchunks}_{chanstart}_{chanend}.uvh5"
    outpath = path.join(args.outdir, outname)
    uv.write_uvh5(outpath, clobber=True)
    print(f"time to write to uvh5: {time.time()-writing_time}")

    #test read the file
    uv_test = UVData()
    read_time = time.time()
    uv_test.read_uvh5(f"{outpath}")
    print(f"Read time: {time.time()-read_time} sec")

    print(uv_test)

    assert uv_test.Nants_data == uv.Nants_data
    assert uv_test.Nfreqs == uv.Nfreqs
    assert uv_test.Nblts == uv.Nblts