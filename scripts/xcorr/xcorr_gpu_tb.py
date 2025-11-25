import numpy as np
import time
import argparse
from os import path
import os
import sys
import helper
from pyuvdata import UVData
from pyuvdata import telescopes
from pyuvdata.utils import ENU_from_ECEF
import helper_gpu
import pyuv_helper as ph
sys.path.insert(0,path.expanduser("~"))
import json
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation
import pathlib


if __name__=="__main__":

    #-----------------------------SET UP FROM CONFIG------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("-o", "--outdir",dest="outdir",type=str,default="/scratch/thomasb")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    ant_names, ant_coords, dir_parents, spec_offsets = [], [], [], []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])
        ant_names.append(details["name"])
        ant_coords.append(details["coordinates"])

    ant_numbers = np.array([int(name.split()[1]) for name in ant_names])
    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = config["correlation"]["osamp"]
    pfb_size = config["correlation"]["pfb_size"]
    new_acclen = config["correlation"]["new_acclen"]
    
    cutsize = 16
    filt_thresh = 0.2
    n_achks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size))
    channels = np.arange(chanstart, chanend)
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    
    fname = f"xcorr_all_ant_1bit_{str(init_t)}_{str(end_t)}_{str(new_acclen)}_{str(osamp)}_{str(n_achks)}_{chanstart}_{chanend}.pyuv"
    fpath = path.join(args.outdir,fname)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    print('---------CONFIG INIT DUMP START-------------')
    #BASIC
    print('names', ant_names)
    print('numbers', ant_numbers)
    print('paths', dir_parents)
    print('coords', ant_coords)
    print('offsets', spec_offsets)

    #PFB STUFF
    print("final idxs", idxs)
    print("nchunks (antenna)", n_achks)
    print("IPFB ROWS (aka pfb_size)", pfb_size)
    print("OSAMP", osamp)
    print('cutsize', cutsize)
    print('filter threshold', filt_thresh)

    #FILE STUFF
    print('file name', fname)
    print('file path', fpath)
    print('-------------------END----------------------')
    #---------------------------PYUV METADATA------------------------------------
    
    #GET LOCATIONS
    print('----------set up locations--------')
    tel_lat, tel_lon, tel_alt = ant_coords[0] #telescope location ant 1 by convention
    tel_loc = EarthLocation(lat=tel_lat*u.deg,lon=tel_lon*u.deg,height=tel_alt*u.m)
    all_ant_ecef = []
    for lat, lon, height in ant_coords:
        xq, yq, zq = EarthLocation(lat=lat*u.deg,
                                   lon=lon*u.deg,
                                   height=height*u.m).to_geocentric()
        all_ant_ecef.append([xq.to_value(u.m),yq.to_value(u.m),zq.to_value(u.m)])
    all_ant_ecef = np.array(all_ant_ecef)

    print('----initialize uvdata object------')
    #BASIC
    uv = UVData()
    nants = len(ant_names)
    uv.Nants_data = nants
    uv.Nbls = int(uv.Nants_data*(uv.Nants_data+1)/2) #assume we do auto also
    uv.Ntimes = n_achks * pfb_size // (osamp * new_acclen) #use number of antenna chunks
    uv.Nblts = uv.Ntimes * uv.Nbls

    #METADATA
    uv.instrument = "ALBATROS Pipeline (-MA, -TB)"
    uv.history = "in process"
    uv.vis_units = 'uncalib'

    #TELESCOPE OBJECT
    alb_tel = telescopes.Telescope()
    alb_tel.instrument = uv.instrument
    alb_tel.name = "ALBATROS"
    alb_tel.Nants = uv.Nants_data
    alb_tel.antenna_names = np.array(ant_names)
    alb_tel.antenna_numbers = ant_numbers
    alb_tel.location = tel_loc
    ant_pos_enu = ENU_from_ECEF(all_ant_ecef, center_loc = tel_loc)
    alb_tel.antenna_positions = ant_pos_enu
    
    #FREQUENCY, POLARIZATION STUFF
    new_channels = np.arange(osamp) + channels[:, None] * osamp
    new_channels = new_channels.ravel()
    uv.Nfreqs = len(new_channels)
    channel_width = (250*10**6)/(4096*osamp)
    uv.channel_width = np.ones(uv.Nfreqs) * channel_width
    uv.freq_array = np.array(new_channels)*channel_width
    uv.Npols = 4
    uv.polarization_array = np.array([-5, -6, -7, -8])

    #BASELINE STUFF
    bl_vectors, bl_idxs, bl_tup, ant1_idxs, ant2_idxs = ph.get_bline_arrays(ant_names, ant_numbers, ant_pos_enu)
    uv.uvw_array = np.tile(bl_vectors, (uv.Ntimes, 1)) #for now, store uvw array as baseline vectors in ENU wrt antenna 1    
    uv.ant_1_array = np.tile(ant1_idxs, uv.Ntimes)   
    uv.ant_2_array = np.tile(ant2_idxs, uv.Ntimes)
    uv.baseline_array = np.tile(bl_idxs, uv.Ntimes)

    #TIME ARRAYS
    integration_time = new_acclen * osamp * 4096 / (250e6)
    uv.integration_time = np.ones(uv.Nblts)*integration_time
    times = Time(init_t + np.arange(uv.Ntimes) * integration_time, format='unix', scale='utc')
    uv.time_array = np.repeat(times.jd, uv.Nbls)  #total shape (Nblts,)
    uv.telescope = alb_tel
    uv.set_lsts_from_time_array(astrometry_library='astropy')

    #MISC PLACEHOLDER
    uv.phase_center_catalog = {
    0: {"cat_type": "drift",
        "cat_name": "phase_center_0",
        "cat_frame": "icrs",
        "cat_lon": 0.0,   
        "cat_lat": 0.0, 
        "cat_epoch": 2000.0}}
    uv.Nphase = 1
    uv.phase_center_id_array = np.zeros(uv.Nblts, dtype=int)
    uv.phase_center_app_ra = np.zeros(uv.Nblts)    
    uv.phase_center_app_dec = np.zeros(uv.Nblts)   
    uv.phase_center_frame_pa = np.zeros(uv.Nblts)
    uv.spw_array = np.array([0])  # just one spectral window for now
    uv.Nspws = len(uv.spw_array)
    uv.flex_spw_id_array = np.zeros(uv.Nfreqs, dtype = int)

    #SET UP CHUNKING
    row_shape = (uv.Nbls, len(new_channels), uv.Npols)
    vchk_nrows = 32  #number of rows per chunk that we save at a time
    vchk_nblts = vchk_nrows * row_shape[0]
    vchk_shape = (vchk_nblts, row_shape[1], row_shape[2])
    total_data_shape = (uv.Nblts, uv.Nfreqs, uv.Npols)
    n_vchks = uv.Nblts/vchk_shape[0]
    n_vchks_int = int(np.ceil(n_vchks))


    print('--------------METADATA DUMP START----------')
    print("Nants:", uv.Nants_data)
    print("Nbls:", uv.Nbls)
    print("Ntimes:", uv.Ntimes)
    print("Nblts:", uv.Nblts)

    print('baseline array shape', uv.baseline_array.shape)
    print('ant1_array shape', uv.ant_1_array.shape)
    print('ant2_array shape', uv.ant_2_array.shape)
    print('freq array shape', uv.freq_array.shape)
    print('chanwidth array shape', uv.channel_width.shape)

    print('integration time array shape', uv.integration_time.shape)
    print('time array shape', uv.time_array.shape)
    print('lst times array shape', uv.lst_array.shape)

    print('total data shape', total_data_shape)
    print('vis chunk_shape', vchk_shape)
    print('actual number of vis chunks', n_vchks)
    print('number of vis chunks we iterate over', n_vchks_int)
    print('nchunks_og', int(np.floor((end_t-init_t)*250e6/4096/pfb_size)))

    print('channels', channels)
    print('osamp', osamp)
    print('new_acclen', new_acclen)
    print('fpath', fpath)
    print('filt_thresh', filt_thresh)

    print(uv.ant_1_array[:28])
    print(uv.ant_2_array[:28])
    print(uv.baseline_array[:28])
    print('--------------END----------')

    #sys.exit()

    print('---------------STARTING COMPUTATION--------------')
    t1=time.time()
    pols,new_channels=helper_gpu.repfb_xcorr_avg_tb(idxs,
                                                    files,
                                                    pfb_size,
                                                    n_achks,
                                                    vchk_shape,
                                                    channels,
                                                    osamp,
                                                    new_acclen,
                                                    fpath,
                                                    uv,
                                                    cutsize=16,
                                                    filt_thresh=filt_thresh)
    t2=time.time()
    print("Total time taken", t2-t1)
