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
    filt_thresh = 0.2
    n_achks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size))
    channels = np.arange(chanstart, chanend)
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    
    fname = f"xcorr_all_ant_1bit_{str(init_t)}_{str(end_t)}_{str(new_acclen)}_{str(osamp)}_{str(n_achks)}_{chanstart}_{chanend}.uvh5"
    fpath = path.join(args.outdir,fname)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    print('---------CONFIG INIT DUMP START-------------')
    #BASIC
    print('names', ant_names)
    print('numbers', ant_numbers)
    print('paths', dir_parents)
    print('offsets', spec_offsets)

    print('antenna coordinates:')
    for i in range(len(ant_coords)):
        print(ant_names[i], ant_coords[i])

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
    ant_pos_enu = ENU_from_ECEF(all_ant_ecef, center_loc=tel_loc)
    alb_tel.antenna_positions = ant_pos_enu
    print('antenna enu positions')
    print(ant_pos_enu)

    #sanity check (want to make sure we can recover the correct coordinates)
    ant_ecef_2 = ECEF_from_ENU(ant_pos_enu, center_loc=tel_loc)
    ant_coords_2 = np.array(LatLonAlt_from_XYZ(ant_ecef_2)).T
    ant_coords_2[:,0] *= 180/np.pi
    ant_coords_2[:,1] *= 180/np.pi
    for i in range(len(ant_coords)):
        assert np.max(np.array(ant_coords[i]) - np.array(ant_coords_2[i]))< 1e-8
    

    #FREQUENCY, POLARIZATION STUFF
    new_channels = np.arange(osamp) + channels[:, None] * osamp
    new_channels = new_channels.ravel()
    uv.Nfreqs = len(new_channels)
    channel_width = (250*10**6)/(4096*osamp)
    uv.channel_width = np.ones(uv.Nfreqs) * channel_width
    uv.freq_array = np.array(new_channels)*channel_width
    uv.Npols = 4
    uv.polarization_array = np.array([-5, -7, -6, -8]) #due to how we reform!

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
    uv.phase_center_catalog = {0: {"cat_type": "zenith",
                                   "cat_name": "zenith"}}
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

    print('ant1 array start:')
    print(uv.ant_1_array[:28])
    print('ant2 array start:')
    print(uv.ant_2_array[:28])
    print('baseline array start:')
    print(uv.baseline_array[:28])
    print('--------------END----------')

    #print(alb_tel.antenna_positions)
    #print(uv.uvw_array)

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
    
    # pols,new_channels=helper_gpu.repfb_xcorr_avg_part(idxs,
    #                                              files,
    #                                              pfb_size,
    #                                              n_achks,
    #                                              channels,
    #                                              osamp,
    #                                              new_acclen,
    #                                              fpath,
    #                                              cutsize=16,
    #                                              filt_thresh=filt_thresh)


    t2=time.time()
    print("Total time taken", t2-t1)
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
