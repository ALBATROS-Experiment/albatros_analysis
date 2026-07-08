import numpy as np
import cupy as cp
import time
import os
import json
from astropy.coordinates import EarthLocation
from astropy import units as u
from pyuvdata.utils import ENU_from_ECEF
from astropy.time import Time
from pyuvdata import UVData
from pyuvdata import telescopes

def log_io_usage(tag, start_time, start_io, end_io):
    duration = time.perf_counter() - start_time
    io_delta = end_io.write_bytes - start_io.write_bytes
    io_MB = io_delta / (1024**2)
    print(f"[{tag}] Duration: {duration:.3f}s, Written: {io_MB:.2f}MB, Rate: {io_MB / duration:.2f} MB/s")


def get_bline_arrays(ant_names, ant_nums, ant_enus):
    nants = len(ant_names)
    bl_vectors, bl_idxs, bl_tup, ant1_idxs, ant2_idxs = [], [], [], [], []
    counter = 0
    for i_idx, i in enumerate(ant_nums):
        for j_idx, j in enumerate(ant_nums[i_idx:], start=i_idx):
            # print(f'---------New Baseline---------')
            # print('first, second ant:', ant_names[i_idx], ant_names[j_idx])
            # print('first, second #', i, j)
            # print('first, second idxs', i_idx, j_idx)
            #print(ant_enus[i_idx])
            #print(ant_enus[j_idx])
            v = ant_enus[j_idx] - ant_enus[i_idx] #by convention ant2-ant1
            #print('vector:', v)
            bl_vectors.append(v)
            bl_idx = (2048 * i) + j + 2**16
            bl_idxs.append(bl_idx)
            #print('baseline index', bl_idx )
            ant1_idxs.append(i)
            ant2_idxs.append(j)
            bl_tup.append((i,j))
            counter += 1
    assert counter == (nants*(nants+1))/2

    bl_vectors, bl_idxs = np.array(bl_vectors), np.array(bl_idxs, dtype = int)
    ant1_idxs, ant2_idxs = np.array(ant1_idxs), np.array(ant2_idxs)

    return bl_vectors, bl_idxs, bl_tup, ant1_idxs, ant2_idxs


def antpol_to_bl(antpol1_idx, antpol2_idx, nants, npols):

    #get ant, pol idxs
    ant1_idx, ant2_idx = antpol1_idx//npols, antpol2_idx//npols
    pol1_idx, pol2_idx = antpol1_idx%npols, antpol2_idx%npols

    #get overall pol idx (convention XX, YY, XY, YX)
    if pol1_idx == pol2_idx:
        poltot_idx = pol1_idx
    else:
        poltot_idx = 2+pol1_idx

    #get baseline indexes
    if ant1_idx > ant2_idx:
        print('BEWARE: passing antenna in wrong order. flipping! only a problem if this is not for reordering rows')
        ant1_idx, ant2_idx = ant2_idx, ant1_idx
    # if not auto and ant1_idx == ant2_idx:
    #     raise ValueError("turned off auto-correlation but recieved ant1 == ant2")
    bl_idx = ant1_idx * nants - (ant1_idx * (ant1_idx - 1)) // 2 + (ant2_idx - ant1_idx)
    return bl_idx, poltot_idx


def get_bl_pol_maps(nant, npol):
    nantpols = nant*npol
    #sk = 1
    bl_idx_map = np.zeros((nantpols, nantpols), dtype = int)
    pol_idx_map = np.zeros((nantpols, nantpols), dtype = int)
    for i in range(nantpols):
        #sk ^= 1
        #for j in range(i-sk, nantpols):
        for j in range(nantpols):
            bl_idx, pol_tot_idx = antpol_to_bl(i, j, nant, npol)
            bl_idx_map[i, j] = bl_idx
            pol_idx_map[i, j] = pol_tot_idx

    return bl_idx_map, pol_idx_map


def compare_metadata(uv1, uv2):
    attributes = ['Nants_data', 'Nbls', 'Nblts', 'Nfreqs', 'Nphase', 'Npols',
                         'Nspws', 'Ntimes', 'ant_1_array', 'ant_2_array', 'baseline_array',
                         'channel_width', 'data_array', 'flag_array', 'flex_spw_id_array',
                         'freq_array', 'freq_array', 'integration_time', 'lst_array',
                         'nsample_array', 'phase_center_app_dec', 'phase_center_app_ra',
                         'phase_center_catalog', 'phase_center_frame_pa', 'phase_center_id_array',
                         'polarization_array', 'spw_array', 'time_array', 'uvw_array', 'vis_units']
    for attr in attributes:
        print(f'\nCOMPARING {attr}')
        attr1 = getattr(uv1, attr)
        attr2 = getattr(uv2, attr)
        if isinstance(attr1, int) or isinstance(attr1, float): 
            assert attr1 == attr2
            print(type(attr1))
        elif isinstance(attr1, str):
            assert attr1 == attr2
            print(type(attr1))
        elif isinstance(attr1, dict):
            assert attr1 == attr2
            print(type(attr1))
        elif isinstance(attr1, np.ndarray):
            assert np.array_equal(attr1, attr2)
            print(type(attr1))
        elif attr1 is None:
            assert np.array_equal(attr1, attr2)
            print(type(attr1))
        else:
            print('couldnt figure out the type')
            print(type(attr1))
            print(attr1)
            print(type(attr2))
            print(attr2)




def benchmark_uvh5_write(filename, configname):
    
    # remove file if exists
    if os.path.exists(filename):
        os.remove(filename)

    #CONFIG FILE STUFF------------------------
    with open(configname, "r") as f:
        config = json.load(f)
    ant_names, ant_coords, dir_parents, spec_offsets = [], [], [], []
    for i, (ant, details) in enumerate(config["antennas"].items()):
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
    print('new_acclen', new_acclen)
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/pfb_size))
    channels = np.arange(chanstart, chanend)
    #print("nchunks", nchunks)
    #print("loaded files", files)
    #print("IPFB ROWS", pfb_size, "OSAMP", osamp)
    #filt_thresh = 0.2
    readsize = pfb_size - (2*cutsize)
    nchans = len(channels)
    chan_width = 1000

    #GET NEW CHANNELS
    new_channels = np.arange(osamp) + channels[:, None] * osamp
    new_channels = new_channels.ravel()
    print('new channels', new_channels)
    
    #GET LOCATIONS
    tel_lat, tel_lon, tel_alt = ant_coords[0]
    tel_loc = EarthLocation(lat=tel_lat * u.deg,
                            lon=tel_lon * u.deg,
                            height=tel_alt * u.m)

    all_ant_ecef = []
    for lat, lon, height in ant_coords:
        xq, yq, zq = EarthLocation(lat=lat * u.deg,
                                   lon=lon * u.deg,
                                   height=height * u.m).to_geocentric()
        all_ant_ecef.append([xq.to_value(u.m),
                             yq.to_value(u.m),
                             zq.to_value(u.m)])
    all_ant_ecef = np.array(all_ant_ecef)


    #SET UP UVH5 STUFF
    #----------------------------------------------------------------------
    uv = UVData()
    
    #set antenna data and positions---------------
    nants = len(ant_names)
    uv.instrument = "ALBATROS Pipeline (-MA, -TB)"
    uv.history = "in process"
    uv.vis_units = 'uncalib'

    #initialize telescope and descriptive metadata-------
    alb_tel = telescopes.Telescope()
    alb_tel.instrument = uv.instrument

    #uv.telescope_name = "ALBATROS"
    #alb_tel.name = uv.telescope_name
    alb_tel.name = "ALBATROS"

    uv.Nants_data = nants
    alb_tel.Nants = uv.Nants_data

    #uv.antenna_names = ant_names
    #alb_tel.antenna_names = uv.antenna_names
    alb_tel.antenna_names = np.array(ant_names)

    #uv.antenna_numbers = np.arange(nants)
    #alb_tel.antenna_numbers = uv.antenna_numbers
    antenna_numbers = np.array([1, 2, 4, 5, 6, 7, 8])
    alb_tel.antenna_numbers = antenna_numbers
    print(alb_tel.antenna_numbers)

    antenna_positions_enu = ENU_from_ECEF(all_ant_ecef, center_loc = tel_loc)
    #uv.telescope_location = tel_loc
    alb_tel.location = tel_loc
    #uv.antenna_positions = antenna_positions_enu
    #alb_tel.antenna_positions = uv.antenna_positions
    alb_tel.antenna_positions = antenna_positions_enu

    #set frequency stuff--------------------
    uv.Nfreqs = len(new_channels)
    channel_width = (250*10**6)/(4096*osamp)
    uv.channel_width = np.ones(uv.Nfreqs) * channel_width
    uv.freq_array = np.array(new_channels)*channel_width

    #set polarization stuff, standard convention for XX, YY, XY, YX
    uv.Npols = 4
    uv.polarization_array = np.array([-5, -6, -7, -8])

    #times/baselines
    uv.Nbls = uv.Nants_data * (uv.Nants_data +1)/2
    uv.Ntimes = nchunks * pfb_size // (osamp * new_acclen)
    uv.Nblts = uv.Ntimes * uv.Nbls

    #phase center stuff
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

    #time stuff
    integration_time = new_acclen * osamp * 4096 / (250e6)
    uv.integration_time = np.ones(uv.Nblts)*integration_time
    times = Time(init_t + np.arange(uv.Ntimes) * integration_time, format='unix', scale='utc')
    uv.time_array = np.repeat(times.jd, uv.Nbls)  #total shape (Nblts,)
    uv.telescope = alb_tel
    uv.set_lsts_from_time_array(astrometry_library='astropy')

    #baseline stuff
    bl_vectors, bl_idxs, bl_tup, ant1_idxs, ant2_idxs = [], [], [], [], []
    for i_idx, i in enumerate(antenna_numbers):
        for j_idx, j in enumerate(antenna_numbers[i_idx:], start=i_idx):
            print('first, second ant:', ant_names[i_idx], ant_names[j_idx])
            print('first, second #', i, j)
            # print(antenna_positions_enu[i])
            # print(antenna_positions_enu[j])
            bl_vectors.append(antenna_positions_enu[j_idx] - antenna_positions_enu[i_idx])
            bl_idx = (2048 * i) + j + 2**16
            bl_idxs.append(bl_idx)
            print('baseline index', bl_idx )
            ant1_idxs.append(i)
            ant2_idxs.append(j)
            bl_tup.append((i,j))

    bl_vectors, bl_idxs = np.array(bl_vectors), np.array(bl_idxs, dtype = int)
    assert len(bl_idxs) == uv.Nbls
    ant1_idxs, ant2_idxs = np.array(ant1_idxs), np.array(ant2_idxs)

    #set up ant1 ant2 baseline arrays
    uv.uvw_array = np.tile(bl_vectors, (uv.Ntimes, 1))         
    uv.ant_1_array = np.tile(ant1_idxs, uv.Ntimes)   
    uv.ant_2_array = np.tile(ant2_idxs, uv.Ntimes)
    uv.baseline_array = np.tile(bl_idxs, uv.Ntimes)

    #temporary placeholders
    uv.spw_array = np.array([0])  # just one spectral window for now
    uv.Nspws = len(uv.spw_array)
    uv.flex_spw_id_array = np.zeros(uv.Nfreqs, dtype = int)

    uv.telescope = alb_tel


    #UVH5 OBJECT FOR STREAM WRITING
    #----------------------------------------------------------------

    row_shape = (28, 1280, 4)
    vis_chunk_size = 32
    chunk_nblts = vis_chunk_size * row_shape[0]
    chunk_shape = (chunk_nblts, row_shape[1], row_shape[2])
    total_data_shape = (uv.Nblts, uv.Nfreqs, uv.Npols)
    nchunks = uv.Nblts/chunk_shape[0]
    nchunks_int = int(np.ceil(nchunks))
    
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
    print('chunk_shape', chunk_shape)
    print('actual number of chunks', nchunks)
    print('number of chunks we iterate over', nchunks_int)

    print(uv.ant_1_array[:28])
    print(uv.ant_2_array[:28])
    print(uv.baseline_array[:28])
    print('--------------END----------')


    uv.initialize_uvh5_file(filename,
                            clobber=True,
                            data_compression=None,
                            flags_compression=None,
                            nsample_compression=None,
                            chunks=chunk_shape
                            )
    

    times_generate = []
    times_write = []
    for i in range(nchunks_int):
        print(f'--------CHUNK {i}--------')
        blt_start, blt_stop = i * chunk_nblts, min((i+1) * chunk_nblts, uv.Nblts) 
        blt_inds = np.arange(blt_start, blt_stop)
        print('blt start, stop:', blt_start, blt_stop)

        chunk_shape_actual = (blt_stop - blt_start, uv.Nfreqs, uv.Npols)
        flags_chunk = np.zeros(chunk_shape_actual, dtype=bool)
        nsamples_chunk = np.ones(chunk_shape_actual, dtype=float)
        ts_g = time.time()
        real = np.random.normal(size=chunk_shape_actual).astype(np.float32)
        imag = np.random.normal(size=chunk_shape_actual).astype(np.float32)
        data_chunk = (real + 1j * imag).astype(np.complex64)
        t_g = time.time() - ts_g
        times_generate.append(t_g)
        print('time taken to generate data', t_g)
        print('using chunk shape', data_chunk.shape)
        print('flag shape', flags_chunk.shape)
        print('nsamples shape', nsamples_chunk.shape)

        ts_w = time.time()
        uv.write_uvh5_part(filename=filename,
                           data_array=data_chunk,
                           flag_array=flags_chunk,
                           nsample_array=nsamples_chunk,
                           blt_inds=blt_inds,
                           bls = bl_tup,
                           check_header = False
                           )
        t_w = time.time() - ts_w
        times_write.append(t_w)
        print(f"time two write chunk to file", t_w)
        

    print('average generate time', np.mean(times_generate))
    print('average write time', np.mean(times_write))
    #MB_written = data.nbytes * nchunks / (1024**2)
    #print(f"Total MB: {MB_written:.1f} MB, Overall speed: {MB_written / np.sum(times):.1f} MB/s")