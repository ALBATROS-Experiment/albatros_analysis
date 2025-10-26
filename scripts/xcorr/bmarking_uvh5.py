import numpy as np
import time
import os
import h5py
import json
import helper
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pyuvdata import UVData

def log_io_usage(tag, start_time, start_io, end_io):
    duration = time.perf_counter() - start_time
    io_delta = end_io.write_bytes - start_io.write_bytes
    io_MB = io_delta / (1024**2)
    print(f"[{tag}] Duration: {duration:.3f}s, Written: {io_MB:.2f}MB, Rate: {io_MB / duration:.2f} MB/s")

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


def benchmark_uvh5_write(filename,
                         configname,
                         chunk_shape,
                         dtype=np.complex64):
    
    # Remove file if exists
    if os.path.exists(filename):
        os.remove(filename)

    #CONFIG FILE STUFF------------------------
    with open(configname, "r") as f:
        config = json.load(f)
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

    nchans = len(channels)
    chan_width = 60000
    nants = len(ant_names)

    #here is where you would do calculations, and get data.
    new_channels = np.arange(osamp) + channels[:, None] * osamp

    #SET UP UVH5 STUFF
    #----------------------------------------------------------------------
    uv = UVData()

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
    uv.channel_width = (250*10**6)/(4096*osamp)
    uv.freq_array = np.array([new_channels])*uv.channel_width

    #set polarization stuff
    uv.Npols = 4
    nantpols = nants * uv.Npols
    uv.polarization_array = np.array([-5, -6, -7, -8]) #standard convention for XX, YY, XY, YX

    #times/baselines
    uv.Nbls = get_nbls(uv.Nants_data, auto=True)
    uv.Ntimes = nchunks * pfb_size // (osamp * new_acclen)
    uv.Nblts = uv.Ntimes * uv.Nbls
    
    #time stuff
    uv.integration_time = new_acclen*osamp*4096/(250*10**6)  #NEED TO CHECK
    times = Time(init_t + np.arange(uv.Ntimes) * uv.integration_time, format='unix', scale='utc')
    uv.time_array = np.repeat(times.jd, uv.Nbls)
    uv.lst_array = np.repeat(times.sidereal_time('mean', longitude=tel_location.lon).radian, uv.Nbls)
    uv.set_uvws_from_antenna_positions()

    #temporary placeholders
    uv.spw_array = np.array([0])  # just one spectral window for now
    
    #random metadata
    uv.telescope_name = "ALBATROS"
    uv.instrument = "ALBATROS Pipeline (-MA, -TB)"
    uv.history = "in process"
    
    #make pyuv variables
    

    #UVH5 OBJECT FOR STREAM WRITING
    #----------------------------------------------------------------

    chunk_shape = ()

    uv.initialize_uvh5_file(
        filename,
        clobber=True,
        data_compression=None,
        flags_compression=None,
        nsample_compression=None,
        chunks=chunk_shape
    )

    #chunks: h5py.create_dataset chunks keyword. 
    #        Tuple for chunk shape, True for auto-chunking, None for no chunking. 
    #        Default is True.

    # Now do writes of zero data (or random) in chunks
    chunk_time = uv.ant_1_arrayNblts // n_chunks  # e.g., split baseline‐time axis
    dummy = np.zeros((chunk_time, uv.Nfreqs, uv.Npols), dtype=dtype)
    flags = np.zeros((chunk_time, uv.Nfreqs, uv.Npols), dtype=dtype) 
    nsamples = np.zeros((chunk_time, uv.Nfreqs, uv.Npols), dtype=dtype) 

    times = []
    for i in range(n_chunks):
        start = i * chunk_time
        end = start + chunk_time
        t0 = time.perf_counter()
        uv.write_uvh5_part(filename=filename,
                            data_array=dummy,
                            flags_array=flags,
                            nsample_array=nsamples,
                            blt_inds=[i, i+chunk_time])
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"Chunk {i}: {end-start}×{uv.Nfreqs}×{uv.Npols}, time = {times[-1]:.3f}s")


    print(f"Average write time per chunk: {np.mean(times):.3f}s")
    MB_written = dummy.nbytes * n_chunks / (1024**2)
    print(f"Total MB: {MB_written:.1f} MB, Overall speed: {MB_written / np.sum(times):.1f} MB/s")

if __name__ == '__main__':
    benchmark_uvh5_write(
        filename='/scratch/thomasb/test_vis.uvh5',
        config='config2.json'
    )
