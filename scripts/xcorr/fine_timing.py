import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import cupy as cp
from albatros_analysis.src.utils import pfb_utils as pu
import numpy as np
import time
import os
import datetime, uuid
import argparse
import json
import albatros_analysis.scripts.xcorr.helper as helper

def dump_upchan_baseband(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,outfile,lblock=4096, ntap=4, cutsize=16,filt_thresh=0.45):
    """Re-PFB baseband spectra for all antennas x polarizations and x-corr all frequencies

    Parameters
    ----------
    idxs : _type_
        _description_
    files : _type_
        _description_
    pfb_size : _type_
        _description_
    nchunks : _type_
        _description_
    channels : _type_
        Channel numbers to feed IPFB [0,2048), should be present in baseband file.
    osamp : _type_
        Up-resolution factor. 64 means 64x times longer PFBs and 64x higher frequency resolution: 61 kHz/64 ~ 1 kHz.
    lblock : int, optional
        Length of a one "original" PFB tap, by default 4096
    ntap : int, optional
        Number of PFB taps (for both inverse and forward PFBs), by default 4
    cutsize : int, optional
        Number of spectra to snip after IPFB to avoid, by default 16.
        Number of samples snipped from the reconstructed timestream = cutsize*lblock.
        IPFB algorithm forces circularity, causing the edges of recons. timestream to be bad.
    filt_thresh : float, optional
        IPFB Wiener filter threshold, by default 0.45
    """
    nant = len(idxs)
    npol = 2

    read_size = pfb_size - 2*cutsize
    timestream_size = read_size * lblock
    nchan = len(channels)
    new_channels = np.arange(osamp) + channels[:, None] * osamp
    new_channels = new_channels.ravel()
    new_nchan = len(new_channels)
    #needs channels that are in the read data
    ipfb = pu.StreamingIPFB(nant, npol, channels, nblock=pfb_size, lblock=4096, ntap=4, window='hamming', cut=cutsize)
    fpfb = pu.StreamingPFB(nant, npol,timestream_size = timestream_size, lblock = lblock*osamp)
    #needs channels you want to cross-correlate in re-PFB'd data
    nrows_total = nchunks * pfb_size // osamp
    baseband = np.empty((nant, npol, nrows_total, new_nchan), dtype='complex64', order='C')
    print("Baseband size:", baseband.nbytes/1e9, "GB")
    header = bdc.get_header(files[0][0])
    #print(header)
    bit_mode = header['bit_mode']
    channel_indices = np.where(np.isin(header['channels'],channels))[0] #channels that are in requested channels
    assert channel_indices[0]%2==0
    assert len(channel_indices)%2 ==0
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0, #fileidx is 0 = start idx is inside the first file
            idxs[i],
            read_size,
            nchunks=nchunks,
            channels=channel_indices,
            type='float'
        )
        antenna_objs.append(aa)
    #print("channels present", aa.obj.channels)
    print("Channel indices loaded", aa.obj.channel_idxs, "corresponding to", aa.obj.channels[aa.obj.channel_idxs])
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    ant_specnums = [ant.spec_num_start for ant in antenna_objs]
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    ant_ptr = np.zeros(nant, dtype=np.int32)
    T_SPECTRA = lblock * osamp / 250e6
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        # start_event.record()
        ts1=time.time()
        for ant_idx in range(nant):
            chunk=chunks[ant_idx]
            expected_start_specnum = start_specnums[ant_idx] + (chunk_idx) * read_size
            # print(f"Ant {ant_idx} specnum @ {antenna_objs[ant_idx].spec_num_start}; should be @ {start_specnums[ant_idx] + (chunk_idx+1) * read_size}") #spec_num start has already been incremented since a block was read
            assert antenna_objs[ant_idx].spec_num_start == start_specnums[ant_idx] + (chunk_idx+1) * read_size
            assert chunk['specnums'][0] == start_specnums[ant_idx] + (chunk_idx) * read_size
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-expected_start_specnum,cp.arange(0,nchan),read_size, nchan)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums']-expected_start_specnum,cp.arange(0,nchan),read_size, nchan)
            # print("continuous pol0 shape", pol0.shape)
            # start_event.record()
            spec0=ipfb.ipfb(ant_idx,0,pol0,thresh=filt_thresh)
            spec1=ipfb.ipfb(ant_idx,1,pol1,thresh=filt_thresh)
            # end_event.record()
            # end_event.synchronize()
            # print("tot ipfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            # start_event.record()
            pol0_new = fpfb.pfb(ant_idx,0, spec0)
            pol1_new = fpfb.pfb(ant_idx,1, spec1)
            # end_event.record()
            # end_event.synchronize()
            # print("tot pfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            n = pol0_new.shape[0]
            # print("Got ant", ant_idx, "chunk", chunk_idx, "n samples", n, "nchans", pol0_new.shape[1])
            baseband[ant_idx, 0, ant_ptr[ant_idx]:ant_ptr[ant_idx]+n, :] = cp.asnumpy(pol0_new[:, new_channels])
            baseband[ant_idx, 1, ant_ptr[ant_idx]:ant_ptr[ant_idx]+n, :] = cp.asnumpy(pol1_new[:, new_channels])
            ant_ptr[ant_idx] += n
        ts2=time.time()
        print(f"chunk {chunk_idx}/{nchunks}, chunk time {ts2-ts1:5.3f}")
    print("final ant idxs", ant_ptr)
    np.save(outfile,baseband)
    return baseband, new_channels

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
    dir_parents = []
    spec_offsets = []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ref_ant, ant, details)
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
    print("init t", init_t, "end_t", end_t)
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)
    print("nchunks", nchunks)
    # print("loaded files", files)
    print("IPFB ROWS", pfb_size, "OSAMP", osamp)
    filt_thresh = 0.2
    # t_acclen = acclen*4096/250e6
    # sys.exit()
    nant = len(dir_parents)
    npol = 2
    nrows_total = nchunks * pfb_size // (osamp * new_acclen)
    tag = 'regular'
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    # uid = str(uuid.uuid4())[:4]  # short unique suffix
    bit_mode = 1
    fname = (
        f"raw_{init_t}:{end_t}_bit={bit_mode}_ant={nant}_pol={npol}_cha={chanstart}:{chanend}_tim={nrows_total}_"
        f"upx={osamp}_acc={new_acclen}_ipfb={filt_thresh}_"
        f"{'complex64'}_{tag}_{timestamp}"
    )
    print(fname)
    data_dir = os.path.join(args.outdir, f'raw_ant={nant}_pol={npol}_cha={chanstart}:{chanend}_{timestamp}') 
    os.makedirs(data_dir, exist_ok=True)
    outfile = os.path.join(data_dir, fname)
    t1=time.time()
    pols,new_channels=dump_upchan_baseband(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,outfile,cutsize=16,filt_thresh=filt_thresh)
    t2=time.time()
    print("Total time taken", t2-t1)