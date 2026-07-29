import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import orbcomm_utils as outils
import cupy as cp
from albatros_analysis.src.utils import pfb_utils as pu
import numpy as np
import time
import os
import datetime, uuid
import argparse
import json
import albatros_analysis.scripts.xcorr.helper as helper


def repfb(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,outfile,lblock=4096, ntap=4, cutsize=16,filt_thresh=0.45, downconvert=True, orig_t=None, delays=None):
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
    downconvert: bool, optional
        Downconvert the passed channels to enable shorter FFT sizes in IPFB and PFB.
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
    # 
    nrows_total = nchunks * pfb_size // (osamp)

    nbl = nant * (nant-1) // 2 + nant #number of baselines including auto
    # On HOST
    baseband = np.empty((nant, npol, nrows_total, new_nchan), dtype='complex64', order='C')
    
    # On Device
    if downconvert:
        ipfb = pu.StreamingIPFB_IQ(nant, npol, channels, nblock=pfb_size, lblock=4096, ntap=4, window='hamming', cut=cutsize)
        fpfb = pu.StreamingPFB(nant, npol,timestream_size = timestream_size, lblock = ipfb.lblock*osamp, dtype='complex64')
        channel_slice = slice(0, new_nchan)
    else:
        ipfb = pu.StreamingIPFB(nant, npol, channels, nblock=pfb_size, lblock=4096, ntap=4, window='hamming', cut=cutsize)
        fpfb = pu.StreamingPFB(nant, npol,timestream_size = timestream_size, lblock = lblock*osamp)
        channel_slice = slice(new_channels[0], new_channels[-1]+1)
    #needs channels you want to cross-correlate in re-PFB'd data
    
    print("Baseband size:", baseband.nbytes/1e9, "GB")

    rowidx=0
    print(ipfb)
    print(fpfb)
    
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

    print("Channel indices loaded", aa.obj.channel_idxs[0], "to", aa.obj.channel_idxs[-1], "corresponding to channels", aa.obj.channels[aa.obj.channel_idxs[0]], "to", aa.obj.channels[aa.obj.channel_idxs[-1]])
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    ant_ptr = np.zeros(nant, dtype=np.int32)
    
    if delays is not None:
        T_SPECTRA = lblock * osamp / 250e6
        delays = cp.asarray(delays,dtype='float64')
        orig_t = cp.asarray(orig_t,dtype='float64')
        freqs = cp.asarray(outils.chan2freq(new_channels,alias=True,fftlen=4096*osamp),dtype='float64')
        print("Freqs for beamforming:", freqs/1e6)
        print("T_SPECTRA", T_SPECTRA)
        print("delays", delays.shape, delays)
        print("orig_t", orig_t)

    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        # start_event.record()
        ts1=time.time()
        n=0
        for ant_idx in range(nant):
            chunk=chunks[ant_idx]
            expected_start_specnum = start_specnums[ant_idx] + (chunk_idx) * read_size
            assert antenna_objs[ant_idx].spec_num_start == start_specnums[ant_idx] + (chunk_idx+1) * read_size
            if len(chunk['specnums']) != read_size:
                print(f"file in antenna {ant_idx}",antenna_objs[ant_idx].file_paths[antenna_objs[ant_idx].fileidx])
                print(f"chunk specnums {chunk['specnums'][0:10]}, start_specnums {start_specnums[ant_idx] + (chunk_idx) * read_size}")
                print(f"for antenna {ant_idx}, len specnums is {len(chunk['specnums'])}")
            pn = [0, 0]
            for pol_idx in (0,1):
                tag = 'pol'+str(pol_idx)
                pol = bdc.make_continuous_gpu(chunk[tag],chunk['specnums']-expected_start_specnum,cp.arange(0,nchan),read_size, nchan)    
                spec = ipfb.ipfb(ant_idx,pol_idx,pol,thresh=filt_thresh)
                pol_new = fpfb.pfb(ant_idx, pol_idx, spec)
                if pol_new is not None:
                    pn[pol_idx] = pol_new.shape[0]
                    # print("Got ant", ant_idx, "chunk", chunk_idx, "n samples", n, "nchans", pol0_new.shape[1])
                    baseband[ant_idx, pol_idx, ant_ptr[ant_idx]:ant_ptr[ant_idx]+pn[pol_idx], :] = cp.asnumpy(pol_new[:, channel_slice]) #only in case of IQ, this is not new_channels
            assert pn[0]==pn[1]
            ant_ptr[ant_idx] += pn[0]
    np.save(outfile,baseband)
    return baseband


    # if you have 1834 : 1854 as orig channels
    # osamp of 64 gives
    # channel_slice of 1834 * 64 : 1854 * 64 "new_channels"
    # After IQ, the band starting 1834 is moved to channel 0.
    # This means, the channels you're interested in are
    # 0 * 64 : 20 * 64
    # OR
    # channel_slice of 0 : new_nchan * 64