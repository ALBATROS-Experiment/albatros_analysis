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
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.utils import orbcomm_utils_gpu as outils_gpu

def xcorr_avg(idxs,files,pfb_size,nchunks,channels):
    nant = len(idxs)
    npol = 2
    nchan = len(channels)
    read_size = pfb_size
    xcorr = pu.StreamingCorrelator(nant, npol, read_size, np.arange(0, nchan), bufsize_frac = 10) #correlate all input channels

    #on HOST
    vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
    rowidx=0

    print(xcorr)
    
    header = bdc.get_header(files[0][0])
    print(header)
    channel_indices = np.where(np.isin(header['channels'],channels))[0] #channels that are in requested channels
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
    print("channels present", aa.obj.channels)
    print("Channel indices loaded", aa.obj.channel_idxs, "corresponding to", aa.obj.channels[aa.obj.channel_idxs])
    start_specnums = [ant.spec_num_start for ant in antenna_objs]

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        # start_event.record()
        for ant_idx in range(nant):
            chunk=chunks[ant_idx]
            start_specnum = start_specnums[ant_idx]
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
            xcorr.load(ant_idx, 0, pol0)
            xcorr.load(ant_idx, 1, pol1)
        start_event.record()
        rows = xcorr.xcorr()
        end_event.record()
        end_event.synchronize()
        print("xcorr time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        n = len(rows)
        if n > 0:
            print(f"chunk {chunk_idx}/{nchunks}, rcv {n}")
            for row in rows:
                t1=time.time()
                vis[:,:,:,rowidx] = cp.asnumpy(row) #dev to host
                t2=time.time()
                print("dev to host time", t2-t1)
                rowidx+=1
    return vis, channels

def repfb_xcorr_avg(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,outfile,lblock=4096, ntap=4, cutsize=16,filt_thresh=0.45, orig_t=None, delays=None):
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
    nrows_total = nchunks * pfb_size // (osamp * new_acclen)
    xcorr = pu.StreamingCorrelator(nant, npol, new_acclen, new_channels, bufsize_frac = 2)
    nbl = nant * (nant-1) // 2 + nant #number of baselines including auto
    nblt = nbl * nrows_total #total number of baselines times time samples
    print(nrows_total)
    #on HOST
    
    # vis_file = np.memmap(outfile,mode="w+",shape=(nant*npol, nant*npol, new_nchan, nrows_total), dtype="complex64",order="F")
    # vis_file = np.memmap(outfile,mode="w+",shape=(nblt, new_nchan, npol*npol), dtype="complex64",order="F")
    file_size_limit = 500*1024**2 # 500 MB
    vis_chunk_size = int(file_size_limit/(nbl*new_nchan*npol*npol*8))
    vis_file = np.empty(shape=(nbl, vis_chunk_size, new_nchan, npol*npol), dtype="complex64",order="F") #trying out direct writing

    # vis_file = np.empty(shape=(nant*npol, nant*npol,new_nchan,nrows_total), dtype="complex64",order="F") #this is for direct dumping
    # vis_file = np.memmap(outfile, mode="w+", shape=(nbl, nrows_total, new_nchan, npol*npol), dtype="complex64",order="F") #trying out direct writing
    print("OUTFILE SHAPE", vis_file.shape)
    print("EXPECTED OUTFILE SIZE", np.prod(vis_file.shape)*8/1024**3, "GB")
    print("EXPECTED NUM OF FILE CHUNKS", nrows_total//vis_chunk_size + 1)
    vis_chunk_id = 0
    # vis = np.zeros((nant*npol, nant*npol, new_nchan, vis_chunk_size), dtype="complex64", order="F")
    ai_gpu, aj_gpu = cp.triu_indices(nant) # ai_gpu, aj_gpu are 1-D cupy arrays of length nbl
    rowidx=0
    print(ipfb)
    print(fpfb)
    print(xcorr)
    
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
    ant_ptr = np.zeros(nant, dtype=np.int32)
    T_SPECTRA = lblock * osamp / 250e6
    freqs = cp.asarray(outils.chan2freq(new_channels,alias=True,fftlen=4096*osamp),dtype='float64')
    if delays is not None:
        delays = cp.asarray(delays,dtype='float64')
        orig_t = cp.asarray(orig_t,dtype='float64')
    print("Freqs for beamforming:", freqs/1e6)
    print("T_SPECTRA", T_SPECTRA)
    if delays is not None:
        print("delays", delays.shape, delays)
        print("orig_t", orig_t)
    # sys.exit()
    # start_event = cp.cuda.Event()
    # end_event = cp.cuda.Event()
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        # start_event.record()
        ts1=time.time()
        n=0
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
            # print("PFB returned shape", pol0_new.shape, pol1_new.shape)
            # start_event.record()
            if pol0_new is not None and pol1_new is not None:
                print("new pfb shape", pol0_new.shape)
                if delays is not None:
                    interp_t = cp.arange(ant_ptr[ant_idx], ant_ptr[ant_idx]+pol0_new.shape[0])*T_SPECTRA
                    interp_delay = cp.interp(interp_t, orig_t, delays[ant_idx])
                    pol0_new[:, new_channels]=outils_gpu.apply_delay(pol0_new[:, new_channels], interp_delay, freqs, copy=True)
                    pol1_new[:, new_channels]=outils_gpu.apply_delay(pol1_new[:, new_channels], interp_delay, freqs, copy=True)
                    ant_ptr[ant_idx] += pol0_new.shape[0] #these many rows read
                xcorr.load(ant_idx, 0, pol0_new)
                xcorr.load(ant_idx, 1, pol1_new)
            # end_event.record()
            # end_event.synchronize()
            # print("load time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        # start_event.record()
        if xcorr.loaded_num==nant*npol:
            rows = xcorr.xcorr()
            n = len(rows)
            print("all loaded",n)
            # end_event.record()
            # end_event.synchronize()
            # print("xcorr time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        # end_event.record()
        # end_event.synchronize()
        # print("time for one chunk xcorr all ant,pol,freq", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        if n > 0:
            for row in rows:
                if rowidx == vis_chunk_size:
                    #write the part-file to disk
                    fname = outfile + f"_part{vis_chunk_id:05d}"
                    print("writing", fname)
                    th1=time.time()
                    np.save(fname, vis_file)
                    th2=time.time()
                    print("time to write to disk", th2-th1)
                    vis_chunk_id +=1
                    rowidx = 0
                # print("row is", row)
                #reshape row to (nbl, nchan, npol*npol)
                # print("row flags", row.shape, row.flags)
                # row_orig = row.copy()
                row_reshaped = cp.reshape(row, (npol, nant, npol, nant, -1), order='F')
                # print("row reshaped", row_reshaped.shape, row_reshaped.flags)
                # extract only upper triangle including autos
                row_ut = row_reshaped[:, ai_gpu, :, aj_gpu, : ] #shape (nbl, npol, npol, nchan)
                row_ut = row_ut.transpose(0, 3, 1, 2).reshape(nbl,-1, 4) #shape (nbl, nchan, npol*npol)
                # ======= some tests to verify correct indexing =======
                #idx = antidx*npol + polidx
                #baseline idx 2 = ant0ant2, pol00 ant0pol0 idx = 0, ant2pol0 idx = 4
                #baseline idx 7 = ant1ant1, pol11 ant1pol1 idx = 3, ant1pol1 idx = 3
                #baseline idx 8 = ant1ant2, pol01 ant1pol0 idx = 2, ant2pol1 idx = 5
                # assert cp.all(row_ut[2,10:20,0] == row_orig[0,4,10:20] ) #passing
                # assert cp.all(row_ut[7,10:20,3] == row_orig[3,3,10:20] ) #passing
                # assert cp.all(row_ut[8,10:20,1] == row_orig[2,5,10:20] ) #passing (remember pols are also in F ordering post-flattening: 00,01,10,11)
                # print("row_ut shape", row_ut.shape, row_ut.flags)
                # ====================================================
                t1=time.time()
                vis_file[ : , rowidx , : , : ] = cp.asnumpy(row_ut,order='F') #dev to host, ensuring same ordering for transfer speed (marginal gain for O(10) baselines.)
                t2=time.time()
                print("dev to host time", t2-t1) #2 OOM faster than time spent doing IPFB+PFB+XCORR
                # print("row_ut",row_ut[10:12,10:15,2])
                # print("vis_file",vis_file[10:12,rowidx,10:15,2])
                rowidx+=1

        ts2=time.time()
        print(f"chunk {chunk_idx}/{nchunks}, chunk time {ts2-ts1:5.3f}")
    if rowidx>0:
        fname = outfile + f"_part{vis_chunk_id:05d}"
        print("writing", fname)
        np.save(fname, vis_file)
    print("final rowidx=", rowidx)
    print("vis file shape", vis_file.shape)
    # th1=time.time()
    # np.save(outfile, vis_file)
    # th2=time.time()
    # print("time to write to disk", th2-th1)
    return vis_file, new_channels
