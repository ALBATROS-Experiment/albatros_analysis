import sys
import os
from os import path
sys.path.append(os.path.expanduser('~'))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations_gpu as crg
import cupy as cp
from albatros_analysis.src.utils import pfb_utils as pu
import numpy as np
import time
import os
import ctypes


lib_path = os.path.expanduser('~/albatros_analysis/src/correlations/libcgemm_batch.so')
lib = ctypes.CDLL(lib_path)

# 2) Declare the C function signature
lib.cgemm_strided_batched.argtypes = [
    ctypes.c_void_p,  # A.ptr
    ctypes.c_void_p,  # B.ptr
    ctypes.c_void_p,  # C.ptr
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
    ctypes.c_int      # batchCount
]
lib.cgemm_strided_batched.restype = None


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

def repfb_xcorr_avg_old(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize=16,filt_thresh=0.45):
    nant = len(idxs)
    npol = 2

    # ----------------- START IPFB SETUP -----------------------#
    cut=int(pfb_size/cutsize)
    acclen=pfb_size - 2*cut
    ntap=4
    nn=2*2048*osamp
    assert acclen%osamp == 0
    re_pfb_size = acclen//osamp - ntap + 1
    print("re_pfb_size is", re_pfb_size)
    dwin=pu.sinc_hamming(ntap,nn)
    cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')
    matft=pu.get_matft(pfb_size)
    to_ipfb_pol0 = cp.empty((pfb_size,2049),dtype='complex64', order='C') 
    to_ipfb_pol1 = to_ipfb_pol0.copy()
    cut_chunks = cp.zeros((nant, npol, 2*cut, 2049),dtype='complex64', order='C')#need to maintain last chunks for each antenna
    print("cupy_win_big size", np.prod(cupy_win_big.shape)*4/1024**3, "GB")
    print("matft size", np.prod(matft.shape)*8/1024**3, "GB")
    print("2x to_ipfb size", np.prod(to_ipfb_pol0.shape)*8*2/1024**3, "GB")
    print("cut_chunks size", np.prod(cut_chunks.shape)*8*2/1024**3, "GB")
    print("acclen", acclen, "pfb_size", pfb_size)
    missing_flag=False
    # ----------------- END IPFB SETUP -------------------------#
    
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0, #fileidx is 0 = start idx is inside the first file
            idxs[i],
            acclen,
            nchunks=nchunks,
            chanstart=chanstart,
            chanend=chanend,
            type='float'
        )
        antenna_objs.append(aa)
    channels=np.asarray(aa.obj.channels,dtype='int64')
    nchan = (aa.obj.chanend - aa.obj.chanstart)*osamp #nchan -> increases due to upchannelization
    repfb_chanstart = channels[aa.obj.chanstart] * osamp
    repfb_chanend = channels[aa.obj.chanend-1] * osamp
    # print("channels are", channels)
    print("start and end chans are", repfb_chanstart, repfb_chanend)
    # sys.exit()
    # print("start and end are", )
    split = 1
    print("nant", nant, "nchunks", nchunks, "nchan", nchan)
    vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
    xin = cp.empty((nant*npol, re_pfb_size, nchan),dtype='complex64',order='F')
    print("xin shape", xin.shape)
    scratch = cp.empty((nant*npol,nant*npol,nchan*split),dtype='complex64',order='F')
    missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='F')
    # missing_mask = cp.empty((nant,nchunks),dtype='float32',order='F') #all integers. float is ok. cuda doesnt like integers.
    print("vis size", np.prod(vis.shape)*8/1024**3, "GB")
    print("scratch size", np.prod(scratch.shape)*8/1024**3, "GB")
    print("xin size", np.prod(xin.shape)*8/1024**3, "GB")

    # rowcounts = np.empty(nchunks, dtype="int64")
    start_specnums = [ant.spec_num_start for ant in antenna_objs] #specnums are always on the host. so numpy is OK.
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    for i, chunks in enumerate(zip(*antenna_objs)):
        if i%10==0: print(i)
        start_event.record()
        for j in range(nant):
            chunk=chunks[j]
            start_specnum = start_specnums[j]
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums']-start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
            # print(start_specnum)
            # print(chunk['pol0'], chunk['pol1'])
            # print(pol0[:,1834:1854],pol1[:,1834:1854])
            # print(chunk['specnums'])
            perc_missing = (1 - len(chunk["specnums"]) / acclen) * 100
            missing_fraction[j, i] = perc_missing
            # CHUNK 0 is going to have all zeros at the top. Ignore first chunk in the saved output.
            to_ipfb_pol0[:2*cut] = cut_chunks[j,0,:,:] # <---- zeros for CHUNK 0
            to_ipfb_pol0[2*cut:] = pol0
            to_ipfb_pol1[:2*cut] = cut_chunks[j,1,:,:] # <---- zeros for CHUNK 0
            to_ipfb_pol1[2*cut:] = pol1
            raw_pol0 = pu.cupy_ipfb(to_ipfb_pol0, matft, thresh=filt_thresh)
            raw_pol1 = pu.cupy_ipfb(to_ipfb_pol1, matft, thresh=filt_thresh)
            # print("IPFB done")
            pol0_new = pu.cupy_pfb(raw_pol0[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4)
            pol1_new = pu.cupy_pfb(raw_pol1[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4)
            # print(pol0_new.shape, "new PFB shape")
            # print("pfb done")
            cut_chunks[j,0,:,:] = pol0[-2*cut:,:]
            cut_chunks[j,1,:,:] = pol1[-2*cut:,:]
            # print("xin shape", xin[j*nant,:,:].shape)
            # print("in shape",pol0_new[cut:-cut, repfb_chanstart : repfb_chanend].shape )
            #TODO: average and save only the channels I want
            xin[j*nant,:,:] = pol0_new[:, repfb_chanstart : repfb_chanend] # BFI data is C-major for IPFB
            xin[j*nant+1,:,:] = pol1_new[:, repfb_chanstart : repfb_chanend] #gotta support arbitrary chans
        # print(xin.shape, xin.flags)
        # print(scratch.shape)
        out=crg.avg_xcorr_all_ant_gpu(xin,nant,npol,re_pfb_size,nchan,split=1,out=scratch)
        # out=cp.empty(scratch.shape,dtype='complex64',order='F')
        # M=nant*npol
        # N=M
        # K=re_pfb_size
        # batchCount=nchan
        # lib.cgemm_strided_batched(
        #     ctypes.c_void_p(xin.data.ptr),
        #     ctypes.c_void_p(xin.data.ptr),
        #     ctypes.c_void_p(scratch.data.ptr),
        #     M, N, K, batchCount
        # )
        # out=1
        end_event.record()
        end_event.synchronize()
        # print("CUDA IPFB+XCORR time ",cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        vis[:,:,:,i]=cp.asnumpy(out) #scratch should still be on the device
    vis = np.ma.masked_invalid(vis)
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend) #TODO: for really large BW/delta-nu, we should probably store only start and end

def repfb_xcorr_avg(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,outfile,lblock=4096, ntap=4, cutsize=16,filt_thresh=0.45):
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
    xcorr = pu.StreamingCorrelator(nant, npol, new_acclen, new_channels, bufsize_frac = 64)
    print(nrows_total)
    #on HOST
    if os.path.exists(outfile):
        os.remove(outfile)
    vis_file = np.memmap(outfile,mode="w+",shape=(nant*npol, nant*npol, new_nchan, nrows_total), dtype="complex64",order="F")
    print("OUTFILE SHAPE", vis_file.shape)
    print("EXPECTED OUTFILE SIZE", np.prod(vis_file.shape)*64/8/1024**3, "GB")
    vis_chunk_size = 64
    vis_file_ptr = 0
    vis = np.zeros((nant*npol, nant*npol, new_nchan, vis_chunk_size), dtype="complex64", order="F")
    rowidx=0
    print(ipfb)
    print(fpfb)
    print(xcorr)
    
    header = bdc.get_header(files[0][0])
    #print(header)
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

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        # start_event.record()
        ts1=time.time()
        for ant_idx in range(nant):
            chunk=chunks[ant_idx]
            start_specnum = start_specnums[ant_idx]
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
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
            xcorr.load(ant_idx, 0, pol0_new)
            xcorr.load(ant_idx, 1, pol1_new)
            # end_event.record()
            # end_event.synchronize()
            # print("load time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            
        # start_event.record()
        rows = xcorr.xcorr()
        # end_event.record()
        # end_event.synchronize()
        # print("xcorr time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        n = len(rows)
        # end_event.record()
        # end_event.synchronize()
        # print("time for one chunk xcorr all ant,pol,freq", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        if n > 0:
        #     end_event.record()
        #     end_event.synchronize()

            # print(f"chunk {chunk_idx}/{nchunks}, rcv {n}")
        #     print("time for one chunk xcorr all ant,pol,freq", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            for row in rows:
                if rowidx == vis_chunk_size:
                    t1=time.time()
                    vis_file[:, : , : , vis_file_ptr : vis_file_ptr + vis_chunk_size] = vis
                    t2=time.time()
                    print("host to disk time", t2-t1)
                    rowidx = 0
                    vis_file_ptr += vis_chunk_size
                t1=time.time()
                vis[:,:,:,rowidx] = cp.asnumpy(row) #dev to host
                t2=time.time()
                # print("dev to host time", t2-t1)
                rowidx+=1
        ts2=time.time()
        print(f"chunk {chunk_idx}/{nchunks}, chunk time {ts2-ts1:5.3f}")
    if rowidx>0:
        vis_file[:, : , : , vis_file_ptr : vis_file_ptr + rowidx] = vis[:,:,:,:rowidx]
    vis_file.flush()
    return vis, new_channels


def repfb_xcorr_avg_tb(idxs,files,pfb_size,nchunks,channels,osamp,new_acclen,outfile,lblock=4096, ntap=4, cutsize=16,filt_thresh=0.45):
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
    xcorr = pu.StreamingCorrelator(nant, npol, new_acclen, new_channels, bufsize_frac = 64)
    print(nrows_total)
    #on HOST
    if os.path.exists(outfile):
        os.remove(outfile)
    vis_file = np.memmap(outfile,mode="w+",shape=(nant*npol, nant*npol, new_nchan, nrows_total), dtype="complex64",order="F")
    print("OUTFILE SHAPE", vis_file.shape)
    print("EXPECTED OUTFILE SIZE", np.prod(vis_file.shape)*64/8/1024**3, "GB")
    vis_chunk_size = 64
    vis_file_ptr = 0
    vis = np.zeros((nant*npol, nant*npol, new_nchan, vis_chunk_size), dtype="complex64", order="F")
    rowidx=0
    print(ipfb)
    print(fpfb)
    print(xcorr)
    
    header = bdc.get_header(files[0][0])
    #print(header)
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

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        # start_event.record()
        ts1=time.time()
        for ant_idx in range(nant):
            chunk=chunks[ant_idx]
            start_specnum = start_specnums[ant_idx]
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
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
            xcorr.load(ant_idx, 0, pol0_new)
            xcorr.load(ant_idx, 1, pol1_new)
            # end_event.record()
            # end_event.synchronize()
            # print("load time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            
        # start_event.record()
        rows = xcorr.xcorr()
        # end_event.record()
        # end_event.synchronize()
        # print("xcorr time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        n = len(rows)
        # end_event.record()
        # end_event.synchronize()
        # print("time for one chunk xcorr all ant,pol,freq", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        if n > 0:
        #     end_event.record()
        #     end_event.synchronize()

            # print(f"chunk {chunk_idx}/{nchunks}, rcv {n}")
        #     print("time for one chunk xcorr all ant,pol,freq", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            for row in rows:
                if rowidx == vis_chunk_size:
                    t1=time.time()
                    vis_file[:, : , : , vis_file_ptr : vis_file_ptr + vis_chunk_size] = vis
                    t2=time.time()
                    print("host to disk time", t2-t1)
                    rowidx = 0
                    vis_file_ptr += vis_chunk_size
                t1=time.time()
                vis[:,:,:,rowidx] = cp.asnumpy(row) #dev to host
                t2=time.time()
                # print("dev to host time", t2-t1)
                rowidx+=1
        ts2=time.time()
        print(f"chunk {chunk_idx}/{nchunks}, chunk time {ts2-ts1:5.3f}")
    if rowidx>0:
        vis_file[:, : , : , vis_file_ptr : vis_file_ptr + rowidx] = vis[:,:,:,:rowidx]
    vis_file.flush()
    return vis, new_channels

