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
import datetime, uuid

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
            # print("pfb done"1753286400)
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
    print('READSIZE', read_size)
    print('CUTSIZE', cutsize)
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
    # start_event = cp.cuda.Event()
    # end_event = cp.cuda.Event()
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
            #sys.exit()
        #     end_event.record()
        #     end_event.synchronize()
            # print(f"chunk {chunk_idx}/{nchunks}, rcv {n}")
        #     print("time for one chunk xcorr all ant,pol,freq", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
            # for row in rows:
            #     if rowidx == vis_chunk_size:
            #         t1=time.time()
            #         vis_file[:, : , : , vis_file_ptr : vis_file_ptr + vis_chunk_size] = vis
            #         t2=time.time()
            #         print("host to disk time", t2-t1)
            #         rowidx = 0
            #         vis_file_ptr += vis_chunk_size
            #     t1=time.time()
                # vis_file[:,:,:,rowidx] = cp.asnumpy(row) #dev to host
            #     t2=time.time()
            #     # print("dev to host time", t2-t1)
                # rowidx+=1
            for row in rows:
                total_rows_seen +=1
                print('shape of row', row.shape)
                if rowidx == vis_chunk_size:
                    #write the part-file to disk
                    fname = outfile + f".part{vis_chunk_id:05d}"
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
                # t1=time.time()
                vis_file[ : , rowidx , : , : ] = cp.asnumpy(row_ut,order='F') #dev to host, ensuring same ordering for transfer speed (marginal gain for O(10) baselines.)
                # t2=time.time()
                # print("dev to host time", t2-t1) #2 OOM faster than time spent doing IPFB+PFB+XCORR
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
    # th1=time.time()
    # np.save(outfile, vis_file)
    # th2=time.time()
    # print("time to write to disk", th2-th1)
    return vis_file, new_channels


def repfb_xcorr_avg_tb(idxs,
                       files,
                       pfb_size,
                       nchunks_ant,
                       vchk_shape,
                       channels,
                       osamp,
                       new_acclen,
                       outfile,
                       uv,
                       lblock=4096,
                       ntap=4, 
                       cutsize=16,
                       filt_thresh=0.45):

    #INITIALIZE UVH5 FILE
    if os.path.exists(outfile):
        os.remove(outfile)
    uv.initialize_uvh5_file(outfile,
                            clobber=True,
                            chunks= vchk_shape)
    
    #set up required variables
    nant, npol = len(idxs), 2
    old_nchan = len(channels)
    new_channels = (np.arange(osamp) + channels[:, None] * osamp).ravel()
    new_nchan = len(new_channels)
    read_size = pfb_size - 2*cutsize
    timestream_size = read_size * lblock
    nrows_total = nchunks_ant * pfb_size // (osamp * new_acclen)

    row_new_shape = (uv.Nbls, uv.Nfreqs, uv.Npols)
    vchk_nrows = int(vchk_shape[0]/uv.Nbls)
    vchk = np.zeros(vchk_shape, dtype="complex64", order="F")

    assert uv.Nants_data == nant
    assert new_nchan == uv.Nfreqs

    print('----correlator dump---')
    print('vis chunk shape', vchk_shape)
    print('nblts from uv', uv.Nblts)
    print('nrows total from file', nrows_total)
    print('nrows times nbls:', uv.Nbls*nrows_total)
    print('READSIZE', read_size)
    print('CUTSIZE', cutsize)
    print('TOTAL ROWS', nrows_total)
    print('---------end---------')
    
    
    ipfb = pu.StreamingIPFB(nant, npol, channels, nblock=pfb_size, lblock=4096, ntap=4, window='hamming', cut=cutsize)
    fpfb = pu.StreamingPFB(nant, npol,timestream_size = timestream_size, lblock = lblock*osamp)
    xcorr = pu.StreamingCorrelator(nant, npol, new_acclen, new_channels, bufsize_frac = 64)
    # print(ipfb)
    # print(fpfb)
    # print(xcorr)
    
    header = bdc.get_header(files[0][0])
    print(header)
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
            nchunks=nchunks_ant,
            channels=channel_indices,
            type='float')
        antenna_objs.append(aa)
    print("channels present", aa.obj.channels)
    print("Channel indices loaded", aa.obj.channel_idxs, "corresponding to", aa.obj.channels[aa.obj.channel_idxs])
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]

    flags_chunk = np.zeros(vchk_shape, dtype = bool)
    nsamples_chunk = np.zeros(vchk_shape, dtype = float)
    #bl_idx_map, pol_idx_map = ph.get_bl_pol_maps(nant, npol)
    #bl_idx_map, pol_idx_map = cp.array(bl_idx_map), cp.array(pol_idx_map)
    ai_gpu, aj_gpu = cp.triu_indices(nant)

    tot_ptr = 0  #total pointer: counts how many total rows we've saved overall
    rowidx = 0  #row index: counts how many rows we've added to one vis chunk
    total_rows_seen = 0 #debug to check all rows are seen

    ai_gpu, aj_gpu = cp.triu_indices(nant)

    #ITERATE THROUGH CHUNKS
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        print(f'----------------CHUNK {chunk_idx+1}-------------') #chunk number vs chunk idx
        ts1=time.time()
        #GET DATA FOR EACH ANT
        for ant_idx in range(nant):
            chunk,start_specnum =chunks[ant_idx], start_specnums[ant_idx]
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,cp.arange(0,old_nchan),read_size, old_nchan)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums']-start_specnum,cp.arange(0,old_nchan),read_size, old_nchan)
            #print("continuous pol0 shape", pol0.shape)
            spec0=ipfb.ipfb(ant_idx,0,pol0,thresh=filt_thresh)
            spec1=ipfb.ipfb(ant_idx,1,pol1,thresh=filt_thresh)
            #print("IPFB returned shape", spec0.shape, spec1.shape)
            pol0_new = fpfb.pfb(ant_idx,0, spec0)
            pol1_new = fpfb.pfb(ant_idx,1, spec1)
            #print("PFB returned shape", pol0_new.shape, pol1_new.shape)
            xcorr.load(ant_idx, 0, pol0_new)
            xcorr.load(ant_idx, 1, pol1_new)
        rows = xcorr.xcorr()
        n = len(rows)
        print('NUMBER OF ROWS', n)
        if n > 0:
            print(f"chunk {chunk_idx+1}/{nchunks_ant}, rcv {n}") #chunk index vs chunk number
            #ITERATE THROUGH EACH ROW
            for row in rows:
                print(f'--------starting row number {rowidx+1}------')
                total_rows_seen +=1
                print('current row idx', rowidx)
                print('shape of row', row.shape)
                #IF VIS CHUNK FULL, DUMP TO UVH5
                if rowidx == vchk_nrows:
                    print(f'vis chunk is full! row {rowidx} will be saved to a new chunk')
                    print('------------SAVING VIS CHUNK TO DISK------------------------------------')
                    twrt=time.time()
                    blt_inds = np.arange(tot_ptr*uv.Nbls, (tot_ptr+vchk_nrows)*uv.Nbls)
                    print('blt_inds', blt_inds)

                    #============================
                    # uv_disk = UVData()
                    # uv_disk.read_uvh5(outfile, read_data=False)
                    # ph.compare_metadata(uv, uv_disk)
                    #============================

                    uv.write_uvh5_part(filename=outfile,
                                       data_array=vchk,
                                       flag_array=flags_chunk,
                                       nsample_array=nsamples_chunk,
                                       blt_inds=blt_inds,
                                       #bls = bl_tup,
                                       check_header = False
                                       )
                    print("host to disk time", time.time()-twrt)
                    print('done saving data. resuming row.')
                    vchk[:] = 0
                    rowidx = 0
                    tot_ptr += vchk_nrows
                #REFORM EACH ROW ANTPOL -> NBLT
                print('START REFORMING:')
                row_nantpol, _, row_nchans = row.shape
                treform = time.time()
                assert row_nchans == uv.Nfreqs
                assert row_nantpol == nant*npol
                row_reshaped = cp.reshape(row, (npol, nant, npol, nant, -1), order='F')
                row_ut = row_reshaped[:, ai_gpu, :, aj_gpu, : ] #shape (nbl, npol, npol, nchan)
                row_ut = row_ut.transpose(0, 3, 1, 2).reshape(uv.Nbls,-1, 4) #shape (nbl, nchan, npol*npol)
                assert row_ut.shape == row_new_shape
                print(row_ut)
                print('DONE REFORMING, time:', time.time() - treform)
                
                #WRITE EACH ROW INTO VIS CHUNK
                t1=time.time()
                vchk[rowidx*uv.Nbls:(rowidx+1)*uv.Nbls,:,:] = cp.asnumpy(row_ut) #into memory
                t2=time.time()
                print("dev to host time", t2-t1)
                print(f'done with row {rowidx}')
                rowidx+=1
        ts2=time.time()
        print(f"chunk {chunk_idx}/{nchunks_ant}, chunk time {ts2-ts1:5.3f}")
    
    #WRITE THE REST OF VIS CHUNK INTO UVH5
    if rowidx>0:
        print('total pointer', tot_ptr)
        print('rowidx', rowidx)
        print('total rows seen', total_rows_seen)
        print('total pointer + rowidx', tot_ptr+rowidx)
        print('nrows total', nrows_total)
        #assert (tot_ptr+rowidx) == nrows_total
        blt_inds = np.arange(tot_ptr*uv.Nbls, (tot_ptr+rowidx)*uv.Nbls)
        print(blt_inds)
        cut_vchk = vchk[:rowidx*uv.Nbls, :, :]
        print('cut chunk shape', cut_vchk.shape)
        print('final blt index',blt_inds[-1])
        print('nblts', uv.Nblts)
        cut_flags_chunk = np.zeros(cut_vchk.shape, dtype = bool)
        cut_nsamples_chunk = np.zeros(cut_vchk.shape, dtype = float)
        uv.write_uvh5_part(filename=outfile,
                           data_array=cut_vchk,
                           flag_array=cut_flags_chunk,
                           nsample_array=cut_nsamples_chunk,
                           blt_inds=blt_inds,
                           #bls = bl_tup,
                           check_header = False
                           )
    return vchk, new_channels

