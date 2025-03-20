import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
import cupy as cp
from albatros_analysis.src.utils import pfb_utils as pu
import numpy as np

def xc_avg(idxs,files,acclen,nchunks,chanstart,chanend):
    nant = len(idxs)
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
        )
        antenna_objs.append(aa)
    print(antenna_objs)
    nchan = aa.obj.chanend - aa.obj.chanstart
    npol = 2
    split = 2
    print("nant", nant, "nchunks", nchunks, "nchan", nchan)
    vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
    xin = cp.empty((nant*npol, acclen, nchan),dtype='complex64',order='F')
    scratch = cp.empty((nant*npol,nant*npol,nchan*split),dtype='complex64',order='F')
    rowcounts = np.empty(nchunks, dtype="int64")
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    for i, chunks in enumerate(zip(*antenna_objs)):
        for j in range(nant):
            xin[j*nant,:,:] = chunks[j].pol0 # BFI data is C-major for IPFB
            xin[j*nant+1,:,:] = chunks[j].pol1
        vis[:,:,:,i]=cr.avg_xcorr_all_ant_gpu(xin,nant,npol,acclen,nchan,split=split,scratch=scratch)  
    vis = np.ma.masked_invalid(vis)
    return vis, rowcounts, aa.obj.channels

def repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize=16,filt_thresh=0.45):
    nant = len(idxs)
    npol = 2

    # ----------------- START IPFB SETUP -----------------------#
    pfb_size = pfb_size * osamp
    cut=int(pfb_size/16)
    acclen=pfb_size - 2*cut
    ntap=4
    nn=2*2048*osamp
    dwin=pu.sinc_hamming(ntap,nn)
    cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')
    matft=pu.get_matft(pfb_size)
    to_ipfb_pol0 = cp.empty((pfb_size,chanend-chanstart),dtype='complex64', order='C') 
    to_ipfb_pol1 = to_ipfb_pol0.copy()
    cut_chunks = cp.zeros((nant, npol, 2*cut, chanend-chanstart),dtype='complex64', order='C')#need to maintain last chunks for each antenna
    print("cupy_win_big size", np.prod(cupy_win_big.shape)*4/1024**3, "GB")
    print("matft size", np.prod(matft.shape)*8/1024**3, "GB")
    print("2x to_ipfb size", np.prod(to_ipfb_pol0.shape)*8*2/1024**3, "GB")
    print("cut_chunks size", np.prod(cut_chunks.shape)*8*2/1024**3, "GB")
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
        )
        antenna_objs.append(aa)
    channels=np.asarray(aa.obj.channels,dtype='int64')
    nchan = (aa.obj.chanend - aa.obj.chanstart)*osamp #nchan -> increases due to upchannelization
    
    split = 2
    print("nant", nant, "nchunks", nchunks, "nchan", nchan)
    vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
    xin = cp.empty((nant*npol, acclen, nchan),dtype='complex64',order='F')
    scratch = cp.empty((nant*npol,nant*npol,nchan*split),dtype='complex64',order='F')
    # missing_mask = cp.empty((nant,nchunks),dtype='float32',order='F') #all integers. float is ok. cuda doesnt like integers.
    missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='C')
    print("vis size", np.prod(vis.shape)*8/1024**3, "GB")
    print("scratch size", np.prod(scratch.shape)*8/1024**3, "GB")
    print("xin size", np.prod(xin.shape)*8/1024**3, "GB")
    # rowcounts = np.empty(nchunks, dtype="int64")
    start_specnums = [ant.spec_num_start for ant in antenna_objs] #specnums are always on the host. so numpy is OK.
    for i, chunks in enumerate(zip(*antenna_objs)):
        for j in range(nant):
            chunk=chunks[i]
            start_specnum = start_specnums[j]
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,channels[ant1.obj.channel_idxs],acclen,nchans=2049)
            pol1=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,channels[ant2.obj.channel_idxs],acclen,nchans=2049)
            perc_missing = (1 - len(chunk["specnums"]) / acclen) * 100
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
            # print("pfb done")
            cut_chunks[j,0,:,:] = pol0[-2*cut:,:]
            cut_chunks[j,1,:,:] = pol1[-2*cut:,:]
            xin[j*nant,:,:] = pol0_new # BFI data is C-major for IPFB
            xin[j*nant+1,:,:] = pol1_new
        vis[:,:,:,i]=cp.asnumpy(cr.avg_xcorr_all_ant_gpu(xin,nant,npol,acclen,nchan,split=split,scratch=scratch)) #scratch should still be on the device
    vis = np.ma.masked_invalid(vis)
    return vis