import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
import cupy as cp

def get_avg_fast_gpu(idxs,files,acclen,nchunks,chanstart,chanend):
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
    xin = cp.empty((nant*npol, acclen, chanend-chanstart),dtype='complex64',order='F')
    out = cp.empty((nant*npol,nant*npol,nchan*split),dtype='complex64',order='F')
    rowcounts = np.empty(nchunks, dtype="int64")
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    st = time.time()
    for i, chunks in enumerate(zip(*antenna_objs)):
        for j in range(nant):
            xin[j*nant,:,:] = chunks[j].pol0 # BFI data is C-major for IPFB
            xin[j*nant+1,:,:] = chunks[j].pol1
        cr.avg_xcorr_all_ant_gpu(xin,nant,npol,acclen,nchan,split=split,out=out)
        
            
    print("Time taken final:", time.time() - st)
    vis = np.ma.masked_invalid(vis)
    return vis, rowcounts, aa.obj.channels