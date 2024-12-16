import numpy as np
import time
import sys
import pytest

sys.path.insert(0,'/home/mohan/Projects/albatros_analysis/')
from src.correlations import baseband_data_classes as bdc
from src.correlations import correlations as cr

def get_avg_fast(acclen, nchunks, chanstart=0, chanend=None):
    
    # idxstart, fileidx, files = butils.get_init_info(init_t, end_t, path)
    idxstart=0
    fileidx=0
    files=['/home/mohan/Projects/albatros_analysis/correlations/tests/data/1627202039_wrap.raw']
    print("Starting at: ",idxstart, "in filenum: ",fileidx)
    print(files[fileidx])

    ant1 = bdc.BasebandFileIterator(files,fileidx,idxstart,acclen,nchunks=nchunks,chanstart=chanstart,chanend=chanend)
    if(ant1.obj.bit_mode!=4):
        raise NotImplementedError(f"BIT MODE {ant1.obj.bit_mode} IS NOT SUPPORTED BY THIS SCRIPT. Do you want to use autocorravg1bit.py?")
    ncols=ant1.obj.chanend-ant1.obj.chanstart
    pol00=np.zeros((nchunks,ncols),dtype='float64',order='c')
    pol11=np.zeros((nchunks,ncols),dtype='float64',order='c')
    pol01=np.zeros((nchunks,ncols),dtype='complex64',order='c')
    j=ant1.spec_num_start
    m=ant1.spec_num_start
    st=time.time()
    for i, chunk in enumerate(ant1):
        t1=time.time()
        pol00[i,:] = cr.avg_autocorr_4bit(chunk['pol0'],chunk['specnums'])
        pol11[i,:] = cr.avg_autocorr_4bit(chunk['pol1'],chunk['specnums'])
        pol01[i,:] = cr.avg_xcorr_4bit(chunk['pol0'], chunk['pol1'],chunk['specnums'])
        t2=time.time()
        print("time taken for one loop", t2-t1)
        j=ant1.spec_num_start
        print("After a loop spec_num start at:", j, "Expected at", m+(i+1)*acclen)
        print(i+1,"CHUNK READ")
    print("Time taken final:", time.time()-st)
    pol00 = np.ma.masked_invalid(pol00)
    pol11 = np.ma.masked_invalid(pol11)
    pol01 = np.ma.masked_invalid(pol01)
    return pol00,pol11,pol01,ant1.obj.channels

def test_autocorravg():
    acclen=122700
    nchunks=1
    pol00,pol11,pol01,chans=get_avg_fast(acclen,nchunks)
    print(pol00)
