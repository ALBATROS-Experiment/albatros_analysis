import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
# print(sys.path)
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import pfb_utils as pu
from albatros_analysis.src.utils import baseband_utils as bu
import cupy as cp
import numpy as np
# import albatros_analysis.src.baseband_data_classes as bdc
# import albatros_analysis.src.utils.pfb_utils as pu
# import albatros_analysis.src.utils.baseband_utils as bu

t_start=1721400005
t_end=t_start+3*3600
files,start_file_idx=bu.get_init_info(t_start,t_end,"/scratch/s/sievers/mohanagr/mars_axion/baseband")
cut=32
acclen=cut*1024
idxstart=0
nchans=2049
ant=bdc.BasebandFileIterator(files, start_file_idx, 0, acclen, nchunks=500, type='float')
pfb_size = acclen + 2*cut
to_ipfb = cp.empty((pfb_size,nchans),dtype='complex64')
channels=np.asarray(ant.obj.channels,dtype='int64')
print("channels are", channels, "dtype", channels.dtype)
print("pfb size", pfb_size)
# IPFB SETUP
ntap=4
nn=4096
dwin=pu.sinc_hamming(ntap,nn)
cupy_win=cp.asarray(dwin,dtype='float32')

osamp=1024
nn=2*2048*osamp
dwin=pu.sinc_hamming(ntap,nn)
cupy_win_big=cp.asarray(dwin,dtype='float32')
print("cupy_win_big size", cupy_win_big.shape)

avg_data = cp.zeros((500-1,2048*osamp+1),dtype="complex64")
# antennas = [ant1,ant2,...]
# for i, chunks in enumerate(zip(*antennas)):
    # for chunk in chunks:
for i, chunk in enumerate(ant):
    print("starting iteration for chunk", i)
    pol0=chunk['pol0'] #this makes a copy...
    print(pol0.dtype, pol0.shape)
    # pol1=chunk['pol1']
    print("specnums", chunk['specnums'])
    print("base pol0", pol0.base)
    print(f"missing : {1-len(chunk['specnums'])/acclen:5.3f}")
    if len(chunk['specnums']<acclen):
        assert cp.sum(pol0[len(chunk['specnums']):])==0.
    pol0=bdc.make_continuous_gpu(pol0,chunk['specnums']-chunk['specnums'][0],channels,acclen,nchans=2049)
    if i==0:
        # matft=pu.get_matft(acclen)
        # raw_pol0 = pu.cupy_ipfb(pol0, matft, thresh=0.15) 
        matft=pu.get_matft(pfb_size)
        to_ipfb[:2*cut] = pol0[-2*cut:]
        continue
    else:
        to_ipfb[2*cut:] = pol0
        assert to_ipfb.base is None
        # print("matft shape is", matft.shape)
        raw_pol0 = pu.cupy_ipfb(to_ipfb, matft, thresh=0.5)
    # print("IPFB done")
    pol0_new = pu.cupy_pfb(raw_pol0[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4) #size a bit smaller acclen - 2*cut for first chunk
    # print("pfb done")
    to_ipfb[:2*cut] = pol0[-2*cut:] #store for next iteration
    #do stuff with new pol0
    avg_data[i-1,:] = cp.mean(pol0_new*pol0_new.conj(),axis=0)

avg_data=cp.asnumpy(avg_data)

np.savez("/project/s/sievers/mohanagr/avgdata_0.5.npz", avg_data)
print("data saved")