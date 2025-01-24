import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
# print(sys.path)
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.utils import pfb_utils as pu
from albatros_analysis.src.utils import baseband_utils as bu
import cupy as cp
import numpy as np
from albatros_analysis.src.utils import pycufft
import gc
#export USE_GPU=1
#export CUPY_CACHE_DIR=${PROJECT}/.cupy/kernel_cache
#conda deactivate
#conda activate albatros
#export LD_LIBRARY_PATH=/home/s/sievers/mohanagr/.conda/envs/albatros/lib:$LD_LIBRARY_PATH
# import albatros_analysis.src.baseband_data_classes as bdc
# import albatros_analysis.src.utils.pfb_utils as pu
# import albatros_analysis.src.utils.baseband_utils as bu

t_start = 1721800002 + 1000*5
t_end = t_start + 3600*8.5

a1_path = "/scratch/s/sievers/mohanagr/mars1_2024/baseband/"
a2_path = "/scratch/s/sievers/mohanagr/mars2_2024/baseband/"

files_a1, idx1 = bu.get_init_info(t_start, t_end, a1_path)
files_a2, idx2 = bu.get_init_info(t_start, t_end, a2_path)
print("old idx", idx1, idx2)
idx_correction=109993-100000
if idx_correction>0:
    idx1+=idx_correction
else:
    idx2+=np.abs(idx_correction)
print("new idx", idx1, idx2)

osamp=64
pfb_size = 1024 * osamp
cut=int(pfb_size/16)
acclen=pfb_size - 2*cut
idxstart=0
nchans=2049
T_ACCLEN = acclen*4096/250e6
nchunks = int((t_end-t_start)/T_ACCLEN)
print("acclen is", acclen, " spectra; corresponding to ", T_ACCLEN, "seconds")
print("pfb size is",pfb_size, "specs" )
print("osamp is", osamp)
print(f"cutting {cut} from each side of ipfb'd data")
print("nchunks is", nchunks)

channels = bdc.get_header(files_a1[0])["channels"].astype('int64')
# chanstart = np.where(channels == 1834)[0][0]
# chanend = np.where(channels == 1852)[0][0]
chanstart = 0
chanend = len(channels)
print(len(channels),"len chans")
ant1=bdc.BasebandFileIterator(files_a1, 0, idx1, acclen, nchunks=nchunks, chanstart=chanstart, chanend=chanend, type='float')
ant2=bdc.BasebandFileIterator(files_a2, 0, idx2, acclen, nchunks=nchunks, chanstart=chanstart, chanend=chanend, type='float')
a1_start = ant1.spec_num_start
a2_start = ant2.spec_num_start
to_ipfb_pol0 = cp.empty((pfb_size,nchans),dtype='complex64', order='C')
to_ipfb_pol1 = cp.empty((pfb_size,nchans),dtype='complex64', order='C')
print("to_ipfb size", np.prod(to_ipfb_pol0.shape)*8/1024**3, "GB")
channels=np.asarray(ant2.obj.channels,dtype='int64')
print(ant1.obj.channel_idxs)
print("channels ant1", channels[ant1.obj.channel_idxs])
print("channels ant2", channels[ant2.obj.channel_idxs])
print("file channel indices", chanstart, chanend)


print("pfb size", pfb_size)
ntap=4

nn=2*2048*osamp
dwin=pu.sinc_hamming(ntap,nn)
cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')
print("cupy_win_big size", np.prod(cupy_win_big.shape)*4/1024**3, "GB")
# pfb_chans = bu.get_pfb_chans(channels,osamp)
# start_chan = 1833*osamp
# end_chan = 1853*osamp
start_chan = 367*osamp
end_chan = 476*osamp
avg_data = np.zeros((nchunks-1,end_chan-start_chan),dtype="complex64")
print("avg data size", avg_data.shape, np.prod(avg_data.shape)*8/1024**3, "GB")
filt_thresh=0.45
flag=False
# antennas = [ant1,ant2,...]
# for i, chunks in enumerate(zip(*antennas)):
    # for chunk in chunks:
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()
matft=pu.get_matft(pfb_size)
start_event.record()
for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
    # print(pycufft.pycufft_cache)
    # pol0=chunk['pol0'] #this makes a copy...
    # pol1=chunk['pol1']
    # print("specnums", chunk['specnums'])
    # print("base pol0", pol0.base)
    perc_missing_a1 = (1 - len(chunk1["specnums"]) / acclen) * 100
    perc_missing_a2 = (1 - len(chunk2["specnums"]) / acclen) * 100
    # print("missing a1", perc_missing_a1, "missing a2", perc_missing_a2)
    # if len(chunk['specnums']<acclen):
    #     assert cp.sum(pol0[len(chunk['specnums']):])==0.
    # print(chunk['pol0'][mm:])
    # print("pol0 properties", pol0.dtype, pol0.shape, pol0.base is None, pol0.flags.c_contiguous)
    # print("pol0 dtype", pol0.dtype)
    # print("pol0 size", np.prod(pol0.shape)*8/1024**3, "GB")
    # print("chunk pol0 size", np.prod(chunk['pol0'].shape)*8/1024**3, "GB")
    if i==0:
        # matft=pu.get_matft(acclen)
        # raw_pol0 = pu.cupy_ipfb(pol0, matft, thresh=0.15) 
        pol0=bdc.make_continuous_gpu(chunk1['pol0'],chunk1['specnums']-a1_start,channels[ant1.obj.channel_idxs],acclen,nchans=2049)
        pol1=bdc.make_continuous_gpu(chunk2['pol0'],chunk2['specnums']-a2_start,channels[ant2.obj.channel_idxs],acclen,nchans=2049)
        to_ipfb_pol0[:2*cut,:] = pol0[-2*cut:,:].copy() # the only problem is if it's all zeros, first cut rows of next block will be zeros
        to_ipfb_pol1[:2*cut,:] = pol1[-2*cut:,:].copy() # the only problem is if it's all zeros, first cut rows of next block will be zeros
        continue
    if perc_missing_a1>10 or perc_missing_a2>10:
        print("MISSING ENCOUNTERED")
        avg_data[i-1,:] = np.nan #ignore this chunk
        flag=True
        continue
    pol0=bdc.make_continuous_gpu(chunk1['pol0'],chunk1['specnums']-a1_start,channels[ant1.obj.channel_idxs],acclen,nchans=2049)
    pol1=bdc.make_continuous_gpu(chunk2['pol0'],chunk2['specnums']-a2_start,channels[ant2.obj.channel_idxs],acclen,nchans=2049)
    if flag:
        #if not missing more than 10% and flag is ON, save the last bit for the next (hopefully) good chunk, and continue
        print("FLAG ENABLED BUT GOOD CHUNK")
        to_ipfb_pol0[:2*cut] = pol0[-2*cut:]
        to_ipfb_pol1[:2*cut] = pol1[-2*cut:]
        avg_data[i-1,:] = np.nan
        flag=False
        continue
    #print("we're not missing more than 10% and flag is OFF, process the whole chunk")
    to_ipfb_pol0[2*cut:] = pol0.copy()
    to_ipfb_pol1[2*cut:] = pol1.copy()
    # print(to_ipfb_pol0[:,channels[chanstart]:channels[chanend]])
    # print(to_ipfb_pol1[:,channels[chanstart]:channels[chanend]])
    # print(pol1)
    raw_pol0 = pu.cupy_ipfb(to_ipfb_pol0, matft, thresh=filt_thresh)
    raw_pol1 = pu.cupy_ipfb(to_ipfb_pol1, matft, thresh=filt_thresh)
    # print("IPFB done")
    pol0_new = pu.cupy_pfb(raw_pol0[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4)
    pol1_new = pu.cupy_pfb(raw_pol1[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4)
    # print("pfb done")
    to_ipfb_pol0[:2*cut] = pol0[-2*cut:].copy() #store for next iteration
    to_ipfb_pol1[:2*cut] = pol1[-2*cut:].copy() #store for next iteration
    # end_event.record()
    # end_event.synchronize()
    # print("cupy total time taken to do ipfb/pfb",cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    # do stuff with new pol0
    # start_event.record()
    avg_data[i-1,:] = cp.asnumpy(cp.mean(pol0_new[:,start_chan:end_chan]*cp.conj(pol1_new[:,start_chan:end_chan]),axis=0))
    # end_event.record()
    # end_event.synchronize()
    # print("cupy total time taken to transfer a row",cp.cuda.get_elapsed_time(start_event, end_event)/1000)
end_event.record()
end_event.synchronize()
print("cupy total time taken ",cp.cuda.get_elapsed_time(start_event, end_event)/1000)

np.savez(f"/project/s/sievers/mohanagr/avgdata_mars12_{acclen}_{filt_thresh}_{osamp}_sky.npz", data=avg_data, channels=np.arange(start_chan,end_chan), osamp=osamp)
print("data saved")