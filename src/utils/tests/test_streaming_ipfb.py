import cupy as cp
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)
cp.random.seed(42)

def get_samples(mult,k,dk,N=4096,snr=10):
    N=N*mult
    dk=int(dk*mult)
    num_k = len(k)
    snr/=dk 
    rns=cp.random.randn(2*dk*num_k)
    f=cp.zeros(N//2+1,dtype='complex64')
    bb=num_k*dk
    for i,j in enumerate(k):
        j= int(j*mult)
        f[j-dk//2:j+dk//2]=cp.sqrt(snr)*rns[i*dk:(i+1)*dk] + 1j*cp.sqrt(snr)*rns[bb+i*dk:bb+(i+1)*dk]
    x_new=cp.fft.irfft(f)
    print(x_new.dtype)
    return x_new

def test_ipfb_IQ():

    lblock=4096
    ntap=4
    dwin=pu.sinc_hamming(ntap,lblock)
    cupy_win=cp.asarray(dwin,dtype='float32',order='c')

    mult=65536-2*10+3
    k = np.asarray([100,])
    dk=10
    x=get_samples(mult,k=k,dk=dk,snr=10)
    spec=pu.cupy_pfb(x, cupy_win, out=None,nchan=2049, ntap=4)
    print(spec.shape)
    

    channels = cp.arange(k[0]-dk//2,k[0]+dk//2,dtype='int32')
    print(channels)

    nant=1
    npol=1
    ipfb_IQ = pu.StreamingIPFB_IQ(nant, npol, channels, nblock=65536, lblock=4096, ntap=4, window='hamming', cut=10)
    ipfb = pu.StreamingIPFB(nant, npol, channels, nblock=65536, lblock=4096, ntap=4, window='hamming', cut=10)
    x_ipfb_lowsamp = ipfb_IQ.ipfb(0,0,spec[:,k[0]-dk//2:k[0]+dk//2],thresh=0.1)
    print("spec shape",spec.shape)
    x_ipfb_highsamp = ipfb.ipfb(0,0,spec[:,k[0]-dk//2:k[0]+dk//2],thresh=0.1)
    osamp=4
    print(x_ipfb_lowsamp.reshape(-1,32*osamp).shape) #both of them have the same number of rows as expected
    print(x_ipfb_highsamp.reshape(-1,4096*osamp).shape)

    spec_new_lowsamp = cp.fft.fft(x_ipfb_lowsamp.reshape(-1,32*osamp),axis=1)
    spec_new_highsamp = cp.fft.rfft(x_ipfb_highsamp.reshape(-1,4096*osamp),axis=1)

    spec_new_lowsamp_slice = spec_new_lowsamp[:,:len(channels)*osamp]
    spec_new_highsamp_slice = spec_new_highsamp[:,channels[0]*osamp:(channels[-1]+1)*osamp]
    print(spec_new_lowsamp_slice.shape)
    print(spec_new_highsamp_slice.shape)
    print(cp.std(spec_new_lowsamp_slice),cp.std(spec_new_highsamp_slice))
    print(cp.std(spec_new_lowsamp_slice - spec_new_highsamp_slice))

def speed_test():
    nant=1
    npol=1
    nchan=300
    nspec=65536 - 2*10
    channels=np.arange(100,100+nchan,dtype='int32')
    spectra = (cp.random.randn(nspec,nchan) + 1j*cp.random.randn(nspec,nchan)).astype('complex64')
    print("spectra shape", spectra.shape)
    ipfb_IQ = pu.StreamingIPFB_IQ(nant, npol, channels, nblock=65536, lblock=4096, ntap=4, window='hamming', cut=10)
    ipfb = pu.StreamingIPFB(nant, npol, channels, nblock=65536, lblock=4096, ntap=4, window='hamming', cut=10)
    niter=10
    start_event= cp.cuda.Event()
    end_event= cp.cuda.Event()
    reg_times=[]
    for i in range(10):
        start_event.record()
        x1=ipfb.ipfb(0,0,spectra,thresh=0.1)
        end_event.record()
        end_event.synchronize()
        reg_times.append((cp.cuda.get_elapsed_time(start_event, end_event))/1000)
    
    IQ_times=[]
    for i in range(10):
        start_event.record()
        x1=ipfb_IQ.ipfb(0,0,spectra,thresh=0.1)
        end_event.record()
        end_event.synchronize()
        IQ_times.append((cp.cuda.get_elapsed_time(start_event, end_event))/1000)

    print("Median IPFB reg time:", np.median(reg_times), "s")
    print("Median IPFB IQ time:", np.median(IQ_times), "s")
    print("Gsamp/s (ADC) reg:", 2*nspec*2048/np.median(reg_times)/1e9)
    print("Gsamp/s (ADC) IQ:", 2*nspec*2048/np.median(IQ_times)/1e9)
    print("ratio reg/IQ:", np.median(reg_times)/np.median(IQ_times))


    # pfb.ipfb(0,0,spec[:,k[0]-dk//2:k[0]+dk//2],thresh=0.1)




if __name__=="__main__":
    # test_ipfb_IQ()
    speed_test()