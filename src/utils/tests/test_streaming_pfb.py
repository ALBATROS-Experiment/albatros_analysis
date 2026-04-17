import cupy as cp
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)

def sinc_hamming(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hamming(ntap*lblock)*np.sinc(w/lblock)

def reproduce_bug():
    N = 1 << 20
    x = cp.arange(N,dtype='float32')
    x[:N-100] = x[100:]
    print(cp.sum(x[:N-100]-cp.arange(100,N))) #usually zero but not always

def pfb(timestream,  window, nchan=2049, ntap=4):
    # old cpu pfb from Steve.
    # Slow but true.

    # number of samples in a sub block
    lblock = 2*(nchan-1)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock==int(nblock): nblock=int(nblock)
    else: raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum 
    dtype=np.complex64 if timestream.dtype=='float32' else np.complex128
    spec = np.zeros((nblock,nchan), dtype=dtype)
    # spec = np.zeros((nblock,lblock), dtype='float32')

    def s(ts_sec):
        return np.sum(ts_sec.reshape(ntap,lblock),axis=0) # this is equivalent to sampling an ntap*lblock long fft - M

    win = window(ntap,lblock)
    win = win.astype(timestream.dtype)
    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi*lblock:(bi+ntap)*lblock].copy()
        to_pfb = ts_sec*win
        # print("Cpu will pfb", to_pfb.reshape(ntap, lblock))
        # print("Cpu will pfb", s(to_pfb))
        spec[bi] = np.fft.rfft(s(to_pfb)) 
        # spec[bi] = s(to_pfb)

    return spec

def test_pfb_manual():
    # mysize = 500 + 0 #1000 extra
    mysize = 9 + 0 #2 extra
    lblock = 8
    pfbobj = pu.StreamingPFB(8,2,chans=cp.arange(200),timestream_size = mysize, lblock = lblock)
    # myts = cp.random.randn(mysize)
    # myts = myts.astype("float32")
    # myts = cp.ones(mysize,dtype='float32')
    myts = cp.arange(8*10,dtype='float32')
    print("MYTS",myts)
    for i in range(20):
        print("iteration", i)
        spec=pfbobj.pfb(myts[i*mysize:(i+1)*mysize])
        # if spec:
        #     print("mean spec", cp.abs(cp.mean(spec,axis=0)))
        #     plt.plot(cp.asnumpy(cp.abs(cp.mean(spec,axis=0))),marker='o')
        #     plt.xlim(0,5)
        #     plt.ylim(0,20)
        #     plt.savefig(f"./test_pfb_{i}.jpg")
        # print(spec)

def test_pfb_gpu_vs_cpu(mysize):
    # mysize simulates how many timestream samples we'll get each time we ipfb a small chunk
    # PFB of ENTIRE timestream SHOULD BE THE SAME
    # whole timestream at once  ==  lots of small timesteram chunks

    ts_specnum = 5000
    lblock = 4096
    assert mysize < ts_specnum * lblock
    ts = np.random.randn(ts_specnum*lblock).astype('float32')
    cpu_pfb1 = pfb(ts,sinc_hamming,nchan=lblock//2+1) #pfb of whole timestream
    # ts = np.arange(ts_specnum*lblock).astype('float32')/10000
    ts_d1 = cp.asarray(ts)
    ts = np.random.randn(ts_specnum*lblock).astype('float32')
    cpu_pfb2 = pfb(ts,sinc_hamming,nchan=lblock//2+1) #pfb of whole timestream
    ts_d2 = cp.asarray(ts)
    assert ts_d1.dtype == 'float32'
    
    print("cpu pfb shape", cpu_pfb1.shape)
    print("mysize", mysize)
    niter = len(ts)//mysize+1
    gpu_pfb_d1 = cp.zeros((ts_specnum,lblock//2+1),dtype='complex64')
    gpu_pfb_d2 = cp.zeros((ts_specnum,lblock//2+1),dtype='complex64')
    # gpu_pfb_d = cp.zeros((ts_specnum,lblock),dtype='float32')
    print("gpu_pfb shape", gpu_pfb_d1.shape)
    pfbobj = pu.StreamingPFB(8,2,chans=cp.arange(200),timestream_size = mysize, lblock = lblock)
    jj=0
    kk=0
    for i in range(niter):
        # print(f"---------------------------iteration {i}------------------------")
        spec1=pfbobj.pfb(0,0,ts_d1[i*mysize:(i+1)*mysize])
        spec2=pfbobj.pfb(5,1,ts_d2[i*mysize:(i+1)*mysize])
        # print("spectra shape",spec.shape)
        if spec1 is not None:
            gpu_pfb_d1[jj:jj+spec1.shape[0],:]=spec1
            jj+=spec1.shape[0]
        if spec1 is not None:
            gpu_pfb_d2[kk:kk+spec2.shape[0],:]=spec2
            kk+=spec2.shape[0]
    gpu_pfb1 = cp.asnumpy(gpu_pfb_d1)
    gpu_pfb2 = cp.asnumpy(gpu_pfb_d2)
    print("Max error1\n", np.max(np.abs(gpu_pfb1[3:,:]-cpu_pfb1[:,:]))) #discard top 3 rows
    print("Max error2\n", np.max(np.abs(gpu_pfb2[3:,:]-cpu_pfb2[:,:]))) #discard top 3 rows
    # print("Max error3\n", np.max(np.abs(gpu_pfb1[3:,:]-gpu_pfb2[3:,:]))) #discard top 3 rows
    # arg = np.argmax(np.abs(gpu_pfb[3:,:]-cpu_pfb[:,:]))
    # argrow = arg//lblock
    # argcol = arg%lblock
    # print(gpu_pfb[3+argrow,argcol-5:argcol+5])
    # print("------------------------------")
    # print(cpu_pfb[argrow, argcol-5:argcol+5])
    # gpu_win = cp.asnumpy(pfbobj.win.ravel())
    # cpu_win = sinc_hamming(4,lblock).astype("float32")
    # print("Max window error", np.max(np.abs(gpu_win-cpu_win)))
<<<<<<< HEAD
if __name__=="__main__":

    test_pfb_gpu_vs_cpu(4096) #some timestream value > lblock
    test_pfb_gpu_vs_cpu(4095) #some timestream value < lblock
    # reproduce_bug()
=======

def speed_test():
    ts = cp.random.randn(65536*4096).astype('float32')

    osamp = 8192
    pfbobj = pu.StreamingPFB(1,1,timestream_size = ts.size, lblock = 4096*osamp)
    niter=10
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    times=[]
    for i in range(niter):
        start_event.record()
        spec=pfbobj.pfb(0,0,ts)
        end_event.record()
        end_event.synchronize()
        # print("spec shape", spec.shape)
        times.append(cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    print("Median PFB time:", np.median(times), "s")
    print("Gsamp/s (ADC):", ts.size/np.median(times)/1e9)

if __name__=="__main__":

    # test_pfb_gpu_vs_cpu(4096) #some timestream value > lblock
    # test_pfb_gpu_vs_cpu(4095) #some timestream value < lblock
    # reproduce_bug()
    speed_test()
>>>>>>> 691fef5d7fa0e0ace00e9a1317533f35cda7656b
