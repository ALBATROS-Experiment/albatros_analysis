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
    mult = 3
    lblock = 10
    mysize = 3
    pfbobj = pu.StreamingPFB(1,1,timestream_size = mysize, lblock = lblock)
    win = cp.asarray(sinc_hamming(4,lblock),dtype='float32')
    # myts = cp.arange(lblock*mysize*mult,dtype='float32')
    myts = cp.random.randn(lblock*mysize*mult).astype("float32")
    # myts = cp.tile(cp.arange(lblock,dtype='float32'), mysize*mult)
    # myts = cp.tile(cp.ones(lblock,dtype='float32'), mysize*mult)
    gpu_pfb = cp.zeros((len(myts)//lblock, lblock//2+1), dtype='complex64')
    print(gpu_pfb.shape)
    cpu_pfb = pfb(cp.asnumpy(myts),sinc_hamming,nchan=lblock//2+1)
    gpu_full_pfb = pu.cupy_pfb(myts, win, nchan=lblock//2+1)
    # myts = cp.arange(8*10,dtype='float32')
    # print("MYTS",myts)
    niter = len(myts)//mysize
    jj=0
    for i in range(niter):
        # print("iteration", i)
        ret=pfbobj.pfb(0,0,myts[i*mysize:(i+1)*mysize])
        if ret is not None:
            gpu_pfb[jj:jj+ret.shape[0],:] = ret
            jj+=ret.shape[0]
            # print(jj)
        # if spec:
        #     print("mean spec", cp.abs(cp.mean(spec,axis=0)))
        #     plt.plot(cp.asnumpy(cp.abs(cp.mean(spec,axis=0))),marker='o')
        #     plt.xlim(0,5)
        #     plt.ylim(0,20)
        #     plt.savefig(f"./test_pfb_{i}.jpg")
        # print(spec)
    print(gpu_pfb.shape, cpu_pfb.shape, jj)
    print("gpu stream pfb\n", gpu_pfb[3:,])
    print("gpu full pfb\n", gpu_full_pfb)
    print("cpu full pfb\n", cpu_pfb)
    print("diff2\n", gpu_full_pfb-gpu_pfb[3:,])
    err1 = gpu_full_pfb-gpu_pfb[3:,:]
    err2 = cp.asnumpy(gpu_pfb[3:,:])-cpu_pfb[:,:]
    print("gpu full pfb vs streaming pfb error1 max:", np.max(np.abs(err1)), "stddev:", np.std(err1))
    print("Max error cpu vs gpu max:", np.max(np.abs(err2)), "stddev:", np.std(err2)) #discard top 3 rows

def test_pfb_gpu_vs_cpu(mysize, lblock = 4096):
    # mysize simulates how many timestream samples we'll get each time we ipfb a small chunk
    # PFB of ENTIRE timestream SHOULD BE THE SAME
    # whole timestream at once  ==  lots of small timesteram chunks

    ts_specnum = 5000
    pfbobj = pu.StreamingPFB(8,2,timestream_size = mysize, lblock = lblock)
    win = pfbobj.win.ravel()
    print("win shape", win.shape, "win dtype", win.dtype)
    win_np = cp.asnumpy(win)
    
    ts = np.random.randn(ts_specnum*lblock).astype('float32')
    ts_d1 = cp.asarray(ts)
    cpu_pfb1 = pfb(ts, lambda n, l: win_np, nchan=lblock//2+1) # Use exact same window bits
    gpu_full_pfb1 = pu.cupy_pfb(ts_d1, win, nchan=lblock//2+1)
    
    ts2 = np.random.randn(ts_specnum*lblock).astype('float32')
    ts_d2 = cp.asarray(ts2)
    cpu_pfb2 = pfb(ts2, lambda n, l: win_np, nchan=lblock//2+1) # Use exact same window bits
    gpu_full_pfb2 = pu.cupy_pfb(ts_d2, win, nchan=lblock//2+1)
    
    print("cpu pfb shape", cpu_pfb1.shape)
    print("mysize", mysize)
    niter = len(ts)//mysize+1
    # gpu_pfb_d1 = cp.zeros((ts_specnum,lblock//2+1),dtype='complex64')
    # gpu_pfb_d2 = cp.zeros((ts_specnum,lblock//2+1),dtype='complex64')
    gpu_pfb_d1 = cp.zeros((ts_specnum,lblock),dtype='float32')
    gpu_pfb_d2 = cp.zeros((ts_specnum,lblock),dtype='float32')
    print("gpu_pfb shape", gpu_pfb_d1.shape)
    
    jj=0
    kk=0
    for i in range(niter):
        spec1=pfbobj.pfb(0,0,ts_d1[i*mysize:(i+1)*mysize])
        spec2=pfbobj.pfb(0,1,ts_d2[i*mysize:(i+1)*mysize])
        if spec1 is not None:
            gpu_pfb_d1[jj:jj+spec1.shape[0],:]=spec1
            jj+=spec1.shape[0]
        if spec2 is not None:
            gpu_pfb_d2[kk:kk+spec2.shape[0],:]=spec2
            kk+=spec2.shape[0]

    err1_full_vs_stream = gpu_full_pfb1-gpu_pfb_d1[3:,:]
    err2_full_vs_stream = gpu_full_pfb2-gpu_pfb_d2[3:,:]
    print("gpu full pfb vs streaming pfb error1 max:", np.max(np.abs(err1_full_vs_stream)), "stddev:", np.std(err1_full_vs_stream))
    print("gpu full pfb vs streaming pfb error2 max:", np.max(np.abs(err2_full_vs_stream)), "stddev:", np.std(err2_full_vs_stream))
    
    # err1_cpu_vs_full = cpu_pfb1 - cp.asnumpy(gpu_full_pfb1)
    # err2_cpu_vs_full = cpu_pfb2 - cp.asnumpy(gpu_full_pfb2)
    # print("CPU vs GPU full error1 max:", np.max(np.abs(err1_cpu_vs_full)), "stddev:", np.std(err1_cpu_vs_full))
    # print("CPU vs GPU full error2 max:", np.max(np.abs(err2_cpu_vs_full)), "stddev:", np.std(err2_cpu_vs_full))

    # gpu_pfb1 = cp.asnumpy(gpu_pfb_d1)
    # gpu_pfb2 = cp.asnumpy(gpu_pfb_d2)
    # err1_cpu_vs_gpu = gpu_pfb1[3:,:]-cpu_pfb1[:,:]
    # err2_cpu_vs_gpu = gpu_pfb2[3:,:]-cpu_pfb2[:,:]
    # print("Max error1 cpu vs gpu max:", np.max(np.abs(err1_cpu_vs_gpu)), "stddev:", np.std(err1_cpu_vs_gpu)) #discard top 3 rows
    # print("Max error2 cpu vs gpu max:", np.max(np.abs(err2_cpu_vs_gpu)), "stddev:", np.std(err2_cpu_vs_gpu)) #discard top 3 rows
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
    # test_pfb_manual()
    test_pfb_gpu_vs_cpu(11000, lblock=4096) #some timestream value > lblock
    test_pfb_gpu_vs_cpu(4096, lblock=4096) #some timestream value = lblock
    test_pfb_gpu_vs_cpu(356, lblock=4096) #some timestream value < lblock
    # reproduce_bug()
    # speed_test()