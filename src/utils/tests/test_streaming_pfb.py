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

def pfb(timestream,  window, nchan=2049, ntap=4):

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

def test_pfb_vs_cpu():
    # mysize = 500 + 0 #1000 extra
    ts_specnum = 5000
    lblock = 4096
    ts = np.random.randn(ts_specnum*lblock).astype('float32')
    # ts = np.arange(ts_specnum*lblock).astype('float32')/10000
    ts_d = cp.asarray(ts)
    assert ts_d.dtype == 'float32'
    cpu_pfb = pfb(ts,sinc_hamming,nchan=lblock//2+1)
    print("cpu pfb shape", cpu_pfb.shape)
    mysize = 629
    niter = len(ts)//mysize+1
    gpu_pfb_d = cp.zeros((ts_specnum,lblock//2+1),dtype='complex64')
    # gpu_pfb_d = cp.zeros((ts_specnum,lblock),dtype='float32')
    print("gpu_pfb shape", gpu_pfb_d.shape)
    pfbobj = pu.StreamingPFB(8,2,chans=cp.arange(200),timestream_size = mysize, lblock = lblock)
    jj=0
    for i in range(niter):
        print(f"---------------------------iteration {i}------------------------")
        spec=pfbobj.pfb(ts_d[i*mysize:(i+1)*mysize])
        # print("spectra shape",spec.shape)
        if spec is not None:
            gpu_pfb_d[jj:jj+spec.shape[0],:]=spec
            jj+=spec.shape[0]
        # if spec:
        #     print("mean spec", cp.abs(cp.mean(spec,axis=0)))
        #     plt.plot(cp.asnumpy(cp.abs(cp.mean(spec,axis=0))),marker='o')
        #     plt.xlim(0,5)
        #     plt.ylim(0,20)
        #     plt.savefig(f"./test_pfb_{i}.jpg")
        # print(spec)
    gpu_pfb = cp.asnumpy(gpu_pfb_d)
    print(cpu_pfb)
    print(gpu_pfb[3:])
    print("Max error\n", np.max(np.abs(gpu_pfb[3:,:]-cpu_pfb[:,:])))
    arg = np.argmax(np.abs(gpu_pfb[3:,:]-cpu_pfb[:,:]))
    argrow = arg//lblock
    argcol = arg%lblock
    print(gpu_pfb[3+argrow,argcol-5:argcol+5])
    print("------------------------------")
    print(cpu_pfb[argrow, argcol-5:argcol+5])
    gpu_win = cp.asnumpy(pfbobj.win.ravel())
    cpu_win = sinc_hamming(4,lblock).astype("float32")
    print("Max window error", np.max(np.abs(gpu_win-cpu_win)))
if __name__=="__main__":
    test_pfb_vs_cpu()

# Cpu will pfb [15.962238 16.002884 16.078419 16.122477 16.09763  16.010254 15.909212
#  15.868957]
# Cpu will pfb [23.943357 23.988358 24.099424 24.175674 24.155144 24.039167 23.893332
#  23.824871]

# [[15.962238 16.002886 16.07842  16.122478 16.09763  16.010256 15.90921 15.868957]]
# [[23.943357 23.988358 24.099426 24.175678 24.155144 24.03917  23.89333 23.824871]]