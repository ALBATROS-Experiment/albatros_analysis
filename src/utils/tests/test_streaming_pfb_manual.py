import numpy as np
import cupy as cp
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
from albatros_analysis.src.utils import pycufft

def sinc_hamming(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hamming(ntap*lblock)*np.sinc(w/lblock)

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

def cupy_pfb(timestream, win, out=None,nchan=2049, ntap=4):
    lblock = 2*(nchan-1)
    nblock = timestream.size // lblock - (ntap - 1)
    timestream=timestream.reshape(-1,lblock)
    if out is not None:
        assert out.shape == (nblock, nchan)
    win=win.reshape(ntap,lblock)
    y = timestream * win[:,cp.newaxis,:]
    y=y[0,:nblock,:]+y[1,1:nblock+1,:]+y[2,2:nblock+2,:]+y[3,3:nblock+3,:]
    print("y shape", y.shape, "y flags", y.flags, "y dtype", y.dtype)
    out=cp.fft.rfft(y,axis=1)
    # out=pycufft.rfft(y,axis=1)
    # print(pycufft.pycufft_cache)
    return out

class StreamingPFB():
    def __init__(self, nant, npol, timestream_size = 50000, lblock=4096, ntap=4, window='hamming', dtype='float'):
        self.dtype = 'float32'
        self.nant = nant
        self.npol = npol
        self.nchan =lblock//2+1
        self.ntap = ntap
        self.lblock = lblock
        self.timestream_size = timestream_size
        N = self.lblock * ntap
        self.win = sinc_hamming(ntap, lblock).astype(self.dtype)
        self.win = self.win.reshape(ntap, lblock)
        self.nblock = timestream_size // lblock
        self.rem = timestream_size % lblock
        self.overlap = (self.ntap - 1)*self.lblock
        print(f"tssize: {timestream_size}\noverlap: {self.overlap}\nrem: {self.rem}\nnblock: {self.nblock}\nlblock: {self.lblock}\n")
        self.tsbuf = np.zeros((self.nant, self.npol, self.nblock*self.lblock + self.overlap + self.lblock), dtype=self.dtype, order='C') #will be used to pfb, one extra lblock to accomodate spillover spectra
        self.rembuf = -1*np.ones((self.nant, self.npol, self.lblock + self.rem), dtype=self.dtype, order='C') #little bit extra to accomodate rem spillover
        print("rembuf size", self.rembuf.shape)
        self.remptr = np.zeros((self.nant, self.npol), dtype='int32')
    
    def pfb(self, antidx, polidx, timestream):
        timestream = np.asarray(timestream)
        remptr = self.remptr[antidx, polidx]
        out=None
        incoming = len(timestream)
        used = 0
        total_available = remptr + incoming
        spec_possible = total_available // self.lblock
        spec_size = spec_possible * self.lblock
        # print("spec possible", spec_possible, "incoming", timestream)
        # print("remptr @", remptr, "rembuf", self.rembuf)
        # print("tsbuf", self.tsbuf)
        if spec_possible > 0:
            self.tsbuf[antidx, polidx, self.overlap : self.overlap + remptr] = self.rembuf[antidx, polidx,  : remptr]
            self.tsbuf[antidx, polidx, self.overlap + remptr : self.overlap + spec_size] = timestream[ : spec_size - remptr]
            used = (spec_size - remptr)
            self.remptr[antidx, polidx] = 0  
            # print("used", used, "reset remptr", self.remptr[antidx, polidx])  
            x = self.tsbuf[antidx, polidx,  : spec_size + self.overlap].reshape(-1, self.lblock)
            # print("tsbuf\n", x)
            #onwards to pfb
            y = x * self.win[:,np.newaxis,:]
            y = y[0,:spec_possible,:]+y[1,1:spec_possible+1,:]+y[2,2:spec_possible+2,:]+y[3,3:spec_possible+3,:]
            out = np.fft.rfft(y,axis=1)
            self.tsbuf[antidx, polidx, :self.overlap] = self.tsbuf[antidx, polidx, spec_size:spec_size+self.overlap].copy()
        self.rembuf[antidx, polidx,  self.remptr[antidx, polidx] : self.remptr[antidx, polidx] + incoming - used] = timestream[used :].copy() #if spec_size 0, just loads the last rem of timestream = entire timestream
        self.remptr[antidx, polidx] += incoming - used
        # print("remptr @",self.remptr[antidx, polidx], "rembuf", self.rembuf)
        # print("tsbuf", self.tsbuf)
        # print("------------------------------------------------------------------")
        # print("PFB OUT SHAPE", out.shape, out.flags)
        return out

def test_pfb_gpu_vs_cpu(mysize, lblock = 4096):

    ts_specnum = 5000
    pfbobj = StreamingPFB(8,2,timestream_size = mysize, lblock = lblock)
    print("pfbobj win dtype", pfbobj.win.dtype, "pfbobj win shape", pfbobj.win.shape)
    ts = np.random.randn(ts_specnum*lblock).astype('float32')
    cpu_pfb1 = pfb(ts, sinc_hamming, nchan=lblock//2+1) # Use exact same window bits
    gpu_pfb1 = cupy_pfb(cp.asarray(ts), cp.asarray(pfbobj.win.ravel()), nchan=lblock//2+1)
    print("cpu vs gpu error1 max:", np.max(np.abs(cpu_pfb1-cp.asnumpy(gpu_pfb1))), "stddev:", np.std(cpu_pfb1-cp.asnumpy(gpu_pfb1)))
    
    ts2 = np.random.randn(ts_specnum*lblock).astype('float32') 
    cpu_pfb2 = pfb(ts2, sinc_hamming, nchan=lblock//2+1) # Use exact same window bits

    cpu_streaming_pfb1 = np.zeros((ts_specnum,lblock//2+1),dtype='complex64')
    cpu_streaming_pfb2 = np.zeros((ts_specnum,lblock//2+1),dtype='complex64')
    niter = len(ts)//mysize+1
    jj=0
    kk=0
    for i in range(niter):
        spec1=pfbobj.pfb(0,0,ts[i*mysize:(i+1)*mysize])
        spec2=pfbobj.pfb(0,1,ts2[i*mysize:(i+1)*mysize])
        if spec1 is not None:
            cpu_streaming_pfb1[jj:jj+spec1.shape[0],:]=spec1
            jj+=spec1.shape[0]
        if spec2 is not None:
            cpu_streaming_pfb2[kk:kk+spec2.shape[0],:]=spec2
            kk+=spec2.shape[0]
    err1_cpu_full_vs_stream = cpu_pfb1-cpu_streaming_pfb1[3:,:]
    err2_cpu_full_vs_stream = cpu_pfb2-cpu_streaming_pfb2[3:,:]
    print("cpu full pfb vs streaming pfb error1 max:", np.max(np.abs(err1_cpu_full_vs_stream)), "stddev:", np.std(err1_cpu_full_vs_stream))
    print("cpu full pfb vs streaming pfb error2 max:", np.max(np.abs(err2_cpu_full_vs_stream)), "stddev:", np.std(err2_cpu_full_vs_stream))

if __name__=="__main__":
    test_pfb_gpu_vs_cpu(10000, lblock=4096)
    test_pfb_gpu_vs_cpu(11000, lblock=4096)
    test_pfb_gpu_vs_cpu(4096, lblock=4096)
    test_pfb_gpu_vs_cpu(353, lblock=4096)
