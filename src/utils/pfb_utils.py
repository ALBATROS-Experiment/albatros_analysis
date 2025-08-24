import numpy as np
import cupy as cp
from . import pycufft
import time

def _print_class_mem_usage(arr_dict, header):
    lines = [header, f"{'Name':<10}{'Shape':<25}{'Dtype':<12}{'Size':>12}{'Memory':>12}"]
    total_mem = 0
    for name, arr in arr_dict.items():
        if arr is None:
            continue
        lines.append(
            f"{name:<10}{str(arr.shape):<25}{str(arr.dtype):<12}"
            f"{arr.size:>12,d}{arr.nbytes/1e6:>11.2f} MB"
        )
        total_mem += arr.nbytes
    lines.append(f"Total memory: {total_mem/1e6:.2f} MB")
    return "\n".join(lines)

class StreamingPFB():
    """
    Works for arbitrary timestream sizes. Can be less than lblock.
    """
    def __init__(self, nant, npol, chans, timestream_size = 50000, lblock=4096, ntap=4, window='hamming'):
        self.nant = nant
        self.npol = npol
        self.nchan =lblock//2+1
        self.ntap = ntap
        self.lblock = lblock
        self.chans = chans
        self.timestream_size = timestream_size
        N = self.lblock * ntap
        self.win = cp.__dict__[window](N) * cp.sinc((cp.arange(0, N) - N // 2) / self.lblock)
        self.win = self.win.astype("float32")
        self.win = self.win.reshape(ntap, lblock)
        self.nblock = timestream_size // lblock
        self.rem = timestream_size % lblock
        self.overlap = (self.ntap - 1)*self.lblock
        # print(f"tssize: {timestream_size}\noverlap: {self.overlap}\nrem: {self.rem}\nnblock: {self.nblock}\nlblock: {self.lblock}\n")
        self.tsbuf = cp.zeros(self.nblock*self.lblock + self.overlap + self.lblock, dtype='float32', order='C') #will be used to pfb, one extra lblock to accomodate spillover spectra
        self.rembuf = -1*cp.ones(self.lblock + self.rem, dtype='float32', order='C') #little bit extra to accomodate rem spillover
        # print("rembuf size", self.rembuf.shape)
        self.tsptr = 0
        self.remptr = 0
    
    def pfb(self, timestream):
        out=None
        incoming = len(timestream)
        used = 0
        total_available = self.remptr + incoming
        spec_possible = total_available // self.lblock
        spec_size = spec_possible * self.lblock
        if spec_possible > 0:
            self.tsbuf[self.overlap : self.overlap + self.remptr] = self.rembuf[ : self.remptr].copy()
            self.tsbuf[self.overlap + self.remptr : self.overlap + spec_size] = timestream[ : spec_size - self.remptr].copy()
            used = (spec_size - self.remptr)
            self.remptr=0
            x = self.tsbuf[ : spec_size + self.overlap].reshape(-1, self.lblock)
            #onwards to pfb
            y = x * self.win[:,cp.newaxis,:]
            y = y[0,:spec_possible,:]+y[1,1:spec_possible+1,:]+y[2,2:spec_possible+2,:]+y[3,3:spec_possible+3,:]
            out = cp.fft.rfft(y,axis=1)
            self.tsbuf[:self.overlap] = self.tsbuf[spec_size:spec_size+self.overlap].copy()
        self.rembuf[self.remptr : self.remptr + incoming - used] = timestream[used :].copy() #if spec_size 0, just loads the last rem of timestream = entire timestream
        self.remptr += incoming - used
        return out

    def __repr__(self):
        arrays = {
            "win": self.win,
            "tsbuf": self.tsbuf,
            "rembuf": self.rembuf
        }
        header = (
            f"StreamingPFB(nant={self.nant}, npol={self.npol}, "
            f"nblock={self.nblock}, lblock={self.lblock}, #chans={len(self.chans)})"
        )
        return _print_class_mem_usage(arrays, header)


class StreamingIPFB():
    def __init__(self, nant, npol, chans, nblock=100, lblock=4096, ntap=4, window='hamming', cut=10):
        self.lblock = lblock
        self.nblock = nblock
        self.nant = nant
        self.npol = npol
        self.read_size = nblock - 2*cut
        self.chans = chans
        N = self.lblock * ntap
        self.win = cp.__dict__[window](N) * cp.sinc((cp.arange(0, N) - N // 2) / self.lblock)
        self.cut = cut
        self.nchan=lblock//2+1
        self.specbuf = cp.empty((self.nant, self.npol, self.nblock, self.nchan), dtype='complex64')
        #mat will be deallocated after function returns
        mat=cp.zeros((nblock, lblock),dtype="float32")
        mat[:ntap,:]=cp.reshape(self.win,[ntap,len(self.win)//ntap])
        mat=mat.T.copy()
        print("mat shape", mat.shape)
        self.matft = pycufft.rfft(mat,axis=1)
        print("matft shape", self.matft.shape)
    
    def ipfb(self, antidx, polidx, spectra, thresh=0.):
        #input spectra, fill timestream
        assert spectra.shape[0]==self.read_size #incoming spectra is pfbsize - 2*cut
        self.specbuf[antidx,polidx,2*self.cut:, self.chans] = spectra
        dd=pycufft.irfft(self.specbuf[antidx, polidx , :, :],axis=1)
        assert dd.flags.c_contiguous and dd.base is None
        self.specbuf[antidx, polidx, :2*self.cut, self.chans] = spectra[-2*self.cut:, :] #copy the last two spectra back to buf
        dd2=dd.T.copy()
        ddft=pycufft.rfft(dd2,axis=1)
        if thresh>0.:
            # print("filtering...")
            filt=cp.abs(self.matft)**2/(thresh**2+cp.abs(self.matft)**2)*(1+thresh**2)
            ddft=ddft*filt
        # print("ddft c conti", ddft.flags.c_contiguous)
        # res = pycufft.irfft(ddft/cp.conj(self.matft),axis=0)
        out = np.fft.irfft(ddft/cp.conj(self.matft),axis=1)
        a=out[0,0]
        b=out[1,0]
        out = out.T[self.cut:-self.cut].ravel()
        assert len(out)==(self.nblock*self.lblock)
        assert out[0]==a and out[1]==b
        return out
    
    def __repr__(self):
        arrays = {
            "win": self.win,
            "specbuf": self.specbuf,
            "mat": getattr(self, "mat", None),
            "matft": self.matft,
        }
        header = (
            f"StreamingIPFB(nant={self.nant}, npol={self.npol}, "
            f"nblock={self.nblock}, lblock={self.lblock}, #chans={len(self.chans)}, cut={self.cut})"
        )
        return _print_class_mem_usage(arrays, header)
        

class StreamingCorrelator():
    def __init__(self, nant, npol, acclen, nchan, split=1):
        self.nant = nant
        self.npol = npol
        self.acclen = acclen
        self.nchan = nchan
        self.split = split
        self.out = cp.zeros((nant*npol, nant*npol, nchan*split), dtype="complex64", order="F")
        self.xin = cp.empty((nant*npol, acclen, nchan),dtype='complex64',order='F')
        self.buf = self.xin.copy()
        self.bufptr = 0
    
    def xcorr(self, antidx, polidx, data):
        incoming = data.shape[0]
        if self.bufptr == self.acclen:
            #buffer full. purge
            self.purge_buffer()
    
        #use buffer to xcorr
        self.xin[antidx*self.nant + polidx, : self.bufptr, :] = self.buf[antidx*self.nant + polidx, :self.bufptr, :]
        self.xin[antidx*self.nant + polidx, self.bufptr :, :] = data[ : self.acclen - self.bufptr, :]
        used = (self.acclen - self.bufptr)
        self.bufptr = 0
        print(f"incoming: {incoming}, used: {used}")
        self.buf[self.bufptr : self.bufptr + incoming - used] = data[ used : , :] #if spec_size 0, just loads the last rem of timestream = entire timestream
        self.bufptr += incoming - used
        
        cr.avg_xcorr_all_ant_gpu(self.xin, self.nant, self.npol, self.acclen, self.nchan, split=self.split, out=self.out)
        
        #if split > 1: sum over split time axis
    
    def purge_buffer(self):
        #purge whatever remains in the buffer.
        cr.avg_xcorr_all_ant_gpu(self.buf[:,:self.bufptr, :], self.nant, self.npol, self.bufptr, self.nchan, split=self.split, out=self.out)
        self.bufptr = 0
    
    def __repr__(self):
        arrays = {
            "buf": self.buf,
            "out": self.out,
            "xin": self.xin,
        }
        header = (
            f"StreamingCorrelator(nant={self.nant}, npol={self.npol}, "
            f"acclen={self.acclen}, nchan={self.nchan}, split={self.split})"
        )
        _print_class_mem_usage(arrays, header)

def print_mem():
    print("Mem stats")
    mempool = cp.get_default_memory_pool()
    print(f'USED: {mempool.used_bytes()/1024**2:5.2f} MB')
    print(f'FREE: {mempool.free_bytes()/1024**2:5.2f} MB')
    print(f'TOTAL: {mempool.total_bytes()/1024**2:5.2f} MB')

def cupy_ipfb(dat,matft,thresh=0.0):
    """On-device ipfb. Expects the data to be iPFB'd to live in GPU memory.

    Parameters
    ----------
    dat : cp.ndarray
        nspec x nchan array of complex64
    matft : cp.ndarray
        lblock x nspec array of float32. Note that this is transpose of Jon's original convention
        for speed reasons.
        [w0  wN  w2N  w3N  0  0  . . . ]
        [w1  .             0  0        ]
        [         .        . . .       ]
        [wN-1 . . .   w4N-1     .    0 ]
    thresh : float, optional
        Wiener filter threshold, by default 0.0

    Returns
    -------
    cp.ndarray
        nspec*nchan timestream values as a C-major matrix of shape (nspec x nchan)
    """
    # start_event = cp.cuda.Event()
    # end_event = cp.cuda.Event()
    # start_event.record()
    dd=pycufft.irfft(dat,axis=1)
    assert dd.flags.c_contiguous and dd.base is None
    dd2=dd.T.copy()
    ddft=pycufft.rfft(dd2,axis=1)
    if thresh>0:
        # print("filtering...")
        filt=cp.abs(matft)**2/(thresh**2+cp.abs(matft)**2)*(1+thresh**2)
        ddft=ddft*filt
    # print("ddft c conti", ddft.flags.c_contiguous)
    # res = pycufft.irfft(ddft/cp.conj(matft),axis=0)
    res = pycufft.irfft(ddft/cp.conj(matft),axis=1)
    res=res.T
    # end_event.record()
    # end_event.synchronize()
    # print("cupy ",cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    return res

def sinc_hamming(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hamming(ntap*lblock)*np.sinc(w/lblock)

def sinc_hanning(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hanning(ntap*lblock)*np.sinc(w/lblock)

def cupy_pfb(timestream, win, out=None,nchan=2049, ntap=4):
    lblock = 2*(nchan-1)
    nblock = timestream.size // lblock - (ntap - 1)
    timestream=timestream.reshape(-1,lblock)
    if out is not None:
        assert out.shape == (nblock, nchan)
    win=win.reshape(ntap,lblock)
    y=timestream*win[:,cp.newaxis]
    y=y[0,:nblock,:]+y[1,1:nblock+1,:]+y[2,2:nblock+2,:]+y[3,3:nblock+3,:]
    # out=cp.fft.rfft(y,axis=1)
    out=pycufft.rfft(y,axis=1)
    return out

def get_matft(nslice,nchan=2049,ntap=4):
    ntap=4
    nn=2*(nchan-1)
    dwin=sinc_hamming(ntap,nn)
    cupy_win=cp.asarray(dwin,dtype='float32',order='c')
    cupy_win=cp.reshape(cupy_win,[ntap,len(cupy_win)//ntap])
    mat=cp.zeros((nslice,nn),dtype='float32',order='c')
    mat[:ntap,:]=cupy_win
    print("mat size is", mat.shape, np.prod(mat.shape)*4/1024**3, "GB")
    mat=mat.T.copy()
    print("mat size is", mat.shape, np.prod(mat.shape)*4/1024**3, "GB")
    print("doing matft axis=1", mat.shape, mat.base is None, mat.flags.c_contiguous)
    matft=pycufft.rfft(mat,axis=1)
    # mat=None
    # matft=cp.fft.rfft(mat,axis=1)
    return matft


if __name__=="__main__":
    #export CUPY_CACHE_DIR=${PROJECT}/.cupy/kernel_cache
    sc = StreamingCorrelator(8, 2, 1024, 8000, split=1)
    ipfb = pu.StreamingIPFB(8, 2, chans=cp.arange(200), nblock=20000, lblock=4096, ntap=4)
    print(sc)
    print(ipfb)
