import numpy as np
import cupy as cp
from . import pycufft
import time
from ..correlations import correlations_gpu
from scipy.fft import next_fast_len
correlation_func = correlations_gpu.avg_xcorr_all_ant_gpu

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

def as_slice_if_contiguous(idx):
    idx = cp.asarray(idx)
    if idx.ndim == 1 and len(idx) > 0:
        d = cp.diff(idx)
        if cp.all(d == 1):
            return slice(idx[0], idx[-1] + 1)
    return idx

class StreamingPFB():
    """
    Works for arbitrary timestream sizes. Can be less than lblock.
    """
    def __init__(self, nant, npol, timestream_size = 50000, lblock=4096, ntap=4, window='hamming', dtype='float'):
        if dtype=='float':
            self.dtype = 'float32'
            self.fft = pycufft.rfft
        else:
            self.dtype = 'complex64'
            self.fft = pycufft.fft
        self.nant = nant
        self.npol = npol
        self.nchan =lblock//2+1
        self.ntap = ntap
        self.lblock = lblock
        self.timestream_size = timestream_size
        N = self.lblock * ntap
        self.win = cp.__dict__[window](N) * cp.sinc((cp.arange(0, N) - N // 2) / self.lblock)
        self.win = self.win.astype(self.dtype)
        self.win = self.win.reshape(ntap, lblock)
        self.nblock = timestream_size // lblock
        self.rem = timestream_size % lblock
        self.overlap = (self.ntap - 1)*self.lblock
        # print(f"tssize: {timestream_size}\noverlap: {self.overlap}\nrem: {self.rem}\nnblock: {self.nblock}\nlblock: {self.lblock}\n")
        self.tsbuf = cp.zeros((self.nant, self.npol, self.nblock*self.lblock + self.overlap + self.lblock), dtype=self.dtype, order='C') #will be used to pfb, one extra lblock to accomodate spillover spectra
        self.rembuf = -1*cp.ones((self.nant, self.npol, self.lblock + self.rem), dtype=self.dtype, order='C') #little bit extra to accomodate rem spillover
        # print("rembuf size", self.rembuf.shape)
        self.remptr = cp.zeros((self.nant, self.npol), dtype='int32')
    
    def pfb(self, antidx, polidx, timestream):
        remptr = self.remptr[antidx, polidx]
        out=None
        incoming = len(timestream)
        used = 0
        total_available = remptr + incoming
        spec_possible = total_available // self.lblock
        spec_size = spec_possible * self.lblock
        if spec_possible > 0:
            self.tsbuf[antidx, polidx, self.overlap : self.overlap + remptr] = self.rembuf[antidx, polidx,  : remptr]
            self.tsbuf[antidx, polidx, self.overlap + remptr : self.overlap + spec_size] = timestream[ : spec_size - remptr]
            used = (spec_size - remptr)
            x = self.tsbuf[antidx, polidx,  : spec_size + self.overlap].reshape(-1, self.lblock)
            #onwards to pfb
            y = x * self.win[:,cp.newaxis,:]
            y = y[0,:spec_possible,:]+y[1,1:spec_possible+1,:]+y[2,2:spec_possible+2,:]+y[3,3:spec_possible+3,:]
            out = self.fft(y,axis=1)
            self.tsbuf[antidx, polidx, :self.overlap] = self.tsbuf[antidx, polidx, spec_size:spec_size+self.overlap].copy()
        self.rembuf[antidx, polidx,  : incoming - used] = timestream[used :].copy() #if spec_size 0, just loads the last rem of timestream = entire timestream
        self.remptr[antidx, polidx] = incoming - used
        # print("PFB OUT SHAPE", out.shape, out.flags)
        return out

    def __repr__(self):
        arrays = {
            "win": self.win,
            "tsbuf": self.tsbuf,
            "rembuf": self.rembuf
        }
        header = (
            f"StreamingPFB(nant={self.nant}, npol={self.npol}, "
            f"nblock={self.nblock}, lblock={self.lblock})"
        )
        return _print_class_mem_usage(arrays, header)


class StreamingIPFB():
    def __init__(self, nant, npol, channels, nblock=100, lblock=4096, ntap=4, window='hamming', cut=10):
        self.lblock = lblock
        self.nblock = nblock
        self.nant = nant
        self.npol = npol
        self.read_size = nblock - 2*cut
        self.channels = cp.asarray(channels, dtype='int32')
        self.chan_sel = as_slice_if_contiguous(self.channels)
        print("chan_select from IPFB", self.chan_sel)
        N = self.lblock * ntap
        self.win = cp.__dict__[window](N) * cp.sinc((cp.arange(0, N) - N // 2) / self.lblock)
        self.cut = cut
        self.nchan=lblock//2+1
        self.specbuf = cp.zeros((self.nant, self.npol, self.nblock, self.nchan), dtype='complex64',order='C')
        print("specbuf IPFB shape", self.specbuf.shape, "len input chans", len(self.channels))
        #mat will be deallocated after function returns
        mat=cp.zeros((nblock, lblock),dtype="float32")
        mat[:ntap,:]=cp.reshape(self.win,[ntap,len(self.win)//ntap])
        mat=mat.T.copy()
        # print("mat shape", mat.shape)
        self.matft = pycufft.rfft(mat,axis=1)
        # print("matft shape", self.matft.shape)
        # print("IPFB CHANNELS", self.channels)
        # print("Specbuf shape", self.specbuf.shape)
    
    def ipfb(self, antidx, polidx, spectra, thresh=0.):
        #input spectra, fill timestream
        # print("incoming spectra shape", spectra.shape)
        assert spectra.shape[0]==self.read_size #incoming spectra is pfbsize - 2*cut
        # print("specbuf slice shape", self.specbuf[antidx,polidx,2*self.cut:, :].shape,self.specbuf[antidx,polidx,2*self.cut:, :].flags )
        self.specbuf[antidx,polidx,2*self.cut:, :][:,self.chan_sel] = spectra
        dd=pycufft.irfft(self.specbuf[antidx, polidx , :, :],axis=1)
        assert dd.flags.c_contiguous and dd.base is None
        if self.cut > 0:
            self.specbuf[antidx, polidx, :2*self.cut, :][:, self.chan_sel] = spectra[-2*self.cut:, :] #copy the last two spectra back to buf
        dd2=dd.T.copy()
        ddft=pycufft.rfft(dd2,axis=1)
        #print("DDFT", ddft)
        if thresh>0.:
            # print("filtering...")
            filt=cp.abs(self.matft)**2/(thresh**2+cp.abs(self.matft)**2)*(1+thresh**2)
            ddft=ddft*filt
        # print("ddft c conti", ddft.flags.c_contiguous)
        # res = pycufft.irfft(ddft/cp.conj(self.matft),axis=0)
        out = pycufft.irfft(ddft/cp.conj(self.matft),axis=1)
        #print("out.shape", out.shape, out.flags)
        #print(out)
        # a=out[0,self.cut].copy()
        # b=out[1,self.cut].copy()
        # if self.cut>0:
        #     out = out.T[self.cut:-self.cut].ravel()
        # else:
        #     out = out.T.ravel()
        out = out.T[self.cut:-self.cut].ravel()
        # print("out after ravel", out, out.shape)
        # assert len(out)==((self.nblock-2*self.cut)*self.lblock)
        # assert out[0]==a and out[1]==b
        # print("IPFB OUT SHAPE", out.shape, out.flags)
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
            f"nblock={self.nblock}, lblock={self.lblock}, #channels={len(self.channels)}, cut={self.cut})"
        )
        return _print_class_mem_usage(arrays, header)

class StreamingIPFB_IQ():
    def __init__(self, nant, npol, channels, nblock=100, lblock=4096, ntap=4, window='hamming', cut=10):
        #no need to pass lblock. we assign it
        self.nblock = nblock
        self.nant = nant
        self.npol = npol
        self.read_size = nblock - 2*cut
        self.channels = cp.asarray(channels, dtype='int32')
        # self.lblock = 2* 2**(int(np.log2(len(channels))) + 1) # closest multiple of 2 times 2 (for nyquist)
        self.lblock = next_fast_len(2 *len(channels)+1) #rely on scipy for fast lengths
        N = self.lblock * ntap
        self.win = cp.__dict__[window](N) * cp.sinc((cp.arange(0, N) - N // 2) / self.lblock)
        self.cut = cut
        self.nchan_input = len(self.channels)
        self.specbuf = cp.zeros((self.nant, self.npol, self.nblock, self.lblock), dtype='complex64',order='C')
        print("specbuf IPFB IQ shape", self.specbuf.shape, "len input chans", self.nchan_input)
        #mat will be deallocated after function returns
        mat=cp.zeros((self.nblock, self.lblock),dtype="float32")
        mat[:ntap,:]=cp.reshape(self.win,[ntap,len(self.win)//ntap])
        mat=mat.T.copy()
        mat=mat.astype("complex64")
        self.matft = pycufft.fft(mat,axis=1)
    
    def ipfb(self, antidx, polidx, spectra, thresh=0.):
        #input spectra, fill timestream
        assert spectra.shape[0]==self.read_size #incoming spectra is pfbsize - 2*cut
        self.specbuf[antidx,polidx,2*self.cut:, :][:,:self.nchan_input] = spectra
        dd=pycufft.ifft(self.specbuf[antidx, polidx , :, :],axis=1)
        assert dd.flags.c_contiguous and dd.base is None
        if self.cut > 0:
            self.specbuf[antidx, polidx, :2*self.cut, :][:, :self.nchan_input] = spectra[-2*self.cut:, :] #copy the last two spectra back to buf
        dd2=dd.T.copy()
        ddft=pycufft.fft(dd2,axis=1)
        #print("DDFT", ddft)
        if thresh>0.:
            # print("filtering...")
            filt=cp.abs(self.matft)**2/(thresh**2+cp.abs(self.matft)**2)*(1+thresh**2)
            ddft=ddft*filt
        out = pycufft.ifft(ddft/cp.conj(self.matft),axis=1)
        out = out.T[self.cut:-self.cut].ravel()
        return out
    
    def __repr__(self):
        arrays = {
            "win": self.win,
            "specbuf": self.specbuf,
            "mat": getattr(self, "mat", None),
            "matft": self.matft,
        }
        header = (
            f"StreamingIPFB_IQ(nant={self.nant}, npol={self.npol}, "
            f"nblock={self.nblock}, lblock={self.lblock}, #channels={len(self.channels)}, cut={self.cut})"
        )
        return _print_class_mem_usage(arrays, header)
        

class StreamingCorrelator():
    def __init__(self, nant, npol, acclen, channels, split=1, bufsize_frac = 1):
        self.nant = nant
        self.npol = npol
        self.acclen = acclen
        self.channels = cp.asarray(channels, dtype='int32')
        self.chan_sel = as_slice_if_contiguous(self.channels)
        print("chan_select from correlator", self.chan_sel)
        self.nchan = len(channels)
        self.split = split
        self.bufsize = int(bufsize_frac * acclen)

        self.inp = cp.zeros((nant*npol, self.bufsize, self.nchan), dtype="complex64", order="C") #incoming input size

        # self.out = cp.zeros((nant*npol, nant*npol, nchan*split), dtype="complex64", order="F")
        self.buf = cp.zeros((nant*npol, nant*npol, self.nchan*split), dtype='complex64',order='F') #intermediate accumulation buf
        self.bufptr = 0 #single buffer pointer since all antennas are processed simultaneously, unlike PFB remptr
        self.loaded_num = 0
        # print(f"acclen {self.acclen}")

    def load(self, antidx, polidx, data):
        # print("load")
        # print('data shape', data.shape)
        assert self.loaded_num < self.nant*self.npol #can't load if you havent purged previous data. NOT fool-proof. should use per ant,pol index
        if data.shape[0] > self.inp.shape[1]:
            raise RuntimeError("Input buffersize not big enough to store the incoming number of spectra.")
        self.incoming = data.shape[0]
        idx = antidx*self.npol + polidx
        self.inp[idx, :self.incoming, :] = data[:, self.chan_sel] #save relevant channels to input buffer
        self.loaded_num += 1
        
    def xcorr(self):
        # print("xcorr")
        assert self.loaded_num == self.nant*self.npol #can't xcorr if you havent loaded everything.
        chunks = []
        out_possible = (self.incoming + self.bufptr)//self.acclen
        # print("bufptr", self.bufptr, "incoming", self.incoming, "=> out possible", out_possible )
        used = 0
        if out_possible > 0:
            while out_possible:
                xin = cp.asfortranarray(self.inp[:,used:used+self.acclen-self.bufptr,:])
                out = correlation_func(xin, self.nant, self.npol, self.acclen-self.bufptr, self.nchan)
                if self.bufptr > 0:
                    out[:] += self.buf #add prev buffer
                    self.buf[:]=0
                out[:] /= self.acclen
                chunks.append(out)
                used += self.acclen-self.bufptr
                # print("used from input now", used)
                self.bufptr = 0
                out_possible -= 1
        if used < self.incoming:  #this was fine without if statement in PFB, but here we don't want to invoke xcorr func if nothing to xcorr
            xin = cp.asfortranarray(self.inp[:, used:self.incoming , :])
            self.buf += correlation_func(xin, self.nant, self.npol, self.incoming-used, self.nchan)
            self.bufptr += self.incoming - used #should be += but re-run unit tests
        self.loaded_num = 0 #ready to load again
        return chunks
    
    def __repr__(self):
        arrays = {
            "inp": self.inp,
            "buf": self.buf,
        }
        header = (
            f"StreamingCorrelator(nant={self.nant}, npol={self.npol}, "
            f"acclen={self.acclen}, nchan={self.nchan}, split={self.split})"
        )
        return _print_class_mem_usage(arrays, header)

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
    out=pycufft.rfft(y,axis=1)
    # print(pycufft.pycufft_cache)
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
