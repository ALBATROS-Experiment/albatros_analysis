import numpy as np
import cupy as cp
from . import pycufft
import time

def print_mem(str):
    print(str)
    mempool = cp.get_default_memory_pool()
    print('USED:',mempool.used_bytes())
    print('FREE:',mempool.free_bytes())    
    print('TOTL:',mempool.total_bytes())  

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
    dd2=dd.T.copy()
    print(f"about to execute R2C, {dd.shape, dd.dtype}. Is dd C contiguous", dd.flags.c_contiguous, dd.base is None)
    ddft=pycufft.rfft(dd2,axis=1)
    if thresh>0:
        print("filtering...")
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
    print("lblock is", lblock, "nchan is", nchan)
    print_mem("from pfb")
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