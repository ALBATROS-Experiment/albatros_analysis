import ctypes
import os
from .. import xp

mylib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/lib_correlations_gpu.so"
)
Sxc = mylib.Sxc
Sxc.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
Cxc = mylib.Cxc
Cxc.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)

def avg_xcorr_all_ant_gpu(x: xp.ndarray,nant: int,npol: int,ntime: int,nfreq: int,split : int = 1, scratch = None, out = None):
    """Correlate all antenna-feed pairs for all times and frequencies.
    Correlation implemented using CuBLAS Cgemm, thus input data must be column-major (Fortran ordered).

    Parameters
    ----------
    x : cp.ndarray, complex64
        F-ordered input data (nant*npol) x (ntime*nfreq).
        Memory layout:
        1st leading dimension: npol * nant
        2nd leading dimension: ntime
        3rd leading dimension: nfreq
    nant : int
        Number of antennas.
    npol : int
        Number of feed/polarizations per antenna.
    ntime : int
        Number of time steps to average over.
    nfreq : int
        Number of independent frequency bins in the data.
    split : int, optional
        Split the summing over time by a factor of `split`, by default 1.
        Implemented to reduce the effect of floating-point error accumulation.
        Necessary for averaging length > 10,000 samples.
    out : cp.ndarray, optional
        Output array to write to if passed.

    Returns
    -------
    cp.ndarray, complex64
        F-ordered, hermitian symmetric output matrix of size (nant*npols) x nfreq

    Raises
    ------
    ValueError
        If the `split` factor is not a divisor of ntime.
    """
    #sgemm/cgemm callsign (m,n,k, ldA, strideA, ldB, strideB, ldC, strideC, nbatch)
    #A = m x k  | B = k x n  | C = m x n
    m=nant*npol
    n=m
    nbatch=nfreq
    k=ntime
    if(k%split!=0):
        raise ValueError("split should be a divisor of ntime")
    if scratch is None:
        scratch = xp.empty((m,n,nbatch*split),dtype=x.dtype,order='F')
    else:
        assert scratch.shape == (m,n,nbatch*split) and scratch.dtype == x.dtype and scratch.flags['F_CONTIGUOUS']
    Cxc(scratch.data.ptr,x.data.ptr,x.data.ptr,m,n,k//split,nbatch*split)  # can trivially split up time blocks too
    if split > 1:
        out=out.reshape(m,n,split,nbatch,order='F') #reshaping is free of cost here
        out=xp.sum(out,axis=2) #reduce along split time axis
    out=out/ntime
    # print(out)
    return out
    