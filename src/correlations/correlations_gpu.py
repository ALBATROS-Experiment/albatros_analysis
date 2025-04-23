import ctypes
import os
from .. import xp

# mylib = ctypes.cdll.LoadLibrary(
#     os.path.realpath(__file__ + r"/..") + "/lib_correlations_gpu.so"
# )
# Sxc = mylib.Sxc
# Sxc.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
# Cxc = mylib.Cxc
# Cxc.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
lib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/libcgemm_batch.so"
)
lib.cgemm_strided_batched.argtypes = [
    ctypes.c_void_p,  # A.ptr
    ctypes.c_void_p,  # B.ptr
    ctypes.c_void_p,  # C.ptr
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
    ctypes.c_int      # batchCount
]
lib.cgemm_strided_batched.restype = None
# def avg_xcorr_all_ant_gpu(x: xp.ndarray,nant: int,npol: int,ntime: int,nfreq: int,split : int = 1, scratch = None):
#     """Correlate all antenna-feed pairs for all times and frequencies.
#     Correlation implemented using CuBLAS Cgemm, thus input data must be column-major (Fortran ordered).

#     Parameters
#     ----------
#     x : cp.ndarray, complex64
#         F-ordered input data (nant*npol) x (ntime*nfreq).
#         Memory layout:
#         1st leading dimension: npol * nant
#         2nd leading dimension: ntime
#         3rd leading dimension: nfreq
#     nant : int
#         Number of antennas.
#     npol : int
#         Number of feed/polarizations per antenna.
#     ntime : int
#         Number of time steps to average over.
#     nfreq : int
#         Number of independent frequency bins in the data.
#     split : int, optional
#         Split the summing over time by a factor of `split`, by default 1.
#         Implemented to reduce the effect of floating-point error accumulation.
#         Necessary for averaging length > 10,000 samples.
#     out : cp.ndarray, optional
#         Output array to write to if passed.

#     Returns
#     -------
#     cp.ndarray, complex64
#         F-ordered, hermitian symmetric output matrix of size (nant*npols) x nfreq

#     Raises
#     ------
#     ValueError
#         If the `split` factor is not a divisor of ntime.
#     """
#     #sgemm/cgemm callsign (m,n,k, ldA, strideA, ldB, strideB, ldC, strideC, nbatch)
#     #A = m x k  | B = k x n  | C = m x n
#     m=nant*npol
#     n=m
#     nbatch=nfreq
#     k=ntime
#     if(k%split!=0):
#         raise ValueError("split should be a divisor of ntime")
#     # if scratch is None:
#     #     scratch = xp.empty((m,n,nbatch*split),dtype=x.dtype,order='F')
#     # else:
#     #     assert scratch.shape == (m,n,nbatch*split) and scratch.dtype == x.dtype and scratch.flags['F_CONTIGUOUS']
#     print("x flags", x.flags)
#     print(x.dtype)
#     print(x.shape)
#     scratch = xp.empty((m,n,nbatch*split),dtype='complex64',order='F')
#     print("scratch shape", scratch.shape)
#     print("m n k  python", m, n, k//split)
#     Cxc(scratch.data.ptr,x.data.ptr,x.data.ptr,m,n,k//split,nbatch*split)  # can trivially split up time blocks too
#     # print(scratch)
#     if split > 1:
#         scratch=scratch.reshape(m,n,split,nbatch,order='F') #reshaping is free of cost here
#         out=xp.sum(scratch,axis=2) #reduce along split time axis
#         print("reduced out", out.flags, out.shape)
#     else:
#         out=scratch
#     # out[:] = out/ntime
#     # print(scratch)
#     # scratch[:] = scratch/ntime
#     # out=out/ntime
#     print(scratch.flags)
#     return out

def avg_xcorr_all_ant_gpu(x: xp.ndarray, nant: int,npol: int, ntime: int, nfreq: int, split : int = 1, scratch = None, out=None):
    #sgemm/cgemm callsign (m,n,k, ldA, strideA, ldB, strideB, ldC, strideC, nbatch)
    #A = m x k  | B = n x k  | C = m x n when B transpose enabled
    M=nant*npol
    N=M
    K=ntime
    if(K%split!=0):
        raise ValueError("split should be a divisor of ntime")
    batchCount=nfreq*split

    if out is None:
        out = xp.empty((M,N,nfreq),dtype='complex64',order='F')
    elif (out.shape != (M, M, nfreq) or out.dtype != x.dtype or not out.flags.f_contiguous):
        raise ValueError("invalid out buffer")

    if split > 1:
        raise NotImplementedError() #do a proper buffered solution later.
        # scratch.reshape(m,n,split,nbatch,order='F').sum(axis=2,out=out) #reduce along split time axis
        # print("reduced out", out.flags, out.shape)
    else:
        lib.cgemm_strided_batched(
        ctypes.c_void_p(x.data.ptr),
        ctypes.c_void_p(x.data.ptr),
        ctypes.c_void_p(out.data.ptr),
        M, N, K//split, batchCount
    )
        out/=K
    return out