import cupy as cp
import numpy as np
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
# from albatros_analysis.src.utils import pycufft_jon
from albatros_analysis.src.utils import pycufft as pycufft

# plan_cache=pycufft_jon.PlanCache()

def sinc_hamming(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hamming(ntap*lblock)*np.sinc(w/lblock)

pfb_size=5000
nchans=2049
# matft=pu.get_matft(pfb_size)
ntap=4
nn=2*(nchans-1)
dwin=sinc_hamming(ntap,nn)
cupy_win=cp.asarray(dwin,dtype='float32',order='c')
cupy_win=cp.reshape(cupy_win,[ntap,len(cupy_win)//ntap])
mat=cp.zeros((pfb_size,nn),dtype='float32',order='c')
mat[:ntap,:]=cupy_win.copy()
mat=mat.T.copy()
print("mat size is", mat.shape, np.prod(mat.shape)*4/1024**3, "GB")
print("doing matft axis=1", mat.shape, mat.base is None, mat.flags.c_contiguous)
# matft=pycufft_jon.rfft(mat,axis=1,plan_cache=plan_cache)
matft=pycufft.rfft(mat,axis=1)

mat = None # <----- need this, but have seen it happen without this too

for i in range(100):
    print("-------------------iter-----------------", i)
    xx=cp.random.randn(pfb_size, nchans) + 1j*cp.random.randn(pfb_size, nchans)
    xx=cp.asarray(xx,dtype='complex64')
    cudd=cp.fft.irfft(xx,axis=1)
    # dd=pycufft_jon.irfft(xx,axis=1,plan_cache=plan_cache)
    dd=pycufft.irfft(xx,axis=1)
    print("irfft error",cp.max(cp.abs(cudd-dd)))
    dd=dd.T.copy()
    assert dd.base is None and dd.flags.c_contiguous is True
    cuddft=cp.fft.rfft(dd,axis=1)
    # ddft=pycufft_jon.rfft(dd,axis=1,plan_cache=plan_cache)
    ddft=pycufft.rfft(dd,axis=1)
    print("rfft error",cp.max(cp.abs(cuddft-ddft)))
