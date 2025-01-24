import cupy as cp
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
from albatros_analysis.src.utils import orbcomm_utils_gpu as ou


f1 = cp.ones((16,2),dtype='complex64') + 1j * cp.ones((16,2),dtype='complex64')
f2=f1.copy()
dN=10
cxcorr=ou.coarse_xcorr(f1,f2,dN)
assert cp.allclose(cxcorr,2,atol=1.5e-7,rtol=1e-6)