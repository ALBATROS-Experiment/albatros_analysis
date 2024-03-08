import math_utils as mutils
import numpy as np
import time
nr=1000
nc=4
xr=50*np.random.randn(nr*nc).reshape(nr,nc)
xi=50*np.random.randn(nr*nc).reshape(nr,nc)
x=xr+1j*xi
xT=mutils.transpose(x)
assert(np.allclose(x.T,xT))

#speed test

nr=3200000
nc=32
xr=np.random.randn(nr*nc).reshape(nr,nc)
xi=np.random.randn(nr*nc).reshape(nr,nc)
x=xr+1j*xi
niter=5
tottime=0
for i in range(niter):
    tt1=time.time()
    xT1=x.T.copy()
    tt2=time.time()
    tottime+=tt2-tt1
    print("numpy exectime", tt2-tt1)
print(f"avg time taken for numpy {tottime/niter:5.3f}s")

tottime=0
for i in range(niter+10000):
    tt1=time.time()
    xT2=mutils.transpose(x)
    tt2=time.time()
    tottime+=tt2-tt1
    print("numpy exectime", tt2-tt1)
print(f"avg time taken for C {tottime/niter:5.3f}s")