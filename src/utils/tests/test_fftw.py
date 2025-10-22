import mkfftw as mk
import numpy as np
import time
from scipy import fft as sfft
import os
# nr=2
# nc=2
# xr=50*np.random.randn(nr*nc).reshape(nr,nc)
# xi=50*np.random.randn(nr*nc).reshape(nr,nc)
# x=xr+1j*xi
# # print("x is (axis=1 tests)", x)
# xf=np.fft.fft(x,axis=1)
# xf2=mk.many_fft_c2c_1d(x.copy(),axis=1) # have verified that x doesn't change after a transform
# x2 = mk.many_fft_c2c_1d(xf2.copy(),axis=1,backward=True)
# assert(np.allclose(xf,xf2))
# assert(np.allclose(x,x2))

# #works with context manager?
# with mk.parallelize_fft() as p:
#     xf3=mk.many_fft_c2c_1d(x,axis=1)
# assert(np.allclose(xf,xf3))

# # print("x is (axis=0 tests)", x)
# xf=np.fft.fft(x,axis=0)
# xf2=mk.many_fft_c2c_1d(x,axis=0)
# x2 = mk.many_fft_c2c_1d(xf2.copy(),axis=0,backward=True)
# assert(np.allclose(xf,xf2))
# assert(np.allclose(x,x2))

# #works with context manager?
# with mk.parallelize_fft() as p:
#     xf3=mk.many_fft_c2c_1d(x,axis=0)
# assert(np.allclose(xf,xf3))

# speed test

# axis = 1
niter=5
nr=18
nc=3000000
xr=np.random.randn(nr*nc).reshape(nr,nc)
xi=np.random.randn(nr*nc).reshape(nr,nc)
x=xr+1j*xi

print(f"starting speed test {x.shape[0]}x{x.shape[1]}, axis=1")
tottime=0
# for i in range(niter):
#     tt1=time.time()
#     xf = np.fft.fft(x,axis=1)
#     tt2=time.time()
#     tottime+=tt2-tt1
#     # print("numpy exectime", tt2-tt1)
# print(f"avg time taken for numpy axis=1 {tottime/niter:5.3f}s")
# tottime=0
# with mk.parallelize_fft():
xf2=mk.many_fft_c2c_1d(x,axis=1) # for the FFTW_MEASURE to do its thing first
t1=time.time()
for i in range(niter):
    tt1=time.time()
    xf2 = mk.many_fft_c2c_1d(x,axis=1)
    tt2=time.time()
    tottime+=tt2-tt1
    # print("FFTW exectime", tt2-tt1)
print(f"avg time taken for fftw axis=1 {tottime/niter:5.3f}s")

n_workers=os.cpu_count()
tottime=0
with sfft.set_workers(n_workers):
    xf3=sfft.fft(x,axis=1,workers=n_workers) # for the FFTW_MEASURE to do its thing first
    t1=time.time()
    for i in range(niter):
        tt1=time.time()
        xf3=sfft.fft(x,axis=1,workers=n_workers)
        tt2=time.time()
        tottime+=tt2-tt1
        # print("Scipy exectime", tt2-tt1)
    print(f"avg time taken for scipy axis=1 {tottime/niter:5.3f}s")

# axis = 0
print("-------------------------------------------------------")
x=x.T.copy()
print(f"starting speed test {x.shape[0]}x{x.shape[1]}, axis=0")
tottime=0
for i in range(niter):
    tt1=time.time()
    xf = np.fft.fft(x,axis=0)
    tt2=time.time()
    tottime+=tt2-tt1
    # print("numpy exectime", tt2-tt1)
print(f"avg time taken for numpy axis=0 {tottime/niter:5.3f}s")
tottime=0
with mk.parallelize_fft():
    xf2=mk.many_fft_c2c_1d(x,axis=0) # for the FFTW_MEASURE to do its thing first
    for i in range(niter):
        tt1=time.time()
        xf2 = mk.many_fft_c2c_1d(x,axis=0)
        tt2=time.time()
        tottime+=tt2-tt1
        # print("FFTW exectime", tt2-tt1)
    print(f"avg time taken for fftw axis=0 {tottime/niter:5.3f}s")

n_workers=os.cpu_count()
tottime=0
with sfft.set_workers(n_workers):
    xf3=sfft.fft(x,axis=0,workers=n_workers) # for the FFTW_MEASURE to do its thing first
    t1=time.time()
    for i in range(niter):
        tt1=time.time()
        xf3=sfft.fft(x,axis=0,workers=n_workers)
        tt2=time.time()
        tottime+=tt2-tt1
        # print("Scipy exectime", tt2-tt1)
    print(f"avg time taken for scipy axis=0 {tottime/niter:5.3f}s")


