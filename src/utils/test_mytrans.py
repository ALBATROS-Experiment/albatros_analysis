import math_utils as mutils
import numpy as np
import time
# nr=100000
# nc=40
# xr=50*np.random.randn(nr*nc).reshape(nr,nc)
# xi=50*np.random.randn(nr*nc).reshape(nr,nc)
# x=xr+1j*xi
# xT=mutils.transpose(x)
# assert(np.allclose(x.T,xT))

# nr=1000
# nc=20
# xr=50*np.random.randn(nr*nc).reshape(nr,nc)
# xi=50*np.random.randn(nr*nc).reshape(nr,nc)
# x=xr+1j*xi
# xT=mutils.transpose(x)
# # print(x,"\n", xT)
# x1=np.vstack([x,np.zeros(x.shape, dtype=x.dtype)])
# assert(np.allclose(x.T,xT))
# xT2=mutils.transpose_zero_pad(x)
# assert(np.allclose(x1.T, xT2))
# xT3=mutils.vstack_zeros_transpose(x)
# assert(np.allclose(x1.T, xT3))

# non integer block
nr=19994
nc=2
xr=50*np.random.randn(nr*nc).reshape(nr,nc)
xi=50*np.random.randn(nr*nc).reshape(nr,nc)
x=xr+1j*xi
xT=mutils.transpose(x)
# print(x.T,"\n\n", xT)
print(x.dtype)
x1=np.vstack([x,np.zeros(x.shape, dtype=x.dtype)])
assert(np.allclose(x.T,xT))
xT2=mutils.transpose_zero_pad(x)
print(xT2-x1.T)
assert(np.allclose(x1.T, xT2))
xT3=mutils.vstack_zeros_transpose(x)
assert(np.allclose(x1.T, xT3))

# #speed test for transpose + zero padding
# print("setting up the arrays...")
# nr=3000000
# nc=18
# xr=np.random.randn(nr*nc).reshape(nr,nc)
# xi=np.random.randn(nr*nc).reshape(nr,nc)
# x=xr+1j*xi
# # x = np.ones((nr,nc),dtype="complex128")
# # xT=mutils.vstack_zeros_transpose(x)
# niter=25
# # tottime=0
# # for i in range(niter):
# #     tt1=time.time()
# #     xT=mutils.vstack_zeros_transpose(x)
# #     tt2=time.time()
# #     tottime+=tt2-tt1
# #     print("numba vstack zeros exectime", tt2-tt1)
# # print(f"avg time taken for numba vstack zeros {tottime/niter:5.3f}s")

# tottime=0
# for i in range(niter):
#     tt1=time.time()
#     xT=mutils.transpose_zero_pad(x)
#     tt2=time.time()
#     tottime+=tt2-tt1
#     print("C vstack zeros exectime", tt2-tt1)
# print(f"avg time taken for C vstack zeros {tottime/niter:5.3f}s")


# #speed test
# print("setting up the arrays...")
# nr=1980000
# nc=44
# x = np.ones((nr,nc),dtype="complex128")
# # xr=np.random.randn(nr*nc).reshape(nr,nc)
# # xi=np.random.randn(nr*nc).reshape(nr,nc)
# # x=xr+1j*xi
# print(f"set up the arrays. starting test for ncols={nc}...")
# niter=100
# # tottime=0
# # for i in range(niter):
# #     tt1=time.time()
# #     xT1=x.T.copy()
# #     tt2=time.time()
# #     tottime+=tt2-tt1
# #     print("numpy exectime", tt2-tt1)
# # print(f"avg time taken for numpy {tottime/niter:5.3f}s")

# tottime=0
# for i in range(niter):
#     tt1=time.time()
#     xT2=mutils.transpose(x)
#     tt2=time.time()
#     tottime+=tt2-tt1
#     # print("numpy exectime", tt2-tt1)
# print(f"avg time taken for C {tottime/niter:5.3f}s")