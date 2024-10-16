import cupy as cp
import numpy as np
import time

def split_sum(arr):
    arrv=arr.reshape(arr.shape[0]//1000,1000,arr.shape[1])
    return arrv.sum(axis=0).sum(axis=0)

def trans_sum(arr):
    arrv = arr.T.copy()
    return cp.sum(arrv, axis=1)

nr=1000000
x = cp.ones((1000000,1000),dtype='complex64')*1.37 + 1j*cp.ones((1000000,1000),dtype='complex64')*1.37

# y = np.ones((2,1000000),dtype='complex64')*1.37 + 1j*np.ones((2, 1000000),dtype='complex64')*1.37
# xd = x.astype("complex128")

print("float", cp.sum(x,axis=0))
print("reshape", split_sum(x))
print("transpose", trans_sum(x))
# print("double", cp.sum(xd,axis=0))

# niter=100
# for i in range(niter):
#     t1=time.time()
#     cp.sum(x,axis=0)
#     t2=time.time()
#     print("cp sum time", t2-t1)

# for i in range(niter):
#     t1=time.time()
#     split_sum(x)
#     t2=time.time()
#     print("split sum time", t2-t1)

# for i in range(niter):
#     t1=time.time()
#     trans_sum(x)
#     t2=time.time()
#     print("trans sum time", t2-t1)
