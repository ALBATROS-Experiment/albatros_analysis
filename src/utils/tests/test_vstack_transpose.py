import orbcomm_utils as outils
import numpy as np
import time as time

arr = np.zeros((3000000,4),dtype="complex128")
bigarr = np.empty((arr.shape[1], arr.shape[0]*2),dtype=arr.dtype)

arr[:,1]=1+1j
arr[:,2]=1+1j
arr[:,3]=1+1j


# niter=50
# t1=time.time()
# for i in range(niter):
#     outils.vstack_zeros_transpose(arr,bigarr)
# t2=time.time()
# print("time taken current:",(t2-t1)/niter)
# print(np.sum(bigarr))

# niter=50
# t1=time.time()
# for i in range(niter):  
#     bigarr2=np.transpose(np.vstack([arr,np.zeros(arr.shape)])).copy()
# t2=time.time()
# print("time taken numpy:",(t2-t1)/niter)
# # print(np.sum(bigarr))

niter=50
t1=time.time()
for i in range(niter):  
    outils.vstack_zeros_transpose2(arr,bigarr,np.asarray([1,2,3]))
t2=time.time()
print(np.sum(bigarr))
print("time taken new:",(t2-t1)/niter)
print(bigarr)