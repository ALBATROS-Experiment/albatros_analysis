import numpy as np
import numba as nb
import time

@nb.njit(parallel=True)
def mytrans(a):
    n=a.shape[0]
    m=a.shape[1]
    out=np.empty((m,n),a.dtype)
    nthread=nb.get_num_threads()
    for ii in np.arange(nthread):
        tmp=np.empty((m,m),dtype=a.dtype)
        myid=nb.get_thread_id()
        for block in np.arange(ii*m,n,nthread*m):
            for i in range(m):
                for j in range(m):
                    tmp[j,i]=a[i+block,j]
                    #tmp[j,i]=a[i,j]
            for i in range(m):
                for j in range(m):
                    out[i,j+block]=tmp[i,j]
                
    return out,tmp

@nb.njit(parallel=True)
def mytrans3(a):
    n=a.shape[0]
    m=a.shape[1]
    out=np.empty((m,n),a.dtype)
    if m<n:
        for i in nb.prange(n):
            for j in np.arange(m):
                out[j,i]=a[i,j]
    else:
        for j in nb.prange(m):
            for i in np.arange(n):
                out[j,i]=a[i,j]
    return out

    
@nb.njit(parallel=True)
def mytrans2(a):
    n=a.shape[0]
    m=a.shape[1]
    out=np.empty((m,n),a.dtype)
    nthread=nb.get_num_threads()
    tmp=np.empty((nthread,m,m),dtype=a.dtype)
    for ii in np.arange(nthread):
        myid=nb.get_thread_id()
        for block in np.arange(ii*m,n,nthread*m):
            for i in range(m):
                for j in range(m):
                    tmp[ii,j,i]=a[i+block,j]
            for i in range(m):
                for j in range(m):
                    out[i,j+block]=tmp[ii,i,j]                
    return out

@nb.njit(parallel=True)
def mytrans2b(a):
    n=a.shape[0]
    m=a.shape[1]
    out=np.empty((m,n),a.dtype)
    nthread=nb.get_num_threads()
    nblock=n//m
    for block in nb.prange(nblock):
        ii=block*m
        tmp1=np.zeros((m,m),dtype=a.dtype)
        tmp2=np.zeros((m,m),dtype=a.dtype)
        for i in range(m):
            for j in range(m):
                tmp1[i,j]=a[ii+i,j]
        for i in range(m):
            for j in range(m):
                tmp2[i,j]=tmp1[j,i]
        for i in range(m):
            for j in range(m):
                out[i,ii+j]=tmp2[i,j]
    return out



nr=3000000
nc=20
xr=np.random.randn(nr*nc).reshape(nr,nc)
xi=np.random.randn(nr*nc).reshape(nr,nc)
x=xr+1j*xi
fwee=mytrans2b(x)
print('error is ',np.std(fwee-x.T))
for i in range(50):
    t1=time.time()
    fwee=mytrans2b(x)
    t2=time.time()
    fwee2=mytrans3(x)
    t3=time.time()
    fwee3=x.T.copy()
    t4=time.time()
    print(t2-t1,t3-t2,t4-t3)