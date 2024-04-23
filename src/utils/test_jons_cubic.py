import numpy as np
from scipy import interpolate
import numba as nb
import time
def fit_spline(spl):
    order=spl[2]
    # print('order is ',order)
    nx=len(spl[0])-order-1
    print("nx is", nx)
    # x=np.linspace(0,nx,(order+1)*nx+1) #Jon's default
    x=np.linspace(0,nx,(order)*nx+1)
    print(x)
    y=interpolate.splev(x,spl)
    # ymat=np.reshape(y[:-1],[nx,order+1]) #Jon's default
    ymat = np.hstack([y[:-1].reshape(-1,3),y[3::3].reshape(-1,1)])
    print(y)
    print("ymat is", ymat)
    # print(x[:12])
    # print('nx is ',nx)
    mat=np.zeros([order+1,order+1])
    for i in range(order+1):
        mat[:,i]=x[:(order+1)]**i
    print("mat.T is\n", mat.T)
    coeffs=ymat@np.linalg.inv(mat.T)
    
    return coeffs

@nb.njit(parallel=True)
def csplev(x,coeffs):
    out=0*x
    order=coeffs.shape[1]-1
    for i in nb.prange(len(x)):
        ind=int(x[i])
        # if(ind==9):
        #     print("breaking...")
        #     ind=8
        frac=x[i]-ind
        cc=coeffs[ind,:]
        tot=cc[0]
        xx=1
        for j in range(1,order+1):
            xx=xx*frac
            tot=tot+cc[j]*xx
        out[i]=tot
    return out

def temp_func(x,coeffs):
    a=1
    b=2
    c=3
    o=csplev(x,coeffs)
    return o

n=10
# y=np.random.randn(n)
# spl=interpolate.splrep(np.arange(n),y)
x = np.arange(n)
y = x**3
spl=interpolate.splrep(x,y)
coeffs=fit_spline(spl)
print("Coeffs shape", coeffs.shape)
# print("coeffs are", coeffs)
# x=np.random.rand(1000)*n
xnew = np.linspace(0,9,100)
# print("xnew :", xnew)
y1=interpolate.splev(xnew,spl)
y2=csplev(xnew,coeffs)

print("error",np.std(y2-y1))
m=int(32e6)
x=np.random.rand(m)*n
t1=time.time()
y1=interpolate.splev(x,spl)
t2=time.time()
print('splev: ',t2-t1)

for i in range(20):
    t1=time.time()
    y2=csplev(x,coeffs)
    t2=time.time()
    print('csplev: ',t2-t1)