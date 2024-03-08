import numpy as np
import time
import numba as nb
from scipy.interpolate import CubicSpline, splrep, splev
import os
import ctypes
mylib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/libmath.so"
)
ctrans_c = mylib.ctrans
ctrans_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int
]

def transpose(x):
    xT = np.empty((x.shape[1], x.shape[0]),dtype=x.dtype)
    ctrans_c(x.ctypes.data,xT.ctypes.data,x.shape[0],x.shape[1])
    return xT

@nb.njit(parallel=True)
def linear_interp(xnew,x,y):
    # beyond the boundaries, continuous the line.
    # differs from numpy behaviour that repeats the boundary element
    ynew = np.empty(len(xnew),dtype=xnew.dtype)
    dx = x[1]-x[0]
    x0 = x[0]
    m = len(x)
    n = len(ynew)
    for i in nb.prange(n):
        idx = min(max(int((xnew[i]- x0)//dx),0),m-2)
        ynew[i] = (y[idx+1]-y[idx])/(x[idx+1]-x[idx]) * (xnew[i] - x[idx]) + y[idx]
    return ynew

# nb.njit(parallel=True)
def spline_eval(xnew, ynew, x, coeffs):
    dx = x[1]-x[0]
    x0 = x[0]
    m = len(x)
    n = len(ynew)
    n_coeffs = coeffs.shape[1]
    for i in nb.prange(n):
        idx = min(max(int((xnew[i]- x0)//dx),0),m-2)
        xx = xnew[i]-x[idx]
        # print("xnew", xnew[i],"xx", xx, "x mine", x[idx], "idx", idx, int(xnew[i]))
        vv=xx
    #     # print(xx,xx**2,xx**3)
        # print("coeffs mine",coeffs[idx,:])
    #     # print("idx:",idx,"xnew[i]",xnew[i], "xx",xx)
        ynew[i] = coeffs[idx,0] #constant term
        for j in range(1,n_coeffs):
            # print("using xx=",vv)
            ynew[i] += vv * coeffs[idx,j]
            vv *= xx
    # for i in nb.prange(len(xnew)):
    #     ind=int(xnew[i])
    #     if(ind==9):
    #         print("breaking...")
    #         ind=8
    #     frac=xnew[i]-ind
    #     cc=coeffs[ind,:]
    #     tot=cc[0]
    #     xx=1
    #     for j in range(1,n_coeffs):
    #         xx=xx*frac
    #         tot=tot+cc[j]*xx
    #     ynew[i]=tot

def cubic_spline(xnew, x, y):
    order=3
    cs_obj = splrep(x,y)
    ynew = np.zeros(len(xnew),dtype=y.dtype)
    dx = x[1]-x[0]
    nn = int((x[-1]-x[0])*3/dx + 1)
    x1 = np.linspace(x[0],x[-1],nn)
    print("nn is", nn)
    print(len(x1),x1)
    # print(x1)
    # y1 = cs_obj(x1)
    y1 = splev(x1,cs_obj)
    # print(y1)
    sections = np.hstack([y1[:-1].reshape(-1,3),y[1:].reshape(-1,1)])
    # sections = y1[:-1].reshape(-1,4)
    # print(y1)
    # print(sections)
    # assert(1==0)
    A=np.zeros([order+1,order+1],dtype="float64")
    for i in range(order+1):
        A[:,i]=(x1[:(order+1)]-x1[0])**i
    # A = np.vander(x1[0:order+1]-x1[0],N=order+1,increasing=True) #map it to start from 0
    # print(x1[0:order+1]-x1[0])
    # print(A.T)
    coeffs = sections@np.linalg.inv(A.T)
    # print(coeffs.shape)
    spline_eval(xnew, ynew, x, coeffs)
    return ynew
    # print(coeffs)
    # print(x1)

