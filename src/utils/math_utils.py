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
ctrans_zero_c = mylib.ctrans_zero
ctrans_zero_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int
]

def add():
    pass

def subtract():
    pass

def conj():
    pass

def mult():
    pass

def hstack():
    pass

def vstack():
    pass

def exp():
    pass

def sum():
    pass

def transpose(x):
    if x.shape[0] < x.shape[1]:
        raise ValueError("rows>columns")
    xT = np.empty((x.shape[1], x.shape[0]),dtype=x.dtype)
    ctrans_c(x.ctypes.data,xT.ctypes.data,x.shape[0],x.shape[1])
    return xT

def transpose_zero_pad(x):
    if x.shape[0] < x.shape[1]:
        raise ValueError("rows>columns")
    xT = np.empty((x.shape[1], 2*x.shape[0]),dtype=x.dtype)
    ctrans_zero_c(x.ctypes.data,xT.ctypes.data,x.shape[0],x.shape[1])
    return xT

@nb.njit(parallel=True)
def vstack_zeros_transpose(x):
    Nrows = x.shape[0]
    Ncols = x.shape[1]
    bigx = np.empty((x.shape[1],x.shape[0]*2),dtype=x.dtype)
    for j in nb.prange(0, Ncols):
        for i in range(0, Nrows):
            bigx[j, i] = x[i, j]
        for i in range(Nrows, 2 * Nrows):
            bigx[j, i] = 0
    return bigx

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

@nb.njit(parallel=True)
def spline_eval(xnew, ynew, x, coeffs):
    dx = x[1]-x[0]
    x0 = x[0]
    m = len(x)
    n = len(ynew)
    n_coeffs = coeffs.shape[1]
    for i in nb.prange(n):
        idx = min(max(int((xnew[i]- x0)//dx),0),m-2)
        xx = xnew[i]-x[idx]
        vv=xx
        cc = coeffs[idx,:]
        tot = cc[0]
        # ynew[i] = coeffs[idx,0] #constant term
        for j in range(1,n_coeffs):
            # print("using xx=",vv)
            tot += vv * cc[j]
            vv *= xx
        ynew[i]=tot

def cubic_spline(xnew, x, y):
    order=3
    cs_obj = splrep(x,y)
    ynew = np.empty(len(xnew),dtype=y.dtype)
    dx = x[1]-x[0]
    nn = int((x[-1]-x[0])*3/dx + 1)
    x1 = np.linspace(x[0],x[-1],nn)
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

