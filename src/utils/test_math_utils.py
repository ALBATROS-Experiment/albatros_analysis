import numpy as np
import math_utils as mutils
from scipy.interpolate import CubicSpline,splrep, splev
import time
# x = np.arange(0,1000,dtype="float64")
x = np.linspace(-13,19,10)
xnew = np.linspace(0,18,101) #make sure points within the interp input
# x = np.linspace(-1,1,11)
# xnew = np.linspace(-1,1,101)
# y = x**3
y = 10*np.random.randn(len(x))

ynew1 = np.interp(xnew,x,y)
ynew2 = mutils.linear_interp(xnew,x,y)
print("linear error", np.std(ynew1-ynew2))
assert(np.allclose(ynew1,ynew2))

spl = splrep(x,y)
ynew1 = splev(xnew,spl)
ynew2 = mutils.cubic_spline(xnew,x,y)
print(np.where((ynew1-ynew2)>1e-10))
assert(np.allclose(ynew1,ynew2))
print("cubic error", np.std(ynew1-ynew2))

#speed test
# x = np.arange(0,100000000,dtype="float64")
x = np.linspace(0,100000000,101)
# xp = (x-x[0])/(x[-1]-x[0])
# print(xp)
xnew = np.linspace(0,99000000,32000000) #make sure points within the interp input
# xnewp = (xnew-x[0])/(x[-1]-x[0])
# x = np.linspace(-1,1,11)
# xnew = np.linspace(-1,1,101)
# y = x**3
y = 100*np.random.randn(len(x))
spl = splrep(x,y)
ynew1 = splev(xnew,spl)
ynew2 = mutils.cubic_spline(xnew,x,y)
print("cubic error", np.std(ynew1-ynew2))
niter= 10
tottime=0
for i in range(1):
    t1=time.time()
    spl = splrep(x,y)
    ynew1 = splev(xnew,spl)
    t2=time.time()
    # print("scipy", t2-t1)
    tottime+=t2-t1
print("avg time scipy", tottime)

tottime=0
for i in range(niter):
    t1=time.time()
    ynew2 = mutils.cubic_spline(xnew,x,y)
    t2=time.time()
    # print("mine", t2-t1)
    tottime+=t2-t1
print("avg time mine", tottime/niter)

assert(np.allclose(ynew1,ynew2))
print("cubic error", np.std(ynew1-ynew2))