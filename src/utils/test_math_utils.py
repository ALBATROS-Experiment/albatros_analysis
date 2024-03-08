import numpy as np
import math_utils as mutils
from scipy.interpolate import CubicSpline,splrep, splev

x = np.arange(0,10,dtype="float64")
# x = np.linspace(-1,1,11)
y = x**3
print("x original", x)
# xnew = np.linspace(-1,1,101)
xnew = np.linspace(0,9,101) #make sure points within the interp input
# ynew1 = np.interp(xnew,x,y)
# ynew2 = mutils.linear_interp(xnew,x,y)
# assert(np.allclose(ynew1,ynew2))
# cs = CubicSpline(x,y)
# ynew1 = cs(xnew)

spl = splrep(x,y)
ynew1 = splev(xnew,spl)
ynew2 = mutils.cubic_spline(xnew,x,y)
print(xnew)
print(ynew1[-1]-y[-1])
print(ynew2[-1]-y[-1])
print("error", np.std(ynew1-ynew2))