import numpy as np

x = np.random.randn(21).reshape(21,1)
x = x@x.T


crap = np.ones((50,50))
get_avg(x, crap)
print(crap[:15,:25])
