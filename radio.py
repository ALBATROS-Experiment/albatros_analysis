import pylab

import numpy as np
import scipy.linalg as la

import pfb_helper as pfb

# ntime = 2**11
# ta = np.linspace(0.0, ntime / 2048, ntime, endpoint=False)

# ts = np.sin(2*np.pi * ta * 122.0) + np.sin(2*np.pi * ta * 378.1 + 1.0)

# spec_pfb = pfb.pfb(ts, 17, ntap=4)

# pylab.imshow(np.abs(spec_pfb), aspect='auto', interpolation='nearest')
# pylab.show()



ntime = 2**20

# Generate a white noise timestream
ts = np.random.standard_normal(ntime)

# Perform the PFB
spec_pfb = pfb.pfb(ts, 65)

# Perform the inverse
rts = pfb.inverse_pfb(spec_pfb, 4)

pylab.plot((ts - rts.ravel())[1000:20000])
pylab.show()
