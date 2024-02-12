import sys
import time
sys.path.insert(0, "/home/s/sievers/mohanagr/")
from albatros_analysis.src.utils import orbcomm_utils as outils
import numpy as np
nspec=4
nchans = 2
f1 = np.ones((nspec,nchans),dtype='complex128')
f2 = np.ones((nspec,nchans),dtype='complex128')
zeros = np.zeros((nspec,nchans),dtype='complex128')
bigf1 = np.vstack([f1,zeros])
bigf2 = np.vstack([f2,zeros])

ff1 = np.fft.fft(bigf1.T, axis=1)
ff2 = np.fft.fft(bigf2.T, axis=1)


cxcorr1 = outils.get_coarse_xcorr(f1,f2)
cxcorr2 = outils.get_coarse_xcorr_fast(f1,f2)

# print(cxcorr1,"\n\n\n",cxcorr2)

print("ft of big fx and their mult:")

print("bigf1 fx\n", ff1)
print("bigf2 fx\n", ff2)
print("conj mult\n",ff1*np.conj(ff2))
# print(np.abs(cxcorr1-cxcorr2))