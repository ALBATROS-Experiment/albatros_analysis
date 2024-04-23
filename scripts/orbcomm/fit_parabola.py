import sys
import os
import time
sys.path.insert(0, "/home/s/sievers/mohanagr/albatros_analysis/")
from src.utils import mkfftw as mk
from src.utils import orbcomm_utils as outils
import numpy as np

fname="/scratch/s/sievers/mohanagr/debug_genph_1627453681_61_1713379589.npz"
# "/gpfs/fs0/scratch/s/sievers/mohanagr/debug_genph_1627453681_dN10_1713121911.npz"
with np.load(fname) as f:
    dat4=f['chunks']

print(dat4.shape)
dx=50
bigdat = np.hstack([dat4, np.zeros(dat4.shape, dtype=dat4.dtype)])
bigft = mk.many_fft_c2c_1d(bigdat,axis=1)
center2=dat4.shape[1]
n_avg = outils.get_weights(center2)
maxvals=[]
for i in range(0,dat4.shape[0]-1):
    t1=time.time()
    xcorr = outils.complex_mult_conj(bigft[i:i+1,:], bigft[i+1:i+2,:])
    xcorr1 = mk.many_fft_c2c_1d(xcorr,axis=1,backward=True)
    xc = outils.get_normalized_stamp(xcorr1, n_avg, 50, 1)
    t2=time.time()
    print(i)
    m=np.argmax(xc[0,:].real)
    params=np.polyfit(np.arange(10),xc[0,50-5:50+5].real,2)
    maxvals.append(-params[1]/params[0]/2)
    # maxvals.append(m)
    # print(f"max at {m}, taking {t2-t1}")
print(maxvals)





