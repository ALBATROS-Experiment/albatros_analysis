import numpy as np
import sys
sys.path.insert(0, '/home/s/sievers/mohanagr/albatros_analysis/')
from src.utils import orbcomm_utils as outils
from src.utils import math_utils as mutils

with np.load('./sim_delay.npz') as f:
    p0_a1=f["f1"].copy()
    p0_a2=f["f2"].copy()
    chans=f["chans"]
    shift=f["shift"] #in units of timestream sample

# print(p0_a1.dtype, p0_a2.dtype)
# print("sum p0", np.sum(np.abs(p0_a1)))
# print("sum p1", np.sum(np.abs(p0_a2)))
# a = np.vstack([p0_a1, np.zeros(p0_a1.shape, dtype="complex128")])
# b = np.vstack([p0_a2, np.zeros(p0_a2.shape, dtype="complex128")])
# a = a.T.copy()
# b = b.T.copy()
# print("sum a", np.sum(np.abs(a)))
# print("sum b", np.sum(np.abs(b)))
# # print(a,"\n",b)
# a1 = mutils.transpose_zero_pad(p0_a1)
# b1 = mutils.transpose_zero_pad(p0_a2)
# print("sum a1", np.sum(np.abs(a1)))
# print("sum b1", np.sum(np.abs(b1)))
# print(a1-a)
# exit(1)
# print(p0_a1.dtype)
osamp = 40
shift = shift*osamp
print("shift is ", shift, "chans are ", chans)
dN_interp1 = 100 #stamp size for calculating upsampled xcorr.
dN_interp2 = 20 #stamp size for storing upsampled xcorr.
sample_no = np.arange(0, 2*dN_interp1 * 4096 * osamp, dtype="float64")
coarse_sample_no = np.arange(0, 2 * dN_interp1, dtype="float64") * 4096 * osamp
UPSAMP_CENTER = dN_interp1*4096*osamp
UPSAMP_dN = dN_interp2*4096*osamp

xcorrtime2 = outils.get_coarse_xcorr(p0_a1,p0_a2)
xcorrtime2 = np.fft.fftshift(xcorrtime2, axes=1)
N=xcorrtime2.shape[1]
print(2*xcorrtime2[0,N//2]/N/4096)
xcorr_stamp = outils.get_coarse_xcorr_fast2(p0_a1, p0_a2, 2*dN_interp1)
print(xcorr_stamp[0,2*dN_interp1])
final_xcorr = np.zeros(2*dN_interp2* 4096 * osamp, dtype="complex128")
interp_xcorr = np.zeros((len(chans), len(sample_no)), dtype="complex128")
COARSE_CENTER = xcorr_stamp.shape[1]//2
for i, chan in enumerate(chans):
        shifted_samples = sample_no-shift
        outils.get_interp_xcorr_fast(
            xcorr_stamp[i,COARSE_CENTER-dN_interp1:COARSE_CENTER+dN_interp1],
            chan,
            shifted_samples,
            coarse_sample_no,
            out=interp_xcorr[i,:]
        )
ff = np.sum(interp_xcorr,axis=0)
print("len sample no//2", len(sample_no)//2)
m = np.argmax(np.real(ff))
print("argmax at: ", m)
print("value at peak:", ff[m])