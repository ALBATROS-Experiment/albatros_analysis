# import skyfield.api as sf
import time
from matplotlib import pyplot as plt
# from scipy.interpolate import CubicSpline
# import skyfield.api as sf
import os
import sys
# sys.path.insert(0,os.path.expanduser("~"))
import cupy as cp
import numpy as np
from albatros_analysis.src.utils import pycufft

def apply_delay(arr, delay, freqs, out=None):
    """Apply a time-dependent exponential phase to a timestream.
    Timestream can be E-field, or visibility, or anything else a user desires.
    Output = Input * exp(j 2 pi freq tau)

    Parameters
    ----------
    arr : cp.ndarray
        Usually nspec x nchan
    delay : 
        _description_
    freqs : _type_
        _description_
    out : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if out is None:
        out = cp.empty(arr.shape,dtype=arr.dtype)
    out[:] = arr * cp.exp(-2j * cp.pi * freqs[cp.newaxis,:]*delay[:,cp.newaxis])
    return out

def coarse_xcorr(f1, f2, dN, Npfb=4096,window=None):
    if len(f1.shape) == 1:
        f1 = f1.reshape(-1, 1)
    if len(f2.shape) == 1:
        f2 = f2.reshape(-1, 1)
    nchans = f1.shape[1]
    Nsmall = f1.shape[0]
    bigf1 = cp.zeros((nchans, 2 * Nsmall), dtype="complex64")
    bigf2 = cp.zeros((nchans, 2 * Nsmall), dtype="complex64")
    bigf1[:,:Nsmall] = f1.T
    bigf2[:,:Nsmall] = f2.T
    xcorr = pycufft.ifft(pycufft.fft(bigf1,axis=1) * cp.conj(pycufft.fft(bigf2,axis=1)), axis=1)
    norm = Nsmall-cp.abs(cp.arange(-dN, dN, dtype='float32'))
    return cp.fft.fftshift(xcorr, axes=1)[:,Nsmall-dN:Nsmall+dN].copy()/norm[cp.newaxis,:]

def median_abs_deviation(x,axis=1):
    med = cp.median(x,axis=axis)
    return cp.median(cp.abs(x-med[:, cp.newaxis]),axis=axis)