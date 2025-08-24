import pytest
from src.correlations import correlations as cr
from src import xp
import cupy as cp
import numpy as np
print(xp.__name__)
def test_all_ant_complex_xcorr1():
    nant = 1
    npol = 2
    ntime = 10000
    nfreq = 2
    x = cp.empty((nant*npol, ntime, nfreq), dtype='complex64', order='F')
    val = 1.37 + 1.37j #something that's not exactly representable in fp32
    x[:] = val
    x[:] = cp.random.randn(nant*npol*ntime*nfreq).reshape(nant*npol, ntime, nfreq) + 10j * cp.random.randn(nant*npol*ntime*nfreq).reshape(nant*npol, ntime, nfreq)
    # print(x)
    # print(xp.abs(val)**2)
    myout = xp.zeros((nant*npol, nant*npol, nfreq), dtype="complex64", order="F")
    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=1)
    print("Out shape", out.shape)
    myout[:]=out
    print(out-xp.abs(val)**2)
    print(out)
    assert xp.all(xp.isclose(out.astype('complex128'),xp.abs(val)**2,rtol=1e-4,atol=1e-8))
    #re-fill the array
    
    val = 1.37
    x[:]= val
    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=1)
    
    #Real should be 1.37^2, imag should be 0
    assert xp.all(xp.isclose(out.astype('complex128').real,xp.abs(val)**2,rtol=1e-4,atol=1e-8))
    assert xp.all(xp.isclose(out.astype('complex128').imag,0,rtol=1e-4,atol=1e-8))

def test_all_ant_complex_xcorr2():
    nant = 3
    npol = 1
    ntime = 10000
    nfreq = 2
    x = xp.empty((nant*npol, ntime, nfreq), dtype='complex64', order='F')
    x[0,:,0] = 3 #ant 1
    x[1,:,0] = 5 #ant 2
    x[2,:,0] = 7 #ant 3
    x[0,:,1] = 4 #ant 1
    x[1,:,1] = 6 #ant 2
    x[2,:,1] = 8 #ant 3
    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=16) #return shape is (nant*npol x nant*npol x nfreq)
    #dont print it all together because numpy printing scheme is to consider first axis as the axis into the screen.
    # print(out)
    # print(out[:,:,0]) #freq 1
    # print(out[:,:,1]) #freq 2
    # ri1,ci1=xp.tril_indices(nant*npol, k=0)
    # ri2,ci2=xp.triu_indices(nant*npol, k=0)
    # print(out[ri1,ci1,0])
    # print(out[ri2,ci2,0])
    # assert xp.all(out[:,:,0]==)
    assert out[0,0,0]==9
    assert out[0,1,0]==15
    assert out[0,2,0]==21
    assert out[1,2,0]==35
    assert out[0,0,1]==16
    assert out[0,1,1]==24
    assert out[0,2,1]==32
    assert out[1,2,1]==48
    assert xp.all(out[:,:,0].real.astype(int)%2==1) #all freq 1 is odd
    assert xp.all(out[:,:,0].imag.astype(int)==0)
    assert xp.all(out[:,:,1].real.astype(int)%2==0) #all freq 2 is even
    assert xp.all(out[:,:,1].imag.astype(int)==0)

def test_all_ant_complex_xcorr3():
    #test reduction of single antenna autocorr
    nant = 1
    npol = 1
    ntime = 10000
    nfreq = 2
    x = xp.empty((nant*npol, ntime, nfreq), dtype='complex64', order='F')
    x[0,:,0] = xp.random.randn(ntime) + 1j * xp.random.randn(ntime)
    x[0,:,1] = xp.random.randn(ntime) + 1j * xp.random.randn(ntime)
    #storing randn (float64) in x causes truncation. copy the actual x value stored.
    x1 = x[0,:,0].copy().astype('complex128') #freq1
    x2 = x[0,:,1].copy().astype('complex128') #freq2
    xcorr1=xp.mean(xp.abs(x1)**2)
    xcorr2=xp.mean(xp.abs(x2)**2)
    print("manual xcorr", xcorr1,xcorr2)

    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=16)
    print("cublas xcorr", out)
    assert xp.isclose(out[:,:,0].astype('complex128'),xcorr1,rtol=1e-4,atol=1e-8)
    assert xp.isclose(out[:,:,1].astype('complex128'),xcorr2,rtol=1e-4,atol=1e-8)
    # #re-fill the array
    # val = 1.37
    # x[:]= val
    # out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=16)
    # #Real should be 1.37^2, imag should be 0
    # assert xp.all(xp.isclose(out.astype('complex128').real,xp.abs(val)**2,rtol=1e-4,atol=1e-8))
    # assert xp.all(xp.isclose(out.astype('complex128').imag,0,rtol=1e-4,atol=1e-8))