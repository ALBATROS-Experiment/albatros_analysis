import pytest
from src.correlations import correlations as cr
from src import xp
import cupy as cp
import numpy as np
print(xp.__name__)
def test_all_ant_complex_xcorr1():
    nant = 1
    npol = 2
    ntime = 5000
    nfreq = 2
    x = cp.empty((nant*npol, ntime, nfreq), dtype='complex64', order='F')
    val = 1.37 + 1.37j #something that's not exactly representable in fp32
    x[:] = val
    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=1, normalize=True)

    assert xp.all(xp.isclose(out.astype('complex128'),xp.abs(val)**2,rtol=1e-4,atol=1e-8))
    #re-fill the array
    
    val = 1.37
    x[:]= val
    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=1, normalize=True)
    
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
    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=1, normalize=True) #return shape is (nant*npol x nant*npol x nfreq)
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
    nant = 2
    npol = 2
    ntime = 100
    nfreq = 2
    timestream1 = xp.arange(1,101)/10
    timestream2 = timestream1 + 1
    timestream3 = timestream2 + 1
    timestream4 = timestream3 + 1
    x = xp.empty((nant*npol, ntime, nfreq), dtype='complex64', order='F')
    outarr = xp.zeros((nant*npol, nant*npol, nfreq, 2), dtype='complex64', order='F')

    #both pols for each antenna+freq are same
    x[0,:,0] = timestream1
    x[1,:,0] = timestream1
    x[0,:,1] = timestream2
    x[1,:,1] = timestream2
    x[2,:,0] = timestream3
    x[3,:,0] = timestream3
    x[2,:,1] = timestream4
    x[3,:,1] = timestream4

    a0pol00 = xp.mean(xp.abs(timestream1)**2)
    a0pol11 = xp.mean(xp.abs(timestream2)**2)

    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=1,out=outarr[:,:,:,1], normalize=True) #pass a slice of output
    assert xp.allclose(outarr[:,:,:,1],out)
    assert xp.allclose(outarr[:,:,:,0],0.) #make sure only the slice we passed is filled

    assert xp.allclose(out[:2,:2,0],xp.mean(xp.abs(timestream1)**2)) #freq 0 antenna 0, pol0 and pol1 all same
    assert xp.allclose(out[:2,:2,1],xp.mean(xp.abs(timestream2)**2)) #freq 1 antenna 0, pol0 and pol1 all same
    assert xp.allclose(out[2:,2:,0],xp.mean(xp.abs(timestream3)**2)) #freq 0 antenna 1, pol0 and pol1 all same
    assert xp.allclose(out[2:,2:,1],xp.mean(xp.abs(timestream4)**2)) #freq 1 antenna 1, pol0 and pol1 all same

    assert xp.allclose(out[:2,2:,0],xp.mean(timestream1*xp.conj(timestream3))) #freq 0 antenna 0 x antenna 1 pol0 and pol1 all same
    assert xp.allclose(out[:2,2:,1],xp.mean(timestream2*xp.conj(timestream4))) #freq 1 antenna 0 x antenna 1 pol0 and pol1 all same


    print("freq0\n",out[:,:,0])
    print("freq1\n",out[:,:,1])
    print("antenna self pols", xp.mean(xp.abs(timestream1)**2), xp.mean(xp.abs(timestream2)**2), xp.mean(xp.abs(timestream3)**2), xp.mean(xp.abs(timestream4)**2))
    print("antenna cross pols",xp.mean(timestream1*xp.conj(timestream3)), xp.mean(timestream2*xp.conj(timestream4)))

def test_all_ant_complex_xcorr4():
    nant = 7
    npol = 2
    ntime = 101 #all weird primes
    nfreq = 97
    x = xp.empty((nant*npol, ntime, nfreq), dtype='complex64', order='F')
    x[:] = cp.random.randn(nant*npol*ntime*nfreq).reshape(x.shape)
    truth = xp.zeros((nant*npol,nant*npol, nfreq),dtype='complex64')

    for freq in range(nfreq):
        for antidx1 in range(nant):
            for polidx1 in range(npol):
                for antidx2 in range(nant):
                    for polidx2 in range(npol):
                        # print(antidx1,polidx1,antidx2,polidx2)
                        idx1 = antidx1*npol + polidx1
                        idx2 = antidx2*npol + polidx2
                        truth[idx1 ,idx2, freq] = cp.mean(x[idx1,:,freq] * np.conj(x[idx2,:,freq]))

    out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=1, normalize=True)

    # mismatch_mask = ~cp.isclose(out, truth)
    # print("out values at mismatches:", out[mismatch_mask])
    # print("truth values at mismatches:", truth[mismatch_mask])
    # print("Differences:", out[mismatch_mask] - truth[mismatch_mask])
    # print("Tolerance:", 1e-8 + 1e-5*cp.abs(truth[mismatch_mask]))
    assert cp.allclose(out,truth,rtol=1e-5,atol=1e-6)


# def test_all_ant_complex_xcorr3():
#     #test reduction of single antenna autocorr
#     nant = 1
#     npol = 1
#     ntime = 10000
#     nfreq = 2
#     x = xp.empty((nant*npol, ntime, nfreq), dtype='complex64', order='F')
#     x[0,:,0] = xp.random.randn(ntime) + 1j * xp.random.randn(ntime)
#     x[0,:,1] = xp.random.randn(ntime) + 1j * xp.random.randn(ntime)
#     #storing randn (float64) in x causes truncation. copy the actual x value stored.
#     x1 = x[0,:,0].copy().astype('complex128') #freq1
#     x2 = x[0,:,1].copy().astype('complex128') #freq2
#     xcorr1=xp.mean(xp.abs(x1)**2)
#     xcorr2=xp.mean(xp.abs(x2)**2)
#     print("manual xcorr", xcorr1,xcorr2)

#     out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=16)
#     print("cublas xcorr", out)
#     assert xp.isclose(out[:,:,0].astype('complex128'),xcorr1,rtol=1e-4,atol=1e-8)
#     assert xp.isclose(out[:,:,1].astype('complex128'),xcorr2,rtol=1e-4,atol=1e-8)
#     # #re-fill the array
#     # val = 1.37
#     # x[:]= val
#     # out=cr.avg_xcorr_all_ant_gpu(x,nant,npol,ntime,nfreq,split=16)
#     # #Real should be 1.37^2, imag should be 0
#     # assert xp.all(xp.isclose(out.astype('complex128').real,xp.abs(val)**2,rtol=1e-4,atol=1e-8))
#     # assert xp.all(xp.isclose(out.astype('complex128').imag,0,rtol=1e-4,atol=1e-8))