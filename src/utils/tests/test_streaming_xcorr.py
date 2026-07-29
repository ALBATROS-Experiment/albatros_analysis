import cupy as cp
import os
import sys

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(42)

def dumb_xcorr(x,nant, npol, nfreq):
    truth = cp.zeros((nant*npol,nant*npol, nfreq),dtype='complex64',order='F')
    for freq in range(nfreq):
        for antidx1 in range(nant):
            for polidx1 in range(npol):
                for antidx2 in range(nant):
                    for polidx2 in range(npol):
                        # print(antidx1,polidx1,antidx2,polidx2)
                        idx1 = antidx1*npol + polidx1
                        idx2 = antidx2*npol + polidx2
                        truth[idx1 ,idx2, freq] = cp.mean(x[idx1,:,freq] * np.conj(x[idx2,:,freq]))
    return truth

def test_xcorr_simple():
    acclen = 10
    specsize = 8 #will pass to xcorr func in chunks of specsize
    acclen_big =  specsize * acclen
    niter = acclen_big//specsize
    nrows = acclen_big//acclen
    data = {
        0: {
            0: cp.ones((acclen_big, 4), dtype="complex64"),
            1: 3 * cp.ones((acclen_big, 4), dtype="complex64"),
        },
        1: {
            0: 2 * cp.ones((acclen_big, 4), dtype="complex64"),
            1: 4 * cp.ones((acclen_big, 4), dtype="complex64"),
        },
    }
    nant = 2
    npol = 2
    nchan = 4
    acclen = 10
    channels = np.arange(4)
    sc = pu.StreamingCorrelator(nant, npol, acclen, channels)
    for i in range(niter):
        print("---iteration----", i)
        for antidx in range(nant):
            for polidx in range(npol):
                spectra = data[antidx][polidx]
                sc.load(antidx, polidx, spectra[i * specsize : (i + 1) * specsize])
        chunks = sc.xcorr()
        print(f"Got {len(chunks)} chunks")
        if len(chunks) > 0:
            print(chunks[0][:,:,0])
            print(chunks[0][:,:,1])
            print(chunks[0][:,:,2])
            print(chunks[0][:,:,3])

def test_xcorr(specsize, bufsize_frac=1):
    acclen = 10
    specsize = specsize #will pass to xcorr func in chunks of specsize
    acclen_big =  specsize * acclen * 5
    niter = acclen_big//specsize
    nrows = acclen_big//acclen
<<<<<<< HEAD
=======
    print(f"acclen: {acclen} specsize: {specsize} acclen_big: {acclen_big} niter: {niter} nrows: {nrows}")
>>>>>>> 691fef5d7fa0e0ace00e9a1317533f35cda7656b
    nant = 8
    npol = 2
    nchan = 4
    out = cp.empty((nant*npol,nant*npol, nchan, nrows),dtype='complex64',order='F')
    truth = cp.empty((nant*npol,nant*npol, nchan, nrows),dtype='complex64',order='F')
    data = {}
    x = cp.empty((nant*npol,acclen_big,nchan),dtype='complex64',order='F') #full input data
    sc = pu.StreamingCorrelator(nant, npol, acclen, np.arange(nchan), bufsize_frac=bufsize_frac)
    for i in range(nant):
        temp = {}
        for j in range(npol):
            temp[j] = (cp.random.randn(acclen_big*nchan).reshape(-1,nchan) + 1j*cp.random.randn(acclen_big*nchan).reshape(-1,nchan)).astype('complex64')
            x[i*npol+j, :,:] = temp[j]
        data[i]=temp

<<<<<<< HEAD
    print("generating truth...")
    for i in range(nrows):
        truth[:,:,:,i] = dumb_xcorr(x[:,i*acclen:(i+1)*acclen,:],nant,npol,nchan)
    print("running streaming xcorr...")
    idx=0
    for i in range(niter):
        print("---iteration----", i)
=======
    # print("generating truth...")
    for i in range(nrows):
        truth[:,:,:,i] = dumb_xcorr(x[:,i*acclen:(i+1)*acclen,:],nant,npol,nchan)
    # print("running streaming xcorr...")
    idx=0
    for i in range(niter):
        # print("---iteration----", i)
>>>>>>> 691fef5d7fa0e0ace00e9a1317533f35cda7656b
        for antidx in range(nant):
            for polidx in range(npol):
                spectra = data[antidx][polidx] #fioll
                sc.load(antidx, polidx, spectra[i * specsize : (i + 1) * specsize])
        chunks = sc.xcorr()
        n = len(chunks)
<<<<<<< HEAD
=======
        # print("n is", n)
>>>>>>> 691fef5d7fa0e0ace00e9a1317533f35cda7656b
        if n > 0:
            for ch in chunks:
                out[:,:,:,idx] = ch
                idx+=1
    print("max error", cp.max(cp.abs(out-truth)))
    assert cp.allclose(out,truth,atol=1e-5,rtol=1e-6)


if __name__ == "__main__":
    # test_xcorr_simple()
<<<<<<< HEAD
=======
    test_xcorr(1, bufsize_frac=1)
    test_xcorr(6, bufsize_frac=10)
>>>>>>> 691fef5d7fa0e0ace00e9a1317533f35cda7656b
    test_xcorr(11,bufsize_frac=1.1)
    test_xcorr(9)
    test_xcorr(8)
    test_xcorr(25, bufsize_frac=10)
