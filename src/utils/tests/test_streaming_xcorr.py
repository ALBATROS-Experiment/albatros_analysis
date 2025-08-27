import cupy as cp
import os
import sys

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(42)

def test_xcorr_simple():
    niter = 10
    specsize = 11
    data = {
        0: {
            0: cp.ones((specsize * niter, 4), dtype="complex64"),
            1: 3 * cp.ones((specsize * niter, 4), dtype="complex64"),
        },
        1: {
            0: 2 * cp.ones((specsize * niter, 4), dtype="complex64"),
            1: 4 * cp.ones((specsize * niter, 4), dtype="complex64"),
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
        print(chunks[0][:,:,0])
        print(chunks[0][:,:,1])
        print(chunks[0][:,:,2])
        print(chunks[0][:,:,3])


if __name__ == "__main__":
    test_xcorr()
