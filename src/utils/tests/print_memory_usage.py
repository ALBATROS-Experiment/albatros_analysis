import cupy as cp
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)

if __name__ == "__main__":

    nant = 8
    npol = 2
    osamp = 64
    chans = np.arange(0,200)
    #input to ipfb
    nblock = 10000
    ipfb = pu.StreamingIPFB(nant, npol, chans, nblock=nblock)
    tsize = ipfb.read_size * 4096 #size of available timestream after ipfb
    pfb = pu.StreamingPFB(nant, npol,chans=cp.arange(200),timestream_size = tsize, lblock = 4096*osamp)
    acclen = pfb.nblock
    xc = pu.StreamingCorrelator(nant, npol, acclen, len(chans)*osamp)
    
    print(ipfb)
    print(pfb)
    print(xc)
    pu.print_mem()