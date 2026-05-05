import cupy as cp
import numpy as np
import time
import sys
import os

# Ensure the local packages are discoverable
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu

def benchmark_async(nant=8, npol=2, pfb_size=65536, nchan=200, osamp=64, nchunks=20):
    lblock = 4096
    cutsize = 16
    read_size = pfb_size - 2*cutsize
    timestream_size = read_size * lblock
    new_acclen = 1024
    new_nchan = nchan * osamp

    ipfb = pu.StreamingIPFB_IQ(nant, npol, np.arange(nchan), nblock=pfb_size, lblock=lblock, cut=cutsize)
    fpfb = pu.StreamingPFB(nant, npol, timestream_size=timestream_size, lblock=ipfb.lblock*osamp, dtype='complex64')
    xcorr = pu.StreamingCorrelator(nant, npol, new_acclen, np.arange(new_nchan))

    # dummy_input = cp.random.randn(read_size, nchan).astype(cp.complex64)
    dummy_input_host = np.random.randn(read_size, nchan).astype(cp.complex64)

    # Warm-up (Important to skip JIT/Planning time)
    for _ in range(2):
        for a in range(nant):
            for p in range(npol):
                dummy_input = cp.asarray(dummy_input_host,order='c',dtype='complex64')
                spec = ipfb.ipfb(a, p, dummy_input)
                pol_new = fpfb.pfb(a, p, spec)
                if pol_new is not None: xcorr.load(a, p, pol_new)
        _warmup_rows = xcorr.xcorr()
    cp.cuda.Device().synchronize()

    print(f"Starting Async Profile Run: {nchunks} chunks...")

    # START ACTUAL WORK
    # No internal synchronizes here!
    start_wall = time.perf_counter()
    
    for c in range(nchunks):
        cp.cuda.nvtx.RangePush(f"Chunk_{c}")
        for a in range(nant):
            for p in range(npol):
                cp.cuda.nvtx.RangePush(f"H2D")
                dummy_input = cp.asarray(dummy_input_host,order='c',dtype='complex64')   
                cp.cuda.nvtx.RangePop()

                cp.cuda.nvtx.RangePush(f"IPFB")
                spec = ipfb.ipfb(a, p, dummy_input)
                cp.cuda.nvtx.RangePop()

                cp.cuda.nvtx.RangePush(f"PFB")
                pol_new = fpfb.pfb(a, p, spec)
                cp.cuda.nvtx.RangePop()

                if pol_new is not None:
                    xcorr.load(a, p, pol_new)

        cp.cuda.nvtx.RangePush("XCorr")
        rows = xcorr.xcorr()
        cp.cuda.nvtx.RangePop()
        
        if len(rows) > 0:
            cp.cuda.nvtx.RangePush("D2H")
            for row in rows:
                res = cp.asnumpy(row)
            cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePop() # End Chunk

    # Final sync only at the very end to get total wall time
    cp.cuda.Device().synchronize()
    end_wall = time.perf_counter()
    
    print(f"Total Wall Time: {(end_wall - start_wall):.4f} seconds")

if __name__ == "__main__":
    benchmark_async()
