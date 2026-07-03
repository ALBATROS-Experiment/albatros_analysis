import cupy as cp
import numpy as np
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu


with np.load("/home/mohanagr/Jupyter/PFB playground/pipeline_test_data_k=100_dk=10.npz") as f:
    spec0 = cp.asarray(f['spec0'])
    spec1 = cp.asarray(f['spec1'])
    start_chan = f['start_chan']

nchan = spec0.shape[1]
nrows = spec0.shape[0]

print(spec1.shape)
osamp = 64

pfb_size = 2000 #only have 10k spectra total
lblock = 4096
cutsize = 16
read_size = pfb_size - 2*cutsize
timestream_size = read_size * lblock
new_acclen = 1024
new_nchan = nchan * osamp

nant = 2
npol = 1

ipfb = pu.StreamingIPFB_IQ(nant, npol, np.arange(nchan), nblock=pfb_size, lblock=lblock, cut=cutsize)

fpfb = pu.StreamingPFB(nant, npol, timestream_size=timestream_size, lblock=ipfb.lblock*osamp, dtype='complex64')

nchunks = nrows // read_size
new_rows = nrows // osamp

new_spec0 = cp.zeros((new_rows, new_nchan), dtype='complex64')
new_spec1 = cp.zeros((new_rows, new_nchan), dtype='complex64')
print("new_spec0 shape",new_spec0.shape)
n = 0 # one accumulator is ok even though I'm using two antennas. it's just a sim. both antennas have same data.

for i in range(nchunks):
    ts0 = ipfb.ipfb(0,0,spec0[i*read_size:(i+1)*read_size, :], thresh = 0.05)
    ts1 = ipfb.ipfb(1,0,spec1[i*read_size:(i+1)*read_size, :], thresh = 0.05)
    
    s0 = fpfb.pfb(0, 0, ts0)
    s1 = fpfb.pfb(1, 0, ts1)
    assert s0.shape == s1.shape
    print(s0[-10,:])

    if s0 is not None and s1 is not None:
        print("start n is", n)
        nr = s0.shape[0]
        new_spec0[n : n + nr, :] =  s0[:, 0 : new_nchan]
        new_spec1[n : n + nr, :] =  s1[:, 0 : new_nchan]
        n += nr
print("final row counts", n)

np.savez("/home/mohanagr/Jupyter/PFB playground/pipeline_test_data_rechannelized_IQ_newPFB", new_spec0 = cp.asnumpy(new_spec0), new_spec1 = cp.asnumpy(new_spec1))
