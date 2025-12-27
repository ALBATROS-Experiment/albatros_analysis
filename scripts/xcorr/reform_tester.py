import numpy as np
import cupy as cp
import pyuv_helper as ph
import importlib as il
import time
il.reload(ph)

#variables
nants = 7
npols = 2
nchans = 1000
nantpol = nants*npols
nbls = int(nants*(nants+1)/2)
npols_tot = 4

#set up data arrays
row_shape = (nantpol, nantpol, nchans)
row_new_shape = (nbls, nchans, npols_tot)
test_data = cp.zeros(row_shape)

#make simulated symmetric data for each channel
for chan_idx in range(nchans):
    d = cp.random.rand(nantpol, nantpol)
    d = cp.triu(d)               
    chan_data = np.round(d + np.triu(d, 1).T, 3)
    test_data[:, :, chan_idx] = chan_data
print("test data row shape", test_data.shape)
row_orig = cp.copy(test_data)

#extract all upper triangle antenna pairs
ai_gpu, aj_gpu = cp.triu_indices(nants)

tm_st = time.time()
row_reshaped = cp.reshape(test_data, (npols, nants, npols, nants, -1), order='F')
row_ut = row_reshaped[:, ai_gpu, :, aj_gpu, : ] #shape (nbl, npol, npol, nchan)
row_ut = row_ut.transpose(0, 3, 1, 2).reshape(nbls,-1, 4) #shape (nbl, nchan, npol*npol)
time_reform = time.time()-tm_st

# ======= some tests to verify correct indexing =======
#idx = antidx*npol + polidx
#baseline idx 2 = ant0ant2, pol00 ant0pol0 idx = 0, ant2pol0 idx = 4
#baseline idx 7 = ant1ant1, pol11 ant1pol1 idx = 3, ant1pol1 idx = 3
#baseline idx 8 = ant1ant2, pol01 ant1pol0 idx = 2, ant2pol1 idx = 5
#baseline idx 15 = ant2ant4, pol10 ant2pol1 idx = 5, ant4pol0 idx = 8
assert cp.all(row_ut[2,10:20,0] == row_orig[0,4,10:20] ) #passes
assert cp.all(row_ut[7,10:20,3] == row_orig[3,3,10:20] ) #passes
assert cp.all(row_ut[8,10:20,1] == row_orig[2,5,10:20] ) #passes 
assert cp.all(row_ut[15, :, 2] == row_orig[5, 8, :]) #passes
# (remember pols are also in F ordering post-flattening: 00,01,10,11)!!!!!
print("row_ut shape", row_ut.shape, row_ut.flags)
# ====================================================

print('final shape', row_ut.shape)
print('reform time', time_reform)

