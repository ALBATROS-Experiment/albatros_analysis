from src.correlations import baseband_data_classes as bdc
import os
from src import xp
from src.correlations import correlations as cr
import numpy as np

def test_specnum_wrap():
    files=['/gpfs/fs1/home/s/sievers/mohanagr/albatros_analysis/src/correlations/tests/data/specnum_wrap1.raw',
     '/gpfs/fs1/home/s/sievers/mohanagr/albatros_analysis/src/correlations/tests/data/specnum_wrap2.raw',
     '/gpfs/fs1/home/s/sievers/mohanagr/albatros_analysis/src/correlations/tests/data/specnum_wrap3.raw']
    start_file_num=0
    acclen=10
    idxstart=0
    ant1=bdc.BasebandFileIterator(files,start_file_num,idxstart,acclen,type='float')
    spec_start=2**32-10
    data=ant1.__next__() #first 10
    print(type(data['pol0']), type(data['pol0']))
    print("First 10", data)
    assert ant1.obj._overflowed==True
    assert len(data['specnums'])==10
    assert data['specnums'][-1]==spec_start+acclen-1

    ########## xcorr tests ###########
    xc, nr = cr.avg_xcorr_4bit_float_gpu(data['pol0'], data['pol1'], data['specnums'], data['specnums']) # should be 1
    assert nr==10
    assert xp.allclose(xc,1,atol=1e-7)
    xc, nr = cr.avg_xcorr_4bit_float_gpu(data['pol0'], data['pol0'], data['specnums'], data['specnums']) # should be 1
    assert nr==10
    assert xp.allclose(xc,1,atol=1e-7)
    xc, nr = cr.avg_xcorr_4bit_float_gpu(data['pol0'][:,0:1], data['pol0'][:,1:], data['specnums'], data['specnums']) # should be 1j
    assert nr==10
    assert xp.allclose(xc,1j,atol=1e-7)
    ###################################

    data=ant1.__next__() #first 20
    print("First 20", data)
    assert len(data['specnums'])==5 #5 missing
    assert xp.allclose(data['pol0'][:5,0]+data['pol0'][:5,1],1+1j,atol=1e-7)
    assert xp.allclose(data['pol0'][:5,0]+data['pol1'][:5,1],1+1j,atol=1e-7)
    assert xp.allclose(data['pol0'][5:],0,atol=1e-7) #should be zero since it wasn't filled
    assert xp.allclose(data['pol1'][5:],0,atol=1e-7)
    assert data['specnums'][-1]==spec_start+2*acclen-1

    ########## xcorr tests ###########
    xc, nr = cr.avg_xcorr_4bit_float_gpu(data['pol0'], data['pol1'], data['specnums'], data['specnums']) # should be 1
    assert nr==5
    assert xp.allclose(xc,1,atol=1e-7)
    xc, nr = cr.avg_xcorr_4bit_float_gpu(data['pol0'], data['pol0'], data['specnums'], data['specnums']) # should be 1
    assert nr==5
    assert xp.allclose(xc,1,atol=1e-7)
    xc, nr = cr.avg_xcorr_4bit_float_gpu(data['pol0'][:,0:1], data['pol0'][:,1:], data['specnums'], data['specnums']) # should be 1j
    assert nr==5
    assert xp.allclose(xc,1j,atol=1e-7)
    ###################################


    data=ant1.__next__() #first 30
    print("First 30", data)
    assert isinstance(data['pol0'], xp.ndarray)
    assert isinstance(data['pol1'], xp.ndarray)
    assert isinstance(ant1.obj.raw_data, xp.ndarray)
    assert isinstance(ant1.obj.spec_idx, np.ndarray)
    assert isinstance(data['specnums'], np.ndarray)

    # print(data)
    data=ant1.__next__() #first 40, 5 from file 1, 5 from file 2
    print("First 40, split across two files", data)
    data['specnums'][-1]+5-1==ant1.obj.spec_idx[5-1] #we must have hit the end of file 1 midway in this block
    # print(data)
    assert ant1.obj._overflowed==False #second file has no overflows
    assert data['specnums'][-1]==spec_start+4*acclen-1
    data=ant1.__next__() #first 50, next 10 from file 2. end of file 2
    # print(data)
    data['specnums'][-1]+5-1==ant1.obj.spec_idx[-1]
    data=ant1.__next__() #file 3, has another wrap
    assert ant1._OVERFLOW_CTR==2
    assert len(data['specnums'])==5 #because we introduced a huge gap by wrapping in middle of file
    assert xp.allclose(data['pol0'][:5,0]+data['pol0'][:5,1],1+1j,atol=1e-7)
    assert xp.allclose(data['pol0'][:5,0]+data['pol1'][:5,1],1+1j,atol=1e-7)
    assert xp.allclose(data['pol0'][5:],0,atol=1e-7) #should be zero since it wasn't filled
    assert xp.allclose(data['pol1'][5:],0,atol=1e-7)