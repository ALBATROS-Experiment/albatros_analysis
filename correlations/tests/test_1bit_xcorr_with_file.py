import pytest
import numpy as np
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
import os

fpath=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/1627202039.raw')

@pytest.fixture(scope='module')
def file_obj_float():
    obj = bdc.BasebandFloat(fpath)
    return obj
@pytest.fixture(scope='module')
def file_obj_packed():
    obj = bdc.BasebandPacked(fpath)
    return obj

def test_1bit_xcorr_all_chans(file_obj_float, file_obj_packed):
    assert(len(file_obj_float.spec_idx)==len(file_obj_packed.spec_idx))
    assert(np.all(file_obj_float.pol0.shape==file_obj_packed.pol0.shape))
    nchans=file_obj_packed.length_channel
    specnums = np.arange(0,obj.pol0.shape[0])
    cr1 = cr.avg_xcorr_1bit(file_obj_packed.pol0[0:,:],file_obj_packed.pol1[0:,:],specnums,nchans)
    cr2 = np.sum(file_obj_float.pol0[0:,:]*np.conj(file_obj_float.pol1[0:,:]),axis=0)/file_obj_float.pol0.shape[0]
    assert(np.all(np.abs(cr2-cr1))<=1e-15)