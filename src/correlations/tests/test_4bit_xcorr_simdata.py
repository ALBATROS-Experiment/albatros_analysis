import pytest
import numpy as np
from src.correlations import baseband_data_classes as bdc
from src.correlations import correlations as cr
import os

@pytest.fixture
def pols_float():
    ones=np.ones((10,5),dtype="float64")
    pol0=ones + 1j*ones
    pol1=ones + 1j*0
    pol1[:,3:] = 0 #last two columns zero
    return pol0,pol1
@pytest.fixture
def pols_packed(pols_float):
    pol0,pol1 = pols_float
    r = np.real(pol0).astype("int")
    im = np.imag(pol0).astype("int")
    r[r < 0] = r[r < 0] + 16
    r = r << 4
    im[im < 0] = im[im < 0] + 16
    packed_pol0 = (r + im).astype("uint8")

    r = np.real(pol1).astype("int")
    im = np.imag(pol1).astype("int")
    r[r < 0] = r[r < 0] + 16
    r = r << 4
    im[im < 0] = im[im < 0] + 16
    packed_pol1 = (r + im).astype("uint8")
    return packed_pol0, packed_pol1

def test_4bit(pols_packed):
    print(pols_packed[0])
    print(pols_packed[1])
    specnum=np.arange(0,pols_packed[0].shape[0])
    avg_packed = cr.avg_xcorr_4bit(pols_packed[0],pols_packed[1],specnum,specnum,0,0)
    print(avg_packed)
