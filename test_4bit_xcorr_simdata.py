import numpy as np
from src.correlations import correlations as cr

def pols_float():
    ones=np.ones((10,5),dtype="float64")
    pol0=ones + 1j*ones
    pol1=ones + 1j*0
    pol1[:,3:] = 0 #last two columns zero
    return pol0,pol1

def pols_packed():
    pol0,pol1 = pols_float()
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

def test_4bit_same_specnum():
    p_packed = pols_packed()
    p_float = pols_float()
    specnum=np.empty(p_packed[0].shape[0],dtype="int64",order="c")
    specnum[:]=np.arange(0,p_packed[0].shape[0],dtype="int64")
    avg_packed = cr.avg_xcorr_4bit(p_packed[0],p_packed[1],specnum,specnum,0,0)
    avg_float = np.sum(p_float[0]*np.conj(p_float[1]), axis=0)
    assert np.allclose(avg_packed,avg_float)

def test_4bit_diff_specnum():
    p_packed = pols_packed()
    p_float = pols_float()
    specnum=np.empty(p_packed[0].shape[0],dtype="int64",order="c")
    specnum[:]=np.arange(0,p_packed[0].shape[0],dtype="int64")
    avg_packed = cr.avg_xcorr_4bit(p_packed[0],p_packed[1],specnum,specnum.copy(),0,0)
    avg_float = np.sum(p_float[0]*np.conj(p_float[1]), axis=0)
    assert np.allclose(avg_packed,avg_float)

test_4bit_diff_specnum()