import pytest
from src import xp
from src.correlations import unpacking as unpk
import numpy as np
#1) MistEnv/2021a (S)   2) cuda/11.2.2   3) gcc/10.3.0   4) fftw/3.3.10
@pytest.fixture
def real_im_4bit_pol0():
    # two rows, 10 channels
    # first row all 1 -1j second row all 0's except channel 3
    r = np.ones(10, dtype="int64").reshape(-1, 10)
    r = np.vstack([r, 0 * r])
    r[1, 3] = 4  # for single channel test.
    im = -1 * np.ones(10, dtype="int64").reshape(-1, 10)
    im = np.vstack([im, 0 * im])
    return r, im

@pytest.fixture
def real_im_4bit_pol1():
    # two rows, 10 channels
    # first row all 7 -7j second row all 0's except channel 3
    r = 7 * np.ones(10, dtype="int64").reshape(-1, 10)
    r = np.vstack([r, 0 * r])
    r[1, 3] = -4
    im = -7 * np.ones(10, dtype="int64").reshape(-1, 10)
    im = np.vstack([im, 0 * im])
    return r, im


@pytest.fixture
def expected_complex_4bit(real_im_4bit_pol0, real_im_4bit_pol1):
    # print(real_im_4bit_pol0,"from expected")
    return (
        real_im_4bit_pol0[0] + 1j * real_im_4bit_pol0[1],
        real_im_4bit_pol1[0] + 1j * real_im_4bit_pol1[1],
    )


@pytest.fixture
def packed_4bit(real_im_4bit_pol0, real_im_4bit_pol1):
    r = real_im_4bit_pol0[0].copy()
    im = real_im_4bit_pol0[1].copy()
    r[r < 0] = r[r < 0] + 16
    r = r << 4
    im[im < 0] = im[im < 0] + 16
    packed_pol0 = (r + im).astype("uint8")

    r = real_im_4bit_pol1[0].copy()
    im = real_im_4bit_pol1[1].copy()
    r[r < 0] = r[r < 0] + 16
    r = r << 4
    im[im < 0] = im[im < 0] + 16
    packed_pol1 = (r + im).astype("uint8")
    pshape = (
        packed_pol0.shape[0] * packed_pol0.shape[1]
        + packed_pol1.shape[0] * packed_pol1.shape[1]
    )
    packed = np.zeros((1, pshape), dtype="uint8") # 1 packet, 2 spectra, 10 chan per spectra
    packed[0, 0::2] = np.ravel(packed_pol0)
    packed[0, 1::2] = np.ravel(packed_pol1)
    return packed

def test_unpack_4bit_cupy(packed_4bit, expected_complex_4bit):
    print("Using", xp.__name__)
    d_packed_4bit=xp.asarray(packed_4bit)
    d_pol0, d_pol1 = unpk.unpack_4bit(d_packed_4bit, 0, 2, np.arange(0,10), 10)
    pol0 = xp.asnumpy(d_pol0)
    pol1 = xp.asnumpy(d_pol1)
    truepol0, truepol1 = expected_complex_4bit
    assert pol0.shape == truepol0.shape
    pol0diff = pol0 - truepol0
    pol1diff = pol1 - truepol1
    assert np.all(np.abs(pol0diff) < 1e-15)
    assert np.all(np.abs(pol1diff) < 1e-15)

def test_unpack_4bit_2chans_cupy(packed_4bit, expected_complex_4bit):
    d_packed_4bit=xp.asarray(packed_4bit)
    d_pol0, d_pol1 = unpk.unpack_4bit(d_packed_4bit, 0, 2, np.asarray([2,5]), 10)
    pol0 = xp.asnumpy(d_pol0)
    pol1 = xp.asnumpy(d_pol1)
    truepol0, truepol1 = expected_complex_4bit
    truepol0 = truepol0[0:2, [2,5]]
    truepol1 = truepol1[0:2, [2,5]]
    assert pol0.shape == truepol0.shape
    pol0diff = pol0 - truepol0
    pol1diff = pol1 - truepol1
    assert np.all(np.abs(pol0diff) < 1e-15)
    assert np.all(np.abs(pol1diff) < 1e-15)