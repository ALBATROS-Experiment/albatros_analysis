import pytest
import numpy as np
from src.correlations import unpacking as unpk

def str_to_bits(str):
    s = 0
    return sum([int(str[i]) * 2 ** (7 - i) for i in range(0, 8)])


@pytest.fixture
def real_im_1bit_pol0():
    # 8 channels
    r = np.ones(8, dtype="int64").reshape(-1, 8)  # first row all 1
    r = np.vstack([r, -1 * r])  # second row all -1
    r[1, 3] = 1  # for single channel test.
    im = -1 * np.ones(8, dtype="int64").reshape(-1, 8)  # first row all -1
    im = np.vstack([im, -1 * im])  # second row is all 1
    return r, im


@pytest.fixture
def real_im_1bit_pol1():
    r = -1 * np.ones(8, dtype="int64").reshape(-1, 8)  # first row all -1
    r = np.vstack([r, -1 * r])  # second row all 1
    r[1, 3] = -1
    im = np.ones(8, dtype="int64").reshape(-1, 8)  # first row all 1
    im = np.vstack([im, -1 * im])  # second row all -1
    return r, im


@pytest.fixture
def expected_complex_1bit(real_im_1bit_pol0, real_im_1bit_pol1):
    # print(real_im_1bit_pol0,"from expected")
    return (
        real_im_1bit_pol0[0] + 1j * real_im_1bit_pol0[1],
        real_im_1bit_pol1[0] + 1j * real_im_1bit_pol1[1],
    )


@pytest.fixture
def packed_1bit(real_im_1bit_pol0, real_im_1bit_pol1):
    pshape = 4 * 2  # nchan/2 * nspec
    packed = np.zeros((1, pshape), dtype="uint8")
    p0r = real_im_1bit_pol0[0].copy()
    p0im = real_im_1bit_pol0[1].copy()
    p1r = real_im_1bit_pol1[0].copy()
    p1im = real_im_1bit_pol1[1].copy()
    p0r[p0r < 0] = p0r[p0r < 0] + 1
    p1r[p1r < 0] = p1r[p1r < 0] + 1
    p0im[p0im < 0] = p0im[p0im < 0] + 1
    p1im[p1im < 0] = p1im[p1im < 0] + 1
    for i in range(0, 2):
        for j in range(0, 4):
            packed[0, i * 4 + j] = (
                (p0r[i, 2 * j] << 7)
                + (p0im[i, 2 * j] << 6)
                + (p1r[i, 2 * j] << 5)
                + (p1im[i, 2 * j] << 4)
                + (p0r[i, 2 * j + 1] << 3)
                + (p0im[i, 2 * j + 1] << 2)
                + (p1r[i, 2 * j + 1] << 1)
                + (p1im[i, 2 * j + 1])
            )
    return packed


def test_float_unpack(packed_1bit, expected_complex_1bit):
    d_packed_1bit = unpk.xp.asarray(packed_1bit)
    d_pol0, d_pol1 = unpk.unpack_1bit(d_packed_1bit, 0, 2, np.arange(2,6), 8)  # rowstart=0, rowend=2, chan 2 to chan 5 (4 channels)
    truepol0, truepol1 = expected_complex_1bit
    d_truepol0 = unpk.xp.asarray(truepol0)
    d_truepol1 = unpk.xp.asarray(truepol1)
    # print("packed",packed_1bit)
    # print("pol0",pol0)
    # print("truth", d_truepol0)
    # print(pol0-d_truepol0)
    assert np.all(d_pol0 == d_truepol0[:, 2:6])
    assert np.all(d_pol1 == d_truepol1[:, 2:6])

    unpk.set_backend('numpy')
    print("Compare with CPU")
    pol0, pol1 = unpk.unpack_1bit(packed_1bit, 0, 2, np.arange(2,6), 8) #do it on CPU
    assert np.all(pol0 == truepol0[:, 2:6])
    assert np.all(pol1 == truepol1[:, 2:6])

def test_float_unpack_one_chan(packed_1bit, expected_complex_1bit):
    #GPU float unpack requires start and end channels to be even
    d_packed_1bit = unpk.xp.asarray(packed_1bit)
    d_pol0, d_pol1 = unpk.unpack_1bit(d_packed_1bit, 1, 2, np.arange(2,4), 8)  # 1,3 element
    truepol0, truepol1 = expected_complex_1bit
    d_truepol0 = unpk.xp.asarray(truepol0)
    d_truepol1 = unpk.xp.asarray(truepol1)
    # print("packed",packed_1bit)
    # print("pol0",pol0)
    # print("truth", d_truepol0)
    # print(pol0-d_truepol0)

    assert np.all(d_pol0[0,1] == 1+1j)
    assert np.all(d_pol1[0,1] == -1-1j)
