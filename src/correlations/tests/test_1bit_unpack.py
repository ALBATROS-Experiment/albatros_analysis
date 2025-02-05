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
    pol0, pol1 = unpk.unpack_1bit(packed_1bit, 8, 0, 2, 2, 6)  # chan 2 to chan 5 (4 channels)
    truepol0, truepol1 = expected_complex_1bit
    truepol0 = truepol0[:, 2:6]
    truepol1 = truepol1[:, 2:6]
    # print("packed",packed_1bit)
    # print("pol0",pol0)
    # print("truth", truepol0)
    # print(pol0-truepol0)
    assert np.all(pol0 == truepol0)
    assert np.all(pol1 == truepol1)


def test_unpack(packed_1bit, real_im_1bit_pol0, real_im_1bit_pol1):
    rowstart = 0
    rowend = 2
    chanstart = 2
    chanend = 7
    pol0, pol1 = unpk.sortpols(packed_1bit, 8, 1, rowstart, rowend, chanstart, chanend)
    assert pol0[1, 0] == str_to_bits(
        "01110101"
    )  # channel 3 the different one is in first byte
    assert pol0[0, 0] == str_to_bits("10101010")
    assert pol0[1, 1] == str_to_bits("01000000")
    assert pol1[1, 0] == str_to_bits("10001010")
    assert pol1[0, 0] == str_to_bits("01010101")
    assert pol1[1, 1] == str_to_bits("10000000")


def test_unpack_arb_row(packed_1bit, real_im_1bit_pol0, real_im_1bit_pol1):
    rowstart = 1
    rowend = 2
    chanstart = 2
    chanend = 7
    pol0, pol1 = unpk.sortpols(packed_1bit, 8, 1, rowstart, rowend, chanstart, chanend)
    # print(pol0,pol1)
    assert pol0[0, 0] == str_to_bits("01110101")
    assert pol0[0, 1] == str_to_bits("01000000")
    assert pol1[0, 0] == str_to_bits("10001010")
    assert pol1[0, 1] == str_to_bits("10000000")


def test_histogram_1bit(packed_1bit, real_im_1bit_pol0, real_im_1bit_pol1):
    rowstart = 0
    rowend = 2
    length_channels = 8
    bit_depth = 1
    mode = 0
    hist = unpk.hist(packed_1bit, rowstart, rowend, length_channels, bit_depth, mode)
    assert hist[0, 3] == 1
    assert hist[1, 3] == 3
    assert hist[0, 4] == 2
    assert hist[0, 4] == 2

    mode = 1
    hist = unpk.hist(packed_1bit, rowstart, rowend, length_channels, bit_depth, mode)
    assert hist[0, 3] == 3
    assert hist[1, 3] == 1
    assert hist[0, 4] == 2
    assert hist[0, 4] == 2

    mode = -1
    hist = unpk.hist(packed_1bit, rowstart, rowend, length_channels, bit_depth, mode)
    assert hist[0, 3] == 4
    assert hist[1, 3] == 4
    assert hist[0, 4] == 4
    assert hist[0, 4] == 4
