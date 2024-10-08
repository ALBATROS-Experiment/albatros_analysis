import pytest
import numpy as np
from src.correlations import unpacking as unpk


# funtions copied from dump baseband
def unpack_4bit(raw):
    re = np.asarray(np.right_shift(np.bitwise_and(raw, 0xF0), 4), dtype="int8")
    re[re > 8] = re[re > 8] - 16
    im = np.asarray(np.bitwise_and(raw, 0x0F), dtype="int8")
    im[im > 8] = im[im > 8] - 16
    vec = 1j * im + re
    return vec


def unpack_packet(packet, bits, spec_per_packet):
    if bits == 4:
        vec = unpack_4bit(
            packet
        )  # unpack_4bit(packet[4:]) first 4 bytes were specnum, which I dont have here
        nchan = len(vec) // spec_per_packet // 2
        # print("nchan inferred", nchan)
        pol0 = np.reshape(vec[::2], [spec_per_packet, nchan])
        pol1 = np.reshape(vec[1::2], [spec_per_packet, nchan])
        return pol0, pol1


@pytest.fixture
def real_im_4bit_pol0():
    r = np.ones(10, dtype="int64").reshape(-1, 10)
    r = np.vstack([r, 0 * r])
    r[1, 3] = 4  # for single channel test.
    im = -1 * np.ones(10, dtype="int64").reshape(-1, 10)
    im = np.vstack([im, 0 * im])
    return r, im


@pytest.fixture
def real_im_4bit_pol1():
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
    packed = np.zeros((1, pshape), dtype="uint8")
    packed[0, 0::2] = np.ravel(packed_pol0)
    packed[0, 1::2] = np.ravel(packed_pol1)
    return packed

def test_unpack_4bit(packed_4bit, expected_complex_4bit):
    pol0, pol1 = unpk.unpack_4bit(packed_4bit, 0, 2, np.arange(0,10),  10)
    truepol0, truepol1 = expected_complex_4bit
    print(pol0,"\n", truepol0)
    assert pol0.shape == truepol0.shape
    pol0diff = pol0 - truepol0
    pol1diff = pol1 - truepol1
    assert np.all(np.abs(pol0diff) < 1e-15)
    assert np.all(np.abs(pol1diff) < 1e-15)


def test_unpack_4bit_1chan(packed_4bit, expected_complex_4bit):
    pol0, pol1 = unpk.unpack_4bit(packed_4bit, 0, 2, np.arange(3,4),  10)
    # print(pol0,pol1)
    truepol0, truepol1 = expected_complex_4bit
    truepol0 = truepol0[0:2, 3:4]
    truepol1 = truepol1[0:2, 3:4]
    assert pol0.shape == truepol0.shape
    pol0diff = pol0 - truepol0
    pol1diff = pol1 - truepol1
    assert np.all(np.abs(pol0diff) < 1e-15)
    assert np.all(np.abs(pol1diff) < 1e-15)

def test_unpack_4bit_2chans(packed_4bit, expected_complex_4bit):
    pol0, pol1 = unpk.unpack_4bit(packed_4bit, 0, 2, np.asarray([2,5],dtype="int64"), 10)
    # print(pol0,pol1)
    truepol0, truepol1 = expected_complex_4bit
    truepol0 = truepol0[0:2, [2,5]]
    truepol1 = truepol1[0:2, [2,5]]
    assert pol0.shape == truepol0.shape
    pol0diff = pol0 - truepol0
    pol1diff = pol1 - truepol1
    assert np.all(np.abs(pol0diff) < 1e-15)
    assert np.all(np.abs(pol1diff) < 1e-15)

def test_against_jon(packed_4bit):
    pol0, pol1 = unpk.unpack_4bit(packed_4bit,0, 2,np.arange(0,10), 10)
    truepol0, truepol1 = unpack_packet(packed_4bit.flatten(), 4, 2)
    # print(packed_4bit,"packed passing")
    # print(truepol0,truepol1, "true stuff")
    assert pol0.shape == truepol0.shape
    pol0diff = pol0 - truepol0
    pol1diff = pol1 - truepol1
    assert np.all(np.abs(pol0diff) < 1e-15)
    assert np.all(np.abs(pol1diff) < 1e-15)


def test_histogram_4bit(packed_4bit, real_im_4bit_pol0, real_im_4bit_pol1):
    r0, im0 = real_im_4bit_pol0
    r1, im1 = real_im_4bit_pol1
    length_channels = 10
    bit_depth = 4
    mode = 0
    rowstart = 0
    rowend = 2
    histvals = unpk.hist(
        packed_4bit, rowstart, rowend, length_channels, bit_depth, mode
    )
    # print(len(np.where(r0==0)[0])+len(np.where(im0==0)[0]))
    # print(histvals)
    assert histvals[0, 3] == 1
    assert np.sum(histvals[0, :]) == len(np.where(r0 == 0)[0]) + len(
        np.where(im0 == 0)[0]
    )
    assert histvals[4, 3] == 1

    mode = 1
    histvals = unpk.hist(
        packed_4bit, rowstart, rowend, length_channels, bit_depth, mode
    )
    # print(len(np.where(r0==0)[0])+len(np.where(im0==0)[0]))
    assert histvals[0, 3] == 1
    assert np.sum(histvals[7, :]) == len(np.where(r1 == 7)[0]) + len(
        np.where(im1 == 7)[0]
    )
    assert histvals[12, 3] == 1

    mode = -1
    histvals = unpk.hist(
        packed_4bit, rowstart, rowend, length_channels, bit_depth, mode
    )
    # print(len(np.where(r0==0)[0])+len(np.where(im0==0)[0]))
    assert np.sum(histvals[0, :]) == len(np.where(r1 == 0)[0]) + len(
        np.where(im1 == 0)[0]
    ) + len(np.where(r0 == 0)[0]) + len(np.where(im0 == 0)[0])
    assert np.sum(histvals[7, :]) == len(np.where(r1 == 7)[0]) + len(
        np.where(im1 == 7)[0]
    ) + len(np.where(r0 == 7)[0]) + len(np.where(im0 == 7)[0])
    assert histvals[12, 3] == 1
    assert histvals[4, 3] == 1
