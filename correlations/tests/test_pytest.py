import pytest
import numpy as np


@pytest.fixture(
    scope="module"
)  # if module then at the end value 3333.. instead of 2222...
def real_im_4bit_pol0():
    r = np.ones(10, dtype="int64")
    print("orig add", r.ctypes.data)  # gets executed only once
    r[2] = -1
    return r


@pytest.fixture
def expected(real_im_4bit_pol0):
    print("address expec", real_im_4bit_pol0.ctypes.data)
    return real_im_4bit_pol0


@pytest.fixture
def packed(real_im_4bit_pol0):
    r = real_im_4bit_pol0
    print("address packed", r.ctypes.data)
    # r[r<0]=r[r<0]+10
    # r=r<<4
    r[:] = r + 1  # gonna modify the original array
    print("address packed", r.ctypes.data)
    return r


def test1(packed, expected):
    print("packed address", packed.ctypes.data, "expec address", expected.ctypes.data)
    print("Packed", packed, "expected", expected)
    assert 1 == 1


def test2(packed, expected):
    print("packed address", packed.ctypes.data, "expec address", expected.ctypes.data)
    print("Packed", packed, "expected", expected)
    assert 1 == 1
