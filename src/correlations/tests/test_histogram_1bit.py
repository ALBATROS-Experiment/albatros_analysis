import numpy as np
import numba as nb
import os
from src.correlations import baseband_data_classes as bdc
import pytest


@nb.njit()
def myhist(arr, res):
    mylen = arr.shape[0]
    mycol = arr.shape[1]
    for i in range(mylen):
        for j in range(mycol):
            res[(arr[i, j]) & 1, 4 * j + 3] += 1
            res[(arr[i, j] >> 1) & 1, 4 * j + 3] += 1
            res[(arr[i, j] >> 2) & 1, 4 * j + 2] += 1
            res[(arr[i, j] >> 3) & 1, 4 * j + 2] += 1
            res[(arr[i, j] >> 4) & 1, 4 * j + 1] += 1
            res[(arr[i, j] >> 5) & 1, 4 * j + 1] += 1
            res[(arr[i, j] >> 6) & 1, 4 * j] += 1
            res[(arr[i, j] >> 7) & 1, 4 * j] += 1


fpath = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/1667664784_1.raw"
)


@pytest.fixture
def file_obj_packed():
    obj = bdc.BasebandPacked(fpath)
    return obj


def test_histogram_1bit_file(file_obj_packed):
    r1 = np.zeros((2, file_obj_packed.pol0.shape[1] * 4), dtype="int64")
    myhist(file_obj_packed.pol0, r1)
    r2 = np.zeros((2, file_obj_packed.pol1.shape[1] * 4), dtype="int64")
    myhist(file_obj_packed.pol1, r2)
    hist1 = file_obj_packed.get_hist(mode=0)
    hist2 = file_obj_packed.get_hist(mode=1)
    assert file_obj_packed.pol0.shape[0] == len(file_obj_packed.spec_idx)
    assert (
        np.sum(r1) == file_obj_packed.pol0.shape[0] * file_obj_packed.pol0.shape[1] * 8
    )  # conservation of bits
    assert (
        np.sum(r2) == file_obj_packed.pol1.shape[0] * file_obj_packed.pol1.shape[1] * 8
    )  # conservation of bits
    assert (
        np.sum(hist1)
        == file_obj_packed.pol0.shape[0] * file_obj_packed.pol0.shape[1] * 8
    )
    assert (
        np.sum(hist2)
        == file_obj_packed.pol1.shape[0] * file_obj_packed.pol1.shape[1] * 8
    )
    assert np.all(r1 == hist1)
    assert np.all(r2 == hist2)
