import pytest
import numpy as np
from src.correlations import baseband_data_classes as bdc
from src.correlations.tests.mars2019 import albatrostools as tools
import os

dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="module")
def full_file():
    obj = bdc.BasebandPacked(os.path.join(dirpath, "1627202039.raw"))
    return obj


@pytest.fixture(scope="module")
def iterator_obj_1():
    fileidx = 0
    idxstart = 550
    acclen = 10000
    files = [
        os.path.join(dirpath, "1627202039_1.raw"),
        os.path.join(dirpath, "1627202039_2.raw"),
    ]
    obj = bdc.BasebandFileIterator(files, fileidx, idxstart, acclen, nchunks=4)
    return obj


@pytest.fixture(scope="module")
def iterator_obj_2():
    fileidx = 0
    idxstart = 550
    acclen = 30000
    # 22420 missing total in between the two files
    files = [
        os.path.join(dirpath, "1627202039_1.raw"),
        os.path.join(dirpath, "1627202039_2.raw"),
    ]
    obj = bdc.BasebandFileIterator(files, fileidx, idxstart, acclen, nchunks=4)
    return obj


def test_missing_inside(iterator_obj_1, full_file):
    v1 = iterator_obj_1.__next__()
    assert len(v1["specnums"]) == 100
    assert np.all(full_file.spec_idx[550:650] == v1["specnums"])
    assert np.all(full_file.pol0[550:650, :] == v1["pol0"][:100, :])
    assert np.all(full_file.pol1[550:650, :] == v1["pol1"][:100, :])
    # only 100 rows filled, rest 0 and discarded. Corr functions controlled by len of specnum

    # this will be completely inside missing block
    v1 = iterator_obj_1.__next__()
    assert len(v1["specnums"]) == 0

    v1 = iterator_obj_1.__next__()
    assert len(v1["specnums"]) == 7480
    assert np.all(full_file.spec_idx[650 : 650 + 7480] == v1["specnums"])
    assert np.all(full_file.pol0[650 : 650 + 7480, :] == v1["pol0"][:7480, :])
    assert np.all(full_file.pol1[650 : 650 + 7480, :] == v1["pol1"][:7480, :])


def test_missing_overall(iterator_obj_2, full_file):
    v1 = iterator_obj_2.__next__()
    assert len(v1["specnums"]) == 7580
    assert np.all(full_file.spec_idx[550 : 550 + 7580] == v1["specnums"])
    assert np.all(full_file.pol0[550 : 550 + 7580, :] == v1["pol0"][:7580, :])
    assert np.all(full_file.pol1[550 : 550 + 7580, :] == v1["pol1"][:7580, :])
