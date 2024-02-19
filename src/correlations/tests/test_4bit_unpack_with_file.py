import pytest
import numpy as np
from src.correlations import baseband_data_classes as bdc
from src.correlations.tests.mars2019 import albatrostools as tools
import os

# fpath=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/baseband/SNAP4/16668/1666886964.raw')
fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/1627202039.raw")


@pytest.fixture
def file_obj_float():
    obj = bdc.BasebandFloat(fpath)
    return obj


@pytest.fixture
def file_obj_packed():
    obj = bdc.BasebandPacked(fpath)
    return obj


def test_float_unpack_against_jons(file_obj_float):
    header, dat = tools.get_data(fpath, unpack_fast=True)
    # print(dat["pol0"], "old")
    # print("new", file_obj_float.pol0)
    assert np.all(dat["pol0"] == file_obj_float.pol0)
    assert np.all(dat["pol1"] == file_obj_float.pol1)
    assert np.all(dat["spectrum_number"] == file_obj_float.spec_num)


def test_packed_unpack_against_jons(file_obj_packed):
    header, specno, pol0, pol1 = tools.get_data_raw(fpath)
    # print(pol0.dtype)
    # print(pol0,file_obj_packed.pol0)
    assert np.all(pol0 == file_obj_packed.pol0)
    assert np.all(pol1 == file_obj_packed.pol1)
    assert np.all(specno == file_obj_packed.spec_num)
