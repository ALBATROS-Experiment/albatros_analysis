import pytest
import numpy as np
from src.correlations import baseband_data_classes as bdc
from src.correlations import correlations as cr
import os

fpath = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/1667664784_1.raw"
)


@pytest.fixture
def file_obj_float():
    obj = bdc.BasebandFloat(fpath)
    return obj


@pytest.fixture
def file_obj_packed():
    obj = bdc.BasebandPacked(fpath)
    return obj


@pytest.fixture
def objrc1():
    obj = bdc.BasebandPacked(fpath, rowstart=10, rowend=20, chanstart=6, chanend=10)
    return obj


@pytest.fixture
def objrc2():
    obj = bdc.BasebandPacked(fpath, rowstart=10, rowend=20, chanstart=6, chanend=12)
    return obj


def test_1bit_xcorr_all_chans(file_obj_float, file_obj_packed):
    assert len(file_obj_float.spec_idx) == len(file_obj_packed.spec_idx)
    assert file_obj_float.pol0.shape[0] == file_obj_packed.pol0.shape[0]
    nchans = file_obj_packed.length_channels
    specnums = np.arange(0, file_obj_float.pol0.shape[0])
    cr1 = cr.avg_xcorr_1bit(
        file_obj_packed.pol0[0:, :], file_obj_packed.pol1[0:, :], specnums, nchans
    )
    cr2 = (
        np.sum(file_obj_float.pol0[0:, :] * np.conj(file_obj_float.pol1[0:, :]), axis=0)
        / file_obj_float.pol0.shape[0]
    )
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15


def test_1bit_xcorr_arb_chan_arb_row1(file_obj_float, objrc1):
    nchans = 4
    rowstart = 10
    rowend = 20
    chanstart = 6
    chanend = 10
    assert objrc1.pol0.shape[0] == (rowend - rowstart)
    specnums = np.arange(0, objrc1.pol0.shape[0])
    cr1 = cr.avg_xcorr_1bit(objrc1.pol0[0:, :], objrc1.pol1[0:, :], specnums, nchans)
    cr2 = np.sum(
        file_obj_float.pol0[rowstart:rowend, chanstart:chanend]
        * np.conj(file_obj_float.pol1[rowstart:rowend, chanstart:chanend]),
        axis=0,
    ) / (rowend - rowstart)
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15


def test_1bit_xcorr_arb_chan_arb_row2(file_obj_float, objrc2):
    nchans = 6
    rowstart = 10
    rowend = 20
    chanstart = 6
    chanend = 12
    assert objrc2.pol0.shape[0] == (rowend - rowstart)
    specnums = np.arange(0, objrc2.pol0.shape[0])
    cr1 = cr.avg_xcorr_1bit(objrc2.pol0[0:, :], objrc2.pol1[0:, :], specnums, nchans)
    cr2 = np.sum(
        file_obj_float.pol0[rowstart:rowend, chanstart:chanend]
        * np.conj(file_obj_float.pol1[rowstart:rowend, chanstart:chanend]),
        axis=0,
    ) / (rowend - rowstart)
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15
