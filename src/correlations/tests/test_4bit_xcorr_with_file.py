import pytest
import numpy as np
from src.correlations import baseband_data_classes as bdc
from src.correlations import correlations as cr
import os

fpath = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/1627202039_1.raw"
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
def objrc():
    obj = bdc.BasebandPacked(fpath, rowstart=10, rowend=20, chanstart=4, chanend=12)
    return obj


def test_4bit_corrs_all_chans(file_obj_float, file_obj_packed):
    specnums = np.arange(0, file_obj_float.pol0.shape[0])
    cr1 = cr.avg_xcorr_4bit(
        file_obj_packed.pol0[0:, :], file_obj_packed.pol1[0:, :], specnums
    )
    cr2 = (
        np.sum(file_obj_float.pol0[0:, :] * np.conj(file_obj_float.pol1[0:, :]), axis=0)
        / file_obj_float.pol0.shape[0]
    )
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15

    cr1 = cr.avg_autocorr_4bit(file_obj_packed.pol0[0:, :], specnums)
    xx = np.real(
        file_obj_float.pol0[0:, :] * np.conj(file_obj_float.pol0[0:, :])
    ).astype(int)
    cr2 = np.sum(xx, axis=0) / file_obj_float.pol0.shape[0]
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15

    cr1 = cr.avg_autocorr_4bit(file_obj_packed.pol1[0:, :], specnums)
    xx = np.real(
        file_obj_float.pol1[0:, :] * np.conj(file_obj_float.pol1[0:, :])
    ).astype(int)
    cr2 = np.sum(xx, axis=0) / file_obj_float.pol0.shape[0]
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15


def test_4bit_corrs_arb_chan_arb_row(file_obj_float, objrc):
    rowstart = 10
    rowend = 20
    chanstart = 4
    chanend = 12
    specnums = np.arange(0, rowend - rowstart)
    cr1 = cr.avg_xcorr_4bit(objrc.pol0, objrc.pol1, specnums)
    cr2 = np.sum(
        file_obj_float.pol0[rowstart:rowend, chanstart:chanend]
        * np.conj(file_obj_float.pol1[rowstart:rowend, chanstart:chanend]),
        axis=0,
    ) / (rowend - rowstart)
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15

    cr1 = cr.avg_autocorr_4bit(objrc.pol0, specnums)
    xx = np.real(
        file_obj_float.pol0[rowstart:rowend, chanstart:chanend]
        * np.conj(file_obj_float.pol0[rowstart:rowend, chanstart:chanend])
    ).astype(int)
    cr2 = np.sum(xx, axis=0) / (rowend - rowstart)
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15

    cr1 = cr.avg_autocorr_4bit(objrc.pol1, specnums)
    xx = np.real(
        file_obj_float.pol1[rowstart:rowend, chanstart:chanend]
        * np.conj(file_obj_float.pol1[rowstart:rowend, chanstart:chanend])
    ).astype(int)
    cr2 = np.sum(xx, axis=0) / (rowend - rowstart)
    assert np.all(np.abs(cr2 - cr1)) <= 1e-15
