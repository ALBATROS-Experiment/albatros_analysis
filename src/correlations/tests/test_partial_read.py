import pytest
import numpy as np
from src.correlations import baseband_data_classes as bdc
from src.correlations import correlations as cr
import os

fpath1 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/1667664784_1.raw"
)
fpath2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/1667664784.raw")
fpath3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/1627202039.raw")


@pytest.fixture
def snipped1bit():
    obj = bdc.BasebandPacked(fpath1)
    return obj


@pytest.fixture
def partial1bit():
    obj = bdc.BasebandPacked(fpath2, readlen=10)
    return obj


@pytest.fixture
def full4bit():
    obj = bdc.BasebandPacked(fpath3)
    return obj


@pytest.fixture
def partial4bit():
    obj = bdc.BasebandPacked(fpath3, readlen=500)
    return obj


def test_partial_snipped_1bit(snipped1bit, partial1bit):
    assert np.all(snipped1bit.spec_idx == partial1bit.spec_idx)
    assert np.all(snipped1bit.pol0 == partial1bit.pol0)
    assert np.all(snipped1bit.pol1 == partial1bit.pol1)


def test_partial_full_4bit(full4bit, partial4bit):
    en = full4bit.spectra_per_packet * 500
    assert np.all(full4bit.pol0[0:en] == partial4bit.pol0)
    assert np.all(full4bit.pol1[0:en] == partial4bit.pol1)
