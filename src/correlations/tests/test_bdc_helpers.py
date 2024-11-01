import pytest
from src.correlations import baseband_data_classes as bdc
from src import xp
import numpy as np
def test_make_continuous():
    assert xp.__name__ == 'cupy'
    nspec=100
    nchans=2049
    channels=[100,200]
    specnum=np.asarray([10,20,30,40,50],dtype='int64')
    spec=xp.random.randn(len(specnum),len(channels))
    spec=spec.astype("complex64")
    new_spec = bdc.make_continuous_gpu(spec,specnum,channels,nspec,nchans)
    assert new_spec[10,100]==spec[0,0]
    assert new_spec[50,200]==spec[4,1]
    assert new_spec[49,200]==0.
    assert new_spec[50,199]==0.
    assert new_spec[9,100]==0.
    assert new_spec[10,101]==0.
    spec=xp.ones((len(specnum),len(channels)),dtype='complex64')+1j*xp.ones((len(specnum),len(channels)),dtype='complex64')
    new_spec = bdc.make_continuous_gpu(spec,specnum,channels,nspec,nchans)
    assert xp.sum(new_spec) == 10+10j
    assert xp.sum(new_spec,axis=0)[100] == 5.+5.j
    assert xp.sum(new_spec,axis=0)[200] == 5.+5.j
