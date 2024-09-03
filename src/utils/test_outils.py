import src.utils.orbcomm_utils as ou
import numpy as np

def test_apply_sat_delay():

    satnorads = [40087, 41187]
    sat2chan = {41187: [10, 20], 40087: [300, 400]}
    chan2sat = {10: 41187, 20: 41187, 300: 40087, 400: 40087}
    corr_chans = np.asarray([10, 20, 300, 400], dtype=int)
    col2sat = np.zeros(len(corr_chans), dtype=int)
    for i, chan in enumerate(corr_chans):
        col2sat[i] = satnorads.index(chan2sat[chan])
    size_micro_chunk = 10
    delays = np.zeros((size_micro_chunk,len(satnorads)),dtype="float64")
    delay_map = {40087:1/400, 41187:1/40}
    for i, satnum in enumerate(satnorads):
        delays[:,i] = delay_map[satnum]
    arr = np.ones((size_micro_chunk, len(corr_chans)),dtype="complex128")
    newarr=arr.copy()*0
    ou.apply_sat_delay(arr, newarr, col2sat, delays, corr_chans)
    print(delays)
    print(col2sat)
    # print(newarr)
    assert(np.allclose(newarr[:,0], 1j))
    assert(np.allclose(newarr[:,1], -1))
    assert(np.allclose(newarr[:,2], -1j))
    assert(np.allclose(newarr[:,3], 1))

def test_apply_window():
    N=4096
    win=ou.generate_window(N,type='hann')
    x=np.ones(N,dtype='complex64').reshape(1,N)
    y=ou.apply_window(x,win)
    assert(np.allclose(y.real,win))
    assert(np.allclose(y.imag,0.))

if __name__ == "__main__":
    test_apply_sat_delay()