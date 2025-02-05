import pytest
import numpy as np
from src.correlations import unpacking as unpk
from src.correlations import correlations as cr
def str_to_bits(str):
    s = 0
    return sum([int(str[i]) * 2 ** (7 - i) for i in range(0, 8)])


@pytest.fixture
def ant1():
    # 8 channels
    nspec=10
    r = np.ones(8*nspec, dtype="int64").reshape(-1, 8)  # first row all 1
    im=r.copy()
    # pol0=r+1j*im
    # pol1=pol0.copy()
    return r,im
@pytest.fixture
def ant2():
    # 8 channels
    nspec=10
    r = np.ones(8*nspec, dtype="int64").reshape(-1, 8)  # first row all 1
    im=-1*r.copy()
    # pol0=r-1j*im
    # pol1=pol0.copy()
    return r,im

@pytest.fixture
def ant1_rand():
    # 8 channels
    nspec=10
    r = 2*(np.random.randn(nspec*8)>0)-1
    r = r.astype('int64').reshape(-1,8)
    im = 2*(np.random.randn(nspec*8)>0)-1
    im = im.astype('int64').reshape(-1,8)
    return r,im

@pytest.fixture
def ant2_rand():
    # 8 channels
    nspec=10
    r = 2*(np.random.randn(nspec*8)>0)-1
    r = r.astype('int64').reshape(-1,8)
    im = 2*(np.random.randn(nspec*8)>0)-1
    im = im.astype('int64').reshape(-1,8)
    return r,im

def packed_1bit(ant):
    p0r = ant[0].copy()
    p0im = ant[1].copy()
    p1r=p0r.copy()
    p1im=p0im.copy()
    nchan=ant[0].shape[1]
    nspec=ant[0].shape[0]
    pshape = nchan//2 * nspec  # nchan/2 * nspec #0.25 byte/chan/pol
    packed = np.zeros((1, pshape), dtype="uint8") # all of nspec form 1 packet
    p0r[p0r < 0] = p0r[p0r < 0] + 1 #-1 -> 0
    p1r[p1r < 0] = p1r[p1r < 0] + 1
    p0im[p0im < 0] = p0im[p0im < 0] + 1
    p1im[p1im < 0] = p1im[p1im < 0] + 1
    for i in range(0, nspec):
        for j in range(0, nchan//2):
            packed[0, i * 4 + j] = (
                  (p0r[i, 2 * j] << 7)
                + (p0im[i, 2 * j] << 6)
                + (p1r[i, 2 * j] << 5)
                + (p1im[i, 2 * j] << 4)
                + (p0r[i, 2 * j + 1] << 3)
                + (p0im[i, 2 * j + 1] << 2)
                + (p1r[i, 2 * j + 1] << 1)
                + (p1im[i, 2 * j + 1])
            )
    return packed

def test_1bit_sortpols(ant1, ant2):
    packet_ant1 = packed_1bit(ant1)
    packet_ant2 = packed_1bit(ant2)
    length_channels=8
    bit_mode=1
    rowstart=5
    rowend=6
    chanstart=2
    chanend=3
    pol0,pol1=unpk.sortpols(packet_ant1,length_channels, bit_mode, rowstart, rowend, chanstart, chanend)
    assert(pol0[0,0]==192)
    assert(pol1[0,0]==192)
    rowstart=9
    rowend=10
    chanstart=4
    chanend=7
    pol0,pol1=unpk.sortpols(packet_ant2,length_channels, bit_mode, rowstart, rowend, chanstart, chanend)
    assert(pol0[0,0]==168)
    assert(pol1[0,0]==168)

# def test_1bit_xcorr_2ant(ant1, ant2):
#     packet_ant1 = packed_1bit(ant1)
#     packet_ant2 = packed_1bit(ant2)
#     length_channels=8
#     bit_mode=1
#     rowstart=0
#     rowend=10
#     chanstart=2
#     chanend=4
#     pol0_a1,pol1_a1=unpk.sortpols(packet_ant1,length_channels, bit_mode, rowstart, rowend, chanstart, chanend)
#     pol0_a2,pol1_a2=unpk.sortpols(packet_ant2,length_channels, bit_mode, rowstart, rowend, chanstart, chanend)
    

#     #missing simulation 1
#     xcorr, rowcount=cr.avg_xcorr_1bit_vanvleck_2ant(pol0_a1,pol0_a2,2,np.arange(4,10),np.arange(0,10),0,0)
#     assert(rowcount==6)
#     # print(xcorr)

#     specnum2=np.asarray([0,1,2,3,4,5,8,9])
#     xcorr, rowcount=cr.avg_xcorr_1bit_vanvleck_2ant(pol0_a1,pol0_a2,2,np.arange(4,10),specnum2,0,0)
#     assert(rowcount==4)
#     # print(xcorr)

def test_1bit_xcorr_2ant_rand(ant1_rand, ant2_rand):
    # print(ant1_rand)
    R0,I0 = ant1_rand
    
    R1,I1 = ant2_rand
    print(R1)
    # print((R0+1j*I0)[0,:2])
    # print((R1+1j*I1)[0,:2])
    # print(R0.shape)
    packet_ant1 = packed_1bit(ant1_rand)
    packet_ant2 = packed_1bit(ant2_rand)
    

    # print(packet_ant1[0,0])
    # print(packet_ant2[0,0])

    length_channels=8
    bit_mode=1
    rowstart=0
    rowend=10
    chanstart=0
    chanend=8
    pol0_a1,pol1_a1=unpk.sortpols(packet_ant1,length_channels, bit_mode, 0, 10, chanstart, chanend)
    pol0_a2,pol1_a2=unpk.sortpols(packet_ant2,length_channels, bit_mode, 0, 10, chanstart, chanend)
    # print(pol0_a1-pol1_a1)
    # print(pol0_a2-pol1_a2)
    specnum0 = np.arange(rowend-rowstart)
    specnum1 = np.arange(rowend-rowstart)
    mask=np.ones((rowend-rowstart),dtype='bool')

    #sim 1
    m0=mask.copy()
    m1=mask.copy()
    m0[0:4]=False
    m1[6:8]=False
    print(m0)
    print(m0.sum())
    #meant to simulate BFI pol0 and pol1
    temp_p0 = np.zeros(pol0_a1.shape,dtype=pol0_a1.dtype)
    temp_p1 = np.zeros(pol0_a1.shape,dtype=pol0_a1.dtype)
    #only fill with rows that were not missing
    temp_p0[:m0.sum()] = pol0_a1[m0] #fill the first len(non-missing rows) with non-missing rows
    temp_p1[:m1.sum()] = pol0_a2[m1]

    m=m0&m1

    true = np.sum(R0[m]*R1[m],axis=0),np.sum(I0[m]*I1[m],axis=0),np.sum(R0[m]*I1[m],axis=0),np.sum(R1[m]*I0[m],axis=0)
    
    test, rowcount=cr.avg_xcorr_1bit_vanvleck_2ant(temp_p0,temp_p1,chanend-chanstart,specnum0[m0].copy(),specnum1[m1].copy(),0,0)
    assert(rowcount==m.sum())
    for i in range(4):
        assert(np.array_equal(true[i],test[i].astype('int64')))