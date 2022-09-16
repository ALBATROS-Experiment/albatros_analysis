import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import glob
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
from utils import baseband_utils as butils

def get_avg_fast(path0, path1, init_t0, end_t0, init_t1, end_t1, delay, acclen, nchunks, chanstart=0, chanend=None):
    
    # delay offset is always of timestream2 w.r.t. timestream1.

    idxstart0, fileidx0, files0 = butils.get_init_info(init_t0, end_t0, path0)
    idxstart1, fileidx1, files1 = butils.get_init_info(init_t1, end_t1, path1)

    print("Antenna 1 starting at: ",idxstart0, "in filenum: ",fileidx0)
    print(files0[fileidx0])
    print("Antenna 1 starting at: ",idxstart1, "in filenum: ",fileidx1)
    print(files1[fileidx1])

    if(delay>0):
        idxstart1+=delay
    else:
        idxstart0+=delay

    obj0 = bdc.BasebandPacked(files0[fileidx0],chanstart,chanend)
    obj1 = bdc.BasebandPacked(files1[fileidx1],chanstart,chanend)

    channels=obj0.channels[chanstart:chanend]
    assert(obj0.pol0.shape[1]==obj1.pol0.shape[1])
    assert(obj0.bit_mode==4)
    assert(obj1.bit_mode==4)

    if(chanend):
        ncols = chanend-chanstart
    else:
        ncols=obj0.length_channels
        

    objlen0=obj0.pol0.shape[0] # remember that zeros are added in place of missing data in 4 bit
    objlen1=obj1.pol0.shape[0]

    pol0xpol0=np.zeros((nchunks,ncols),dtype='complex64',order='c')
    file_spec_gap0=0
    file_spec_gap1=0
    fc0=0 #file counter
    fc1=0

    st=time.time()
    for i in range(nchunks):
        rem0=acclen;rem1=acclen
        rowcount=0
        while(True):
            l0=objlen0-idxstart0
            l1=objlen1-idxstart1
            if(l0<=rem0 or l1<=rem1):
                if(l0<=rem0 and l1>rem1):
                    rowstart0, rowend0 = butils.get_rows_from_specnum(idxstart0,objlen0,obj0.spec_idx)
                    rowstart1, rowend1 = butils.get_rows_from_specnum(idxstart1,idxstart1+l0,obj1.spec_idx)
                    fc0+=1
                    idxstart0=0
                    file_spec_gap0 = -(obj0.spec_num[-1]+obj0.spectra_per_packet)
                    arr,count = cr.avg_xcorr_4bit_2ant(obj0.pol0, obj1.pol0, obj0.spec_idx, obj1.spec_idx, \
                        idxstart0, objlen0, idxstart1, idxstart1+l0, rowstart0, rowend0, rowstart1, rowend1)
                    obj0 = bdc.BasebandPacked(files0[fileidx0+fc0],chanstart,chanend)
                    file_spec_gap0 += obj0.spec_num[0]
                    file_spec_gap0 = int(file_spec_gap0)
                    if(file_spec_gap0>0):
                        raise RuntimeError("SPEC GAP NOTICED BETWEEN ANT 1 FILES")
                    objlen0 = obj0.shape[0]
                    rem0 = rem0 - l0
                    rem1 = rem1 - l0
                if(l1<=rem1 and l0>rem0):
                    rowstart0, rowend0 = butils.get_rows_from_specnum(idxstart0,idxstart0+l1,obj0.spec_idx)
                    rowstart1, rowend1 = butils.get_rows_from_specnum(idxstart1,objlen1,obj1.spec_idx)
                    fc1+=1
                    idxstart1=0
                    file_spec_gap1 = -(obj1.spec_num[-1]+obj1.spectra_per_packet)
                    obj1 = bdc.BasebandPacked(files1[fileidx1+fc1],chanstart,chanend)
                    file_spec_gap1 += obj1.spec_num[0]
                    file_spec_gap1 = int(file_spec_gap1)
                    if(file_spec_gap1>0):
                        raise RuntimeError("SPEC GAP NOTICED BETWEEN ANT 1 FILES")
                    objlen1 = obj1.shape[0]
                    rem0 = rem0 - l1
                    rem1 = rem1 - l1
                else:

                


    