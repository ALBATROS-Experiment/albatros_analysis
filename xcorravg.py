import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import glob
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
from utils import baseband_utils as butils

def get_avg_fast(path1, path2, init_t, end_t, delay, acclen, nchunks, chanstart=0, chanend=None):
    
    idxstart1, fileidx1, files1 = butils.get_init_info(init_t, end_t, path1)
    idxstart2, fileidx2, files2 = butils.get_init_info(init_t, end_t, path2)

    if(delay>0):
        idxstart2+=delay
    else:
        idxstart1+=np.abs(delay)

    # print("Starting at: ",idxstart, "in filenum: ",fileidx)
    # print(files[fileidx])

    ant1 = bdc.BasebandFileIterator(files1,fileidx1,idxstart1,acclen,nchunks=nchunks,chanstart=chanstart,chanend=chanend)
    ant2 = bdc.BasebandFileIterator(files2,fileidx2,idxstart2,acclen,nchunks=nchunks,chanstart=chanstart,chanend=chanend)
    ncols=ant1.obj.chanend-ant1.obj.chanstart
    pol00=np.zeros((nchunks,ncols),dtype='complex64',order='c')
    m1=ant1.spec_num_start
    m2=ant2.spec_num_start
    st=time.time()
    for i, (chunk1,chunk2) in enumerate(zip(ant1,ant2)):
        t1=time.time()
        pol00[i,:] = cr.avg_xcorr_4bit_2ant(chunk1['pol0'], chunk2['pol0'],chunk1['specnums'],chunk2['specnums'],m1+i*acclen,m2+i*acclen)
        t2=time.time()
        print("time taken for one loop", t2-t1)
        j=ant1.spec_num_start
        print("After a loop spec_num start at:", j, "Expected at", m1+(i+1)*acclen)
        print(i+1,"CHUNK READ")
    print("Time taken final:", time.time()-st)
    pol00 = np.ma.masked_invalid(pol00)
    return pol00,ant1.obj.channels

if __name__=="__main__":
    path1='/project/s/sievers/albatros/uapishka/baseband/snap1'
    path2='/project/s/sievers/albatros/uapishka/baseband/snap3'
    init_t = 1627441542 #1627439234
    acclen=10000
    t_acclen = acclen*4096/250e6
    delay=-963933#33005
    nchunks=1000
    end_t = int(init_t + nchunks*t_acclen)
    pol00,channels=get_avg_fast(path1, path2, init_t, end_t, delay, acclen, nchunks, chanstart=35, chanend=50)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(5,10), dpi=200)
    plt.suptitle(f'Delay {delay}, acclen {acclen}')
    myext = np.array([np.min(channels)*125/2048,np.max(channels)*125/2048, pol00.shape[0]*t_acclen/60, 0])
    plt.subplot(211)
    plt.imshow(np.log10(np.abs(pol00)), aspect='auto', extent=myext,interpolation='none')
    plt.title('pol01 magnitude')
    plt.colorbar()

    plt.subplot(212)
    plt.imshow(np.arctan2(pol00.imag,pol00.real), aspect='auto', extent=myext, cmap='RdBu',interpolation='none')
    plt.title('pol01 phase')
    plt.colorbar()

    plt.savefig('/scratch/s/sievers/mohanagr/mytempfig.png')
    print('/scratch/s/sievers/mohanagr/mytempfig.png')



                


    