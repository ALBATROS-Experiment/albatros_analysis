import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os.path import join

if __name__=="__main__":
    from correlations import baseband_data_classes as bdc
    from correlations import correlations as cr
    from utils import baseband_utils as butils
else:
    from .correlations import baseband_data_classes as bdc
    from .correlations import correlations as cr
    from .utils import baseband_utils as butils



def get_avg_fast(path1, path2, init_t, end_t, delay, acclen, nchunks, chanstart=0, chanend=None):
    
    files1, idxstart1 = butils.get_init_info(init_t, end_t, path1)
    files2, idxstart2 = butils.get_init_info(init_t, end_t, path2)
    # idxstart1=2502441
    # idxstart2=1647949
    print(idxstart1,idxstart2, "IDXSTARTS")
    if(delay>0):
        idxstart1+=delay
        
    else:
        idxstart2+=np.abs(delay)

    print("Starting at: ",idxstart1, "in filenum: ",files1[0], "for antenna 1")
    print("Starting at: ",idxstart2, "in filenum: ",files2[0], "for antenna 2")
    # print(files[fileidx])
    fileidx1 = 0
    fileidx2 = 0
    ant1 = bdc.BasebandFileIterator(files1,fileidx1,idxstart1,acclen,nchunks=nchunks,chanstart=chanstart,chanend=chanend)
    ant2 = bdc.BasebandFileIterator(files2,fileidx2,idxstart2,acclen,nchunks=nchunks,chanstart=chanstart,chanend=chanend)
    ncols=ant1.obj.chanend-ant1.obj.chanstart
    pol00=np.zeros((nchunks,ncols),dtype='complex64',order='c')
    m1=ant1.spec_num_start
    m2=ant2.spec_num_start
    st=time.time()
    for i, (chunk1,chunk2) in enumerate(zip(ant1,ant2)):
        t1=time.time()
        # pol00[i,:] = cr.avg_xcorr_4bit_2ant(chunk1['pol0'], chunk2['pol0'],chunk1['specnums'],chunk2['specnums'],m1+i*acclen,m2+i*acclen)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,help='Parent data directory. Should have snap1 and snap3, which have 5 digit time folders.')
    parser.add_argument("time_start",type=int, help="Start timestamp ctime")
    parser.add_argument("acclen", type=int, help="Accumulation length for averaging")
    parser.add_argument("delay", type=int, help="Delay +ve or -ve")
    parser.add_argument('-n', '--nchunks', dest='nchunks',type=int, default=560, help='Number of chunks in output file. If stop time is specfied this is overwritten. Default 560 ~ 1 hr.')
    parser.add_argument('-t', '--time_stop', dest='time_stop',type=int, default=False, help='Stop time. Overwrites nchunks if specified')
    parser.add_argument("-c", '--chans', type=int, nargs=2, help="Indices of start and end channels.")
    parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='/project/s/sievers/mohanagr/',
              help='Output directory for data and plots')
    args = parser.parse_args()

    if(args.time_stop):
        args.nchunks = int(np.floor((args.time_stop-args.time_start)*250e6/4096/args.acclen))
    else:
        args.time_stop = args.time_start + int(np.ceil(args.nchunks*args.acclen*4096/250e6))
    if(not args.chans):
        args.chans=[0,None]

    path1=join(args.data_dir,'snap3')
    path2=join(args.data_dir,'snap1')
    print(path1,path2)
    init_t = args.time_start #c#1627441379 #1627441542 #1627439234
    acclen=args.acclen
    t_acclen = acclen*4096/250e6
    delay=args.delay #-34060 #-50110 #-963933
    nchunks=args.nchunks #2947 #1959
    end_t = int(init_t + nchunks*t_acclen)
    pol00,channels=get_avg_fast(path1, path2, init_t, end_t, delay, acclen, nchunks, chanstart=args.chans[0], chanend=args.chans[1])

    fname = f"xcorr_pol00_4bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{str(args.delay)}_{args.chans[0]}_{args.chans[1]}.npz"
    fpath = join(args.outdir,fname)
    np.savez_compressed(fpath,datap00=pol00.data,maskp00=pol00.mask,chans=channels)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(5,10), dpi=200)
    plt.suptitle(f'Delay {delay}, acclen {acclen}')
    myext = np.array([np.min(channels)*125/2048,np.max(channels)*125/2048, pol00.shape[0]*t_acclen, 0])
    plt.subplot(211)
    plt.imshow(np.log10(np.abs(pol00)), aspect='auto', extent=myext,interpolation='none')
    plt.title('xcorr pol00 magnitude')
    plt.colorbar()

    ph = np.angle(pol00)
    plt.subplot(212)
    plt.imshow(ph, aspect='auto', extent=myext, cmap='RdBu',interpolation='none')
    plt.title('xcorr pol00 phase')
    plt.colorbar()

    fname = f"xcorr_pol00_4bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{str(args.delay)}_{args.chans[0]}_{args.chans[1]}.png"
    fpath = join(args.outdir,fname)
    plt.savefig(fpath)
    print(fpath)
