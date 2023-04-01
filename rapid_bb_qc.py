import numpy as np
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str,help='Path to baseband file')
    parser.add_argument('-a','--acclen', dest='acclen',type=int,default=100000,help="Accumulation length for averaging. If bigger then number of spectra, only one output row")
    parser.add_argument('-r', '--readlen', dest='readlen',type=float, default=-1,
              help='Either integer number of packets you want to read from the file OR fraction of total packets')
    parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='./',
              help='Output directory for data')
    parser.add_argument("-c", '--chans', type=int, nargs=2, help="Indices of start and end channels. Start channel index MUST be even in case of 1 bit file.")
    
    args = parser.parse_args()

    print(args.dirpath)
    if(not args.chans):
        args.chans=[0,None]
    obj=bdc.BasebandPacked(args.fpath,readlen=args.readlen,chanstart=args.chans[0],chanend=args.chans[1])
    nchunks = int(len(obj.spec_idx)/args.acclen)
    nchans=obj.chanend-obj.chanstart
    timestart=fpath.split('/')[-1][:10]
    print("File timestamp is:", timestart)
    specnum = np.arange(0,args.acclen) # we don't care about missing spectra etc. rn
    pol01=np.zeros((nchunks,nchans),dtype='complex64',order='c')

    if(obj.bit_mode==4):
        pol00=np.zeros((nchunks,nchans),dtype='float64',order='c')
        pol11=np.zeros((nchunks,nchans),dtype='float64',order='c')
        for i in range(0,nchunks):
            pol00[i,:]=cr.avg_autocorr_4bit(obj.pol0,specnum)
            pol11[i,:]=cr.avg_autocorr_4bit(obj.pol1,specnum)
            pol01[i,:]=cr.avg_xcorr_4bit(obj.pol0,obj.pol1,specnum)
        fname=f'rapid_4bit_{str(time_start)}_{str(args.acclen)}_{str(nchunks)}_{args.chans[0]}_{args.chans[1]}.png'
        fpath=os.path.join(args.outdir,fname)
        butils.plot_4bit(pol00,pol11,pol01,channels,args.acclen,time_start,fpath,minutes=False,logplot=True)
    elif(obj.bit_mode==1):
        for i in range(0,nchunks):
            pol01[i,:]=cr.avg_xcorr_1bit(obj.pol0,obj.pol1,specnum,nchans)
        fname=f'rapid_1bit_{str(time_start)}_{str(args.acclen)}_{str(nchunks)}_{args.chans[0]}_{args.chans[1]}.png'
        fpath=os.path.join(args.outdir,fname)
        plot_1bit(pol01,channels,args.acclen,time_start,fpath,minutes=False,logplot=False)
    
    print(fpath)