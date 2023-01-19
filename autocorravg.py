import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import glob
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
from utils import baseband_utils as butils
import argparse

def get_avg_fast(path, init_t, end_t, acclen, nchunks, chanstart=0, chanend=None):
    
    idxstart, fileidx, files = butils.get_init_info(init_t, end_t, path)
    print("Starting at: ",idxstart, "in filenum: ",fileidx)
    print(files[fileidx])

    ant1 = bdc.BasebandFileIterator(files,fileidx,idxstart,acclen,nchunks=nchunks,chanstart=chanstart,chanend=chanend)
    if(ant1.obj.bit_mode!=4):
        raise NotImplementedError(f"BIT MODE {ant1.obj.bit_mode} IS NOT SUPPORTED BY THIS SCRIPT. Do you want to use autocorravg1bit.py?")
    ncols=ant1.obj.chanend-ant1.obj.chanstart
    pol00=np.zeros((nchunks,ncols),dtype='float64',order='c')
    pol11=np.zeros((nchunks,ncols),dtype='float64',order='c')
    pol01=np.zeros((nchunks,ncols),dtype='complex64',order='c')
    j=ant1.spec_num_start
    m=ant1.spec_num_start
    st=time.time()
    for i, chunk in enumerate(ant1):
        t1=time.time()
        pol00[i,:] = cr.avg_autocorr_4bit(chunk['pol0'],chunk['specnums'])
        pol11[i,:] = cr.avg_autocorr_4bit(chunk['pol1'],chunk['specnums'])
        pol01[i,:] = cr.avg_xcorr_4bit(chunk['pol0'], chunk['pol1'],chunk['specnums'])
        t2=time.time()
        print("time taken for one loop", t2-t1)
        j=ant1.spec_num_start
        print("After a loop spec_num start at:", j, "Expected at", m+(i+1)*acclen)
        print(i+1,"CHUNK READ")
    print("Time taken final:", time.time()-st)
    pol00 = np.ma.masked_invalid(pol00)
    pol11 = np.ma.masked_invalid(pol11)
    pol01 = np.ma.masked_invalid(pol01)
    return pol00,pol11,pol01,ant1.obj.channels

def get_plot_lims(pol,acclen):

    # numpy percentile method ignores mask and may generate garbage with 0s (missing specs). 
    # Pivot to using mean if acclen too small.

    if(acclen>250000):
        med = np.mean(pol)
        xx=np.ravel(pol).copy()
        u=np.percentile(xx,99)
        b=np.percentile(xx,1)
        xx_clean=xx[(xx<=u)&(xx>=b)] # remove some outliers for better plotting
        stddev = np.std(xx_clean)
    else:
        med = np.mean(pol)
        stddev = np.std(pol)
    vmin= max(med - 2*stddev,1)
    vmax = med + 2*stddev
    print(med,vmin,vmax)
    return med,vmin,vmax

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,help='Parent data directory. Should have 5 digit time folders.')
    parser.add_argument("time_start",type=int, help="Start timestamp ctime")
    parser.add_argument("acclen", type=int, help="Accumulation length for averaging")
    parser.add_argument('-n', '--nchunks', dest='nchunks',type=int, default=560, help='Number of chunks in output file. If stop time is specfied this is overwritten. Default 560 ~ 1 hr.')
    parser.add_argument('-t', '--time_stop', dest='time_stop',type=int, default=False, help='Stop time. Overwrites nchunks if specified')
    parser.add_argument("-c", '--chans', type=int, nargs=2, help="Indices of start and end channels.")
    parser.add_argument("-l", "--logplot", action="store_true", help="Plot in logscale")
    parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='/scratch/s/sievers/mohanagr/',
              help='Output directory for data and plots')
    args = parser.parse_args()

    if(args.time_stop):
        args.nchunks = int(np.floor((args.time_stop-args.time_start)*250e6/4096/args.acclen))
    else:
        args.time_stop = args.time_start + int(np.ceil(args.nchunks*args.acclen*4096/250e6))
    if(not args.chans):
        args.chans=[0,None]
    
    print("nchunks is: ", args.nchunks,"and stop time is ", args.time_stop)
    # assert(1==0)
    pol00,pol11,pol01,channels = get_avg_fast(args.data_dir, args.time_start, args.time_stop, args.acclen, args.nchunks, args.chans[0], args.chans[1])
    print("RUN 1 DONE")

    # pol01_2,channels = get_avg_fast_1bit(args.data_dir, args.time_start, args.time_stop, args.acclen, args.nchunks)
    # print("RUN 2 DONE")

    # diff1=np.sum(np.abs(pol01_1-pol01_2),axis=1)
    # print(diff1) checked that this is zero. 

    import os

    fname = f"pols_4bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.npz"
    fpath = os.path.join(args.outdir,fname)
    np.savez_compressed(fpath,datap01=pol01.data,maskp01=pol01.mask,datap00=pol00.data,maskp00=pol00.mask,\
        datap11=pol11.data,maskp11=pol11.mask,chans=channels)
    
    freq = channels*125/2048 #MHz
    pol00_med = np.median(pol00, axis=0)
    pol11_med = np.median(pol11, axis=0)
    pol00_mean = np.mean(pol00, axis=0)
    pol11_mean = np.mean(pol11, axis=0)
    pol00_max = np.max(pol00, axis=0)
    pol11_max = np.max(pol11, axis=0)
    pol00_min = np.min(pol00, axis=0)
    pol11_min = np.min(pol11, axis=0)
    med,vmin,vmax=get_plot_lims(pol00,args.acclen)
    med2,vmin2,vmax2=get_plot_lims(pol11,args.acclen)
    pol01_mag = np.abs(pol01)
    if(args.logplot):
        pol00 = np.log10(pol00)
        pol11 = np.log10(pol11)
        pol00_med = np.log10(pol00_med)
        pol11_med = np.log10(pol11_med)
        pol00_mean = np.log10(pol00_mean)
        pol11_mean = np.log10(pol11_mean)
        pol00_max = np.log10(pol00_max)
        pol11_max = np.log10(pol11_max)
        pol00_min = np.log10(pol00_min)
        pol11_min = np.log10(pol11_min)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
        vmin2 = np.log10(vmin2)
        vmax2 = np.log10(vmax2)
        pol01_mag = np.log10(pol01_mag)

    from matplotlib import pyplot as plt

    plt.figure(figsize=(18,10), dpi=200)
    t_acclen = args.acclen*2048/125e6 #seconds
    myext = np.array([np.min(channels)*125/2048,np.max(channels)*125/2048, pol00.shape[0]*t_acclen/60, 0])

    plt.subplot(2,3,1)
    plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
    plt.title(f'pol00 - minutes since {args.time_start}')
    cb00 = plt.colorbar()
    cb00.ax.plot([0, 1], [7.0]*2, 'w')

    plt.subplot(2,3,4)
    plt.imshow(pol11, vmin=vmin2, vmax=vmax2, aspect='auto', extent=myext)
    plt.title('pol11')
    plt.colorbar()

    plt.subplot(2,3,2)
    plt.title('Basic stats for frequency bins')
    plt.plot(freq, pol00_max, 'r-', label='Max')
    plt.plot(freq, pol00_min, 'b-', label='Min')
    plt.plot(freq, pol00_mean, 'k-', label='Mean')
    plt.plot(freq, pol00_med, color='#666666', linestyle='-', label='Median')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('pol00')

    plt.subplot(2,3,5)
    plt.plot(freq, pol11_max, 'r-', label='Max')
    plt.plot(freq, pol11_min, 'b-', label='Min')
    plt.plot(freq, pol11_mean, 'k-', label='Mean')
    plt.plot(freq, pol11_med, color='#666666', linestyle='-', label='Median')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('pol11')
    plt.legend(loc='lower right', fontsize='small')

    plt.subplot(2,3,3)
    plt.imshow(pol01_mag, aspect='auto', extent=myext)
    plt.title('pol01 magnitude')
    plt.ylabel('minutes')
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.imshow(np.angle(pol01), vmin=-np.pi, vmax=np.pi, aspect='auto', extent=myext, cmap='RdBu')
    plt.ylabel('minutes')
    plt.title('pol01 phase')
    plt.colorbar()


    fname=f'pols_4bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.png'
    fpath=os.path.join(args.outdir,fname)
    plt.savefig(fpath)
    print(fpath)




        

                
                
            
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    