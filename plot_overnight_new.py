import os, sys
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import datetime, time, re
from scio import scio
import SNAPfiletools as sft
import argparse
from datetime import datetime
import matplotlib.dates as mdates
from multiprocessing import Pool
from functools import partial


def get_ts_from_name(f):
    return int(f.split('/')[-1])

#============================================================
def get_data_arrs(data_dir, ctime_start, ctime_stop, chunk_time, blocklen):
    '''
    Given the path to a Big data directory (i.e. directory contains the directories 
    labeled by the first 5 digits of the ctime date), gets all the data in some time interval.

    Parameters:
    -----------

    data_dir: str
        path to data directory

    ctime_start, ctime_stop: str
        desired start and stop time in ctime

    Returns:
    --------

    cimte_start, ctime_stop: int
        start and stop times in ctime

    pol00,pol11,pol01r,pol01i: array
        2D arrays containing the data for given time interval for autospectra 
        as well as cross spectra. pol00 corresponds to adc0 and pol11 to adc3
    '''
    print("\n################### READING DATA ###################")
    print(f'Getting data from timestamps {ctime_start} to {ctime_stop}')
    print(f"In UTC time: {datetime.utcfromtimestamp(ctime_start)} to {datetime.utcfromtimestamp(ctime_stop)}")
    print(f"In local time: {datetime.fromtimestamp(ctime_start)} to {datetime.fromtimestamp(ctime_stop)}")

    #all the dirs between the timestamps. read all, append, average over chunk length
    data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)
    data_subdirs.sort()
    
    #rough estimate of number of rows we'll read
    nrows_guess = len(data_subdirs)*((int(3600/chunk_time/blocklen)+1)+1)
    print("Starting with a guess of ", nrows_guess)
    pol00 = np.zeros((nrows_guess,2048))
    
    nrows = 0

    t1=time.time()
    new_dirs = [d+'/pol00.scio.bz2' for d in data_subdirs]
    datpol00 = scio.read_files(new_dirs)
    new_dirs = [d+'/pol11.scio.bz2' for d in data_subdirs]
    datpol11 = scio.read_files(new_dirs)
    new_dirs = [d+'/pol01r.scio.bz2' for d in data_subdirs]
    datpol01r = scio.read_files(new_dirs)
    new_dirs = [d+'/pol01i.scio.bz2' for d in data_subdirs]
    datpol01i = scio.read_files(new_dirs)
    print(time.time()-t1, f"Read {len(data_subdirs)} files")

    #average everything if blocklen>1
    myavgfunc = partial(get_avg, block=blocklen)
    if(blocklen>1):
        t1=time.time()
        with Pool(os.cpu_count()) as p:
            avgpol00 = p.map(myavgfunc,datpol00)
            avgpol11 = p.map(myavgfunc,datpol11)
            avgpol01r = p.map(myavgfunc,datpol01r)
            avgpol01i = p.map(myavgfunc,datpol01i)
        print(time.time()-t1, "averaged everything")
    else:
        avgpol00 = datpol00
        avgpol11 = datpol11
        avgpol01r = datpol01r
        avgpol01i = datpol01i

    
    t1=time.time()
    tstart=0
    tend=0
    for i, d in enumerate(avgpol00):
        print("reading", data_subdirs[i], "with size ", d.shape[0])
        if(i==0):
            pol00[:d.shape[0]] = d
            nrows+=d.shape[0]
            ts=get_ts_from_name(data_subdirs[i])+d.shape[0]*chunk_time*blocklen
            tstart=ts
        newts = get_ts_from_name(data_subdirs[i])
        diff=int((newts-ts)/chunk_time/blocklen) 
        # each cell in the plot represents a minimum time of blocklen * chunktime. 
        # That's the time resolution for the plot. Can't catch gaps < resolution.
        if(diff>0):
            print("significant diff with previous file.",diff)
            pol00[nrows:nrows+diff,:]=np.nan
            pol00=np.append(pol00, np.zeros((diff,2048)), axis=0)
            nrows+=diff
        # print(nrows, d.shape)
        # print(nrows,nrows+d.shape[0],pol00.shape,"heh")
        pol00[nrows:nrows+d.shape[0],:]=d
        nrows+=d.shape[0]
        ts=newts+d.shape[0]*chunk_time*blocklen
    tend=ts
    # print("HERE")
    #once we have pol00, we know the exact size. use it
    print("Final nrows:", nrows)
    pol00 = pol00[:nrows].copy()
    # print(pol00.shape)

    pol11 = np.zeros((nrows,2048))
    pol01r = np.zeros((nrows,2048))
    pol01i = np.zeros((nrows,2048))
    
    nrows=0
    for i in range(len(avgpol00)):
        if(i==0):
            r=avgpol11[i].shape[0]
            pol11[:r,:]=avgpol11[i]
            pol01r[:r,:]=avgpol01r[i]
            pol01i[:r,:]=avgpol01i[i]
            nrows+=r
            ts=get_ts_from_name(data_subdirs[i])+r*chunk_time*blocklen
        newts = get_ts_from_name(data_subdirs[i])
        diff=int((newts-ts)/chunk_time/blocklen)
        if(diff>1):
            pol11[nrows:nrows+diff,:]=np.nan
            pol01r[nrows:nrows+diff,:]=np.nan
            pol01i[nrows:nrows+diff,:]=np.nan
            nrows+=diff
        r=avgpol11[i].shape[0]
        pol11[nrows:nrows+r,:]=avgpol11[i]
        pol01r[nrows:nrows+r,:]=avgpol01r[i]
        pol01i[nrows:nrows+r,:]=avgpol01i[i]
        nrows+=r
        ts=newts+r*chunk_time*blocklen

    t2=time.time()
    print('Time taken to concatenate data:',t2-t1)
    print("pol00, pol11,pol01r, pol01i shape:", pol00.shape,pol11.shape,pol01r.shape,pol01i.shape)
    pol00=np.ma.masked_invalid(pol00)
    pol11=np.ma.masked_invalid(pol11)
    pol01r=np.ma.masked_invalid(pol01r)
    pol01i=np.ma.masked_invalid(pol01i)
    return pol00, pol11, pol01r, pol01i, tstart, tend

def get_avg(arr,block=10):
    '''
    Averages some array over a given block size
    '''
    iters=arr.shape[0]//block
    leftover=arr.shape[0]%block
    # print(iters,leftover)
    nrows = iters+int(leftover>0)
    ncols = arr.shape[1]
    newarr = np.zeros((nrows,ncols),dtype=arr.dtype)
    # print(f"Shape of passed arr {arr.shape} and shape of new arr {newarr.shape}")
    newarr[:iters,:] = np.mean(arr[:iters*block,:].reshape(-1,block,ncols),axis=1)
    if(leftover):
        newarr[iters,:] = np.mean(arr[iters*block:,:],axis=0)
    return newarr

def get_stats(data_arr):
    '''
    Given a 2D array containing some data chunk, returns the 
    min, median, mean, and max over that chunk.
    '''
    if logplot:
        stats = {"min":np.log10(np.min(data_arr,axis=0)), "median":np.log10(np.median(data_arr,axis=0)), 
                "mean":np.log10(np.mean(data_arr,axis=0)), "max":np.log10(np.max(data_arr,axis=0))}
    else:
        stats = {"min":np.min(data_arr,axis=0), "median":np.median(data_arr,axis=0), 
                "mean":np.mean(data_arr,axis=0), "max":np.max(data_arr,axis=0)}
    return stats

def get_vmin_vmax(data_arr):
    '''
    Automatically gets vmin and vmax for colorbar
    '''
    print("shape of passed array", data_arr.shape, data_arr.dtype)
    xx=data_arr[~data_arr.mask].data
    med = np.percentile(xx,50)
    print(med, "median")
    u=np.percentile(xx,99)
    b=np.percentile(xx,1)
    xx_clean=xx[(xx<=u)&(xx>=b)] # remove some outliers for better plotting
    stddev = np.std(xx_clean)
    vmin= max(med - 2*stddev,10**7)
    vmax = med + 2*stddev
    print("vmin, vmax are", vmin, vmax)
    return vmin,vmax   

def get_ylim_times(t_i,t_f,tz):
    '''
    Gets the y limits in matplotlib's date format for a given initial time
    and final time. t_i and t_f must be given in ctime
    '''

    if tz=="utc":
        y_lims = list(map(datetime.utcfromtimestamp, [t_i, t_f]))
    elif tz=="local":
        y_lims = list(map(datetime.fromtimestamp, [t_i, t_f])) #local is not observing site! it's computer's tz
    else:
        print("Invalid timezone")

    y_lims_plt = mdates.date2num(y_lims)
    print(y_lims_plt, "y lims plt")
    return y_lims_plt

#================= plotting functions =======================
def full_plot(data_arrs):
    '''
    Makes a plot that contains autospectra waterfalls for each pol, as well
    as some statistics (min,max,med,mean spectra), and cross spectra
    '''

    pol00,pol11,pol01,tstart,tend = data_arrs

    pol00_stats = get_stats(pol00)
    pol11_stats = get_stats(pol11)
    
    
    if logplot is True:
        pol00 = np.log10(pol00)
        pol11 = np.log10(pol11)
    
    y_extent = get_ylim_times(tstart,tend,timezone)
    ticks = np.linspace(y_extent[0], y_extent[1],20)
    print(y_extent)
 
    myext = np.array([0,125,y_extent[1],y_extent[0]])
        
    plt.figure(figsize=(18,10), dpi=200)
    plt.subplot(2,3,1)

    plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
    plt.title('pol00')
    cb00 = plt.colorbar()
    plt.yticks(ticks)
    plt.gca().yaxis.set_major_formatter(datetimefmt)
    
    plt.subplot(2,3,4)
    plt.imshow(pol11, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
    plt.title('pol11')
    plt.colorbar()
    plt.yticks(ticks)
    plt.gca().yaxis.set_major_formatter(datetimefmt)

    plt.subplot(2,3,2)
    plt.title('Basic stats for frequency bins')
    plt.plot(freq, pol00_stats["max"], 'r-', label='Max')
    plt.plot(freq, pol00_stats["min"], 'b-', label='Min')
    plt.plot(freq, pol00_stats["mean"], 'k-', label='Mean')
    plt.plot(freq, pol00_stats["median"], color='#666666', linestyle='-', label='Median')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('pol00')

    plt.subplot(2,3,5)
    plt.plot(freq, pol11_stats["max"], 'r-', label='Max')
    plt.plot(freq, pol11_stats["min"], 'b-', label='Min')
    plt.plot(freq, pol11_stats["mean"], 'k-', label='Mean')
    plt.plot(freq, pol11_stats["median"], color='#666666', linestyle='-', label='Median')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('pol11')
    
    plt.legend(loc='lower right', fontsize='small')

    plt.subplot(2,3,3)
    plt.imshow(np.log10(np.abs(pol01)), vmin=3,vmax=8,aspect='auto',extent=myext)
    plt.title('pol01 magnitude')
    plt.colorbar()
    plt.gca().set_yticklabels([])
    
    plt.subplot(2,3,6)
    plt.imshow(np.angle(pol01), vmin=-np.pi, vmax=np.pi, aspect='auto', extent=myext, cmap='RdBu')
    plt.title('pol01 phase')
    plt.colorbar()
    plt.gca().set_yticklabels([])

    plt.suptitle(f'{datetime.utcfromtimestamp(ctime_start)} UTC to {datetime.utcfromtimestamp(ctime_stop)} UTC, Averaged over {blocksize} chunks')

    outfile = os.path.join(outdir,'output'+ '_' + str(ctime_start) + '_' + str(ctime_stop) + '.png')
    plt.savefig(outfile)
    
    print('Wrote ' + outfile)



#============================================================
def main():

    parser = argparse.ArgumentParser()
    # parser.set_usage('python plot_overnight_data.py <data directory> <start time as YYYYMMDD_HHMMSS or ctime> <stop time as YYYYMMDD_HHMMSS or ctime> [options]')
    # parser.set_description(__doc__)
    parser.add_argument('data_dir', type=str,help='Direct data directory')
    parser.add_argument("time_start", type=str, help="Start time YYYYMMDD_HHMMSS or ctime")
    parser.add_argument("time_stop", type=str, help="Stop time YYYYMMDD_HHMMSS or ctime")
    parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='.',
              help='Output plot directory [default: .]')
    
    parser .add_argument('-n', '--length', dest='readlen', type=int, default=1000, help='length of integration time in seconds')
    parser.add_argument("-a", "--avglen",dest="blocksize",default=0,type=int,help="number of chunks (rows) of direct spectra to average over. One chunk is roughly 6 seconds.")

    parser.add_argument("-l", "--logplot", dest='logplot', default = True, action="store_true", help="Plot in logscale")
    parser.add_argument("-p", "--plottype",dest="plottype",default="full",type=str,
        help="Type of plot to generate. 'full': pol00 and pol11 waterfall autospectra, min/max/mean/med autospectra, waterfall cross spectra. 'waterfall': same as 1, but no stats")
    parser.add_argument("-t", "--timezone", dest='timezone', default = "utc", type=str, help="Timezone to use for plot axis. Can do 'utc' or 'local'")
    parser.add_argument("-vmi", "--vmin", dest='vmin', default = None, type=float, help="minimum for colorbar. if nothing is specified, vmin is automatically set")
    parser.add_argument("-vma", "--vmax", dest='vmax', default = None, type=float, help="maximum for colorbar. if nothing is specified, vmax is automatically set")
    parser.add_argument("-d", "--datetimefmt", dest='datetimefmt', default = "%b-%d %H:%M", type=str, help="Format for dates on axes of plots")
    

    args = parser.parse_args()

    #=============== defining some global variables ===============#
    global freq, timezone, logplot, vmin, vmax, ctime_start, ctime_stop, blocksize, outdir, datetimefmt
    
    timezone = args.timezone
    vmin = args.vmin
    vmax = args.vmax
    logplot=args.logplot
    blocksize = args.blocksize
    outdir = args.outdir
    datetimefmt = mdates.DateFormatter(args.datetimefmt)
    #=============================================================#
    
    # figuring out if human time or ctime was passed with pattern matching
    rx_human = re.compile(r'^\d{8}_\d{6}$')
    rx_ctime = re.compile(r'^\d{10}$')
    m1 = rx_human.search(args.time_start)
    m2 = rx_ctime.search(args.time_start)
    if(m1):
        ctime_start = sft.timestamp2ctime(args.time_start)
        ctime_stop = sft.timestamp2ctime(args.time_stop)
    elif(m2):
        ctime_start = int(args.time_start)
        ctime_stop = int(args.time_stop)
    else:
        raise ValueError("INVALID time format entered.")

    #================= reading data =================#
    pol00,pol11,pol01r,pol01i, tstart, tend = get_data_arrs(args.data_dir, ctime_start, ctime_stop, 6.44, args.blocksize)
    # import sys
    # sys.exit(0)

    pol01 = pol01r + 1J*pol01i
    freq = np.linspace(0, 125, np.shape(pol00)[1]) #125 MHz is max frequency
    
    
    #============ setting vmin and vmax ============#
    # setting vmin and vmax
    if vmin==None and vmax==None:
        vmin,vmax = get_vmin_vmax(pol00)
        if logplot==True:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

    #============ and finally: plotting! ============#
    if args.plottype == "full":
        full_plot([pol00,pol11,pol01, tstart, tend])
    
    
if __name__ == '__main__':
    main()
