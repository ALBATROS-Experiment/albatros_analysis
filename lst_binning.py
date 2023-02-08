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
import pytz
import skyfield.api as sf


def get_ts_from_name(f):
    return int(f.split('/')[-1])

def get_localtime_from_UTC(tstamp, mytz):
    return datetime.fromtimestamp(int(tstamp),tz=pytz.utc).astimezone(tz=mytz)

def get_binned(fname,ctime_start,ctime_stop):
    
    # data_dir='/project/s/sievers/mohanagr/uapishka_aug_oct_2022/data_auto_cross/'
    data_dir='/project/s/sievers/mohanagr/uapishka_franken_oct_nov_2022/data_auto_cross/SNAP3/'
    # data_dir='/project/s/sievers/albatros/marion/albatros-hydroshack/data_auto_cross/'
    data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)
    data_subdirs.sort()
    new_dirs = [d+f'/{fname}.scio.bz2' for d in data_subdirs]
    datpol00 = scio.read_files(new_dirs)

    print(data_subdirs)
    # data_subdirs = ['/home/mohan/Projects/direct/16272/1627202093']

    ts=sf.load.timescale()
    # jd = tstart/86400+2440587.5 
    # t=ts.ut1_jd(jd)
    earth_loc = sf.wgs84.latlon(51.4641932, -68.2348603,300)
    dH = 24/1440
    # print(earth_loc.lst_hours_at(t))
    binned = [np.asarray([]).reshape(-1,2048)]*1440
    for i,pol00 in enumerate(datpol00):
        tstamp = get_ts_from_name(data_subdirs[i])
        print("File timestamp is ", tstamp)
        if(pol00 is None):
            pol00_1 = scio.read(os.path.join(data_subdirs[i], f"{fname}.scio.bz2"))
            if(pol00_1 is None):
                print("YO WTF, READ", data_subdirs[i])
                continue
        mytstamps = tstamp+np.arange(0,pol00.shape[0])*6.44
        myjds = mytstamps/86400 + 2440587.5
        # print(ts.ut1_jd(myjds))
        lsthrs=earth_loc.lst_hours_at(ts.ut1_jd(myjds))
        # print("LST hours are:",lsthrs )
        lstbins = np.floor(lsthrs/dH).astype(int)
        # print(lstbins)
        branch_points=list(np.where(np.diff(lstbins)!=0)[0])
        branch_points.append(pol00.shape[0]-1)
        # print(branch_points)
        st=0
        for i in range(len(branch_points)):
            b=branch_points[i]
            en=b+1
            # print("branch point",b,"lst bin", lstbins[b])
            # print(st,en,lstbins[b])
            binned[lstbins[b]]=np.vstack([binned[lstbins[b]],pol00[st:en,:]]) #prealocated arrays would prolly speeden this up
            st=en
    return binned

# def myredux(xx):
#     u=np.percentile(xx,99)
#     b=np.percentile(xx,1)
#     xx_clean=xx[(xx<=u)&(xx>=b)]
#     return np.mean(xx_clean)

def myredux(bigarr):
    # def redux(xx):
    #     u=np.percentile(xx,50)
    #     b=np.percentile(xx,1)
    #     xx_clean=xx[(xx<=u)&(xx>=b)]
    #     return np.mean(xx_clean)
    # return np.apply_along_axis(redux,0,bigarr)
    return np.median(bigarr,axis=0)

def reduce_binned(binned,nbins,nchan):
    counts=np.zeros(nbins)
    bmedian = np.zeros((nbins,nchan)) #bmedian = bin median
    bmean = np.zeros((nbins,nchan))
    for i in range(nbins):
        counts[i]=binned[i].shape[0]
        if(counts[i]>0):
            # bmean[i,:]=np.mean(binned[i],axis=0)
            bmean[i,:] = np.mean(binned[i],axis=0)
            bmedian[i,:]=np.median(binned[i],axis=0)
        
    return {'counts':counts,'mean':bmean,'median':bmedian}

def reduce_binned_parallel(binned,nbins,nchan):
    # counts=np.zeros(nbins)
    # bmedian = np.zeros((nbins,nchan)) #bmedian = bin median
    bmean = np.zeros((nbins,nchan))

    # custom_mean = lambda bigarr: np.apply_along_axis(myredux,0,bigarr)
    # custom_median = lambda bigarr: np.median(bigarr,axis=0)

    with Pool(os.cpu_count()) as p:
        means = p.map(myredux, binned)
        # medians = p.map(custom_median,binned)
    
    bmean[:] = np.asarray(means)
    # bmedian[:] = np.asarray(medians)
        
    return {'mean':bmean}

if __name__ == '__main__':
    
    #marion
    # ctime_start=1640200237
    # ctime_stop=1647597878
    # mytz=pytz.timezone('Africa/Johannesburg')

    #uapishka
    # ctime_start= 1661011607
    # ctime_stop = 1666620593
    # mytz=pytz.timezone('US/Eastern')

    ctime_start= 1666816036
    ctime_stop = 1667663037
    mytz=pytz.timezone('US/Eastern')
    
    sttime=get_localtime_from_UTC(ctime_start,mytz).strftime("%b-%d %H:%M")
    entime=get_localtime_from_UTC(ctime_stop,mytz).strftime("%b-%d %H:%M")

    binned=get_binned('pol01r',ctime_start,ctime_stop)
    t1=time.time()
    statsreal=reduce_binned(binned,1440,2048)
    t2=time.time()
    print("time taken for reduction",t2-t1)

    binned=get_binned('pol01i',ctime_start,ctime_stop)
    statsimag=reduce_binned(binned,1440,2048)

    meanz = statsreal['mean'] + 1J*statsimag['mean']
    medianz=statsreal['median'] + 1J*statsimag['median']

    # f=plt.gcf()
    # f.set_size_inches(5,15)
    
    # plt.suptitle(f'Plotting from: {sttime} to {entime}')
    # myext=[0, 125, 24, 0]
    # plt.subplot(311)
    # plt.title("Mean")
    # plt.imshow(np.log10(stats['mean']),vmin=7,vmax=8.2,extent=myext,aspect='auto')
    # plt.colorbar()

    # plt.subplot(312)
    # plt.title("Median")
    # plt.imshow(np.log10(stats['median']),vmin=7,vmax=8.2,extent=myext,aspect='auto')
    # plt.colorbar()

    # plt.subplot(313)
    # plt.title('bin count')
    # plt.plot(np.arange(0,1440),stats['counts'])

    f=plt.gcf()

    f.set_size_inches(10,10)
    
    plt.suptitle(f'Plotting from: {sttime} to {entime}')
    myext=[0, 125, 24, 0]
    plt.subplot(221)
    plt.title("Mean xpower")
    plt.imshow(np.log10(np.abs(meanz)),vmin=3,vmax=8,extent=myext,aspect='auto')
    plt.colorbar()

    plt.subplot(222)
    plt.title("Mean phase")
    plt.imshow(np.angle(meanz),extent=myext,aspect='auto',cmap='RdBu')
    plt.colorbar()

    plt.subplot(223)
    plt.title("Median")
    plt.imshow(np.log10(np.abs(medianz)),vmin=3,vmax=8,extent=myext,aspect='auto')
    plt.colorbar()

    plt.subplot(224)
    plt.title("Median phase")
    plt.imshow(np.angle(medianz),extent=myext,aspect='auto',cmap='RdBu')
    plt.colorbar()

    # plt.subplot(313)
    # plt.title('bin count')
    # plt.plot(np.arange(0,1440),stats['counts'])

    
    plt.savefig('/project/s/sievers/mohanagr/lstbinned_franken_uapishka.png')
    print('/project/s/sievers/mohanagr/lstbinned_franken_uapishka.png')


        

        


