import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import datetime, time, re
from scio import scio
import SNAPfiletools as sft
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

def get_binned(fname,data_dir,coords,ctime_start,ctime_stop,nbins):
    
    # data_dir='/project/s/sievers/mohanagr/uapishka_aug_oct_2022/data_auto_cross/'
    # data_dir='/project/s/sievers/mohanagr/uapishka_franken_oct_nov_2022/data_auto_cross/SNAP3/'
    # data_dir='/project/s/sievers/albatros/marion/albatros-hydroshack/data_auto_cross/'
    data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)
    data_subdirs.sort()
    new_dirs = [d+f'/{fname}.scio.bz2' for d in data_subdirs]
    datpol00 = scio.read_files(new_dirs)
    print("Files of ",fname, "read")
    # print(data_subdirs)
    # data_subdirs = ['/home/mohan/Projects/direct/16272/1627202093']

    ts=sf.load.timescale()
    # jd = tstart/86400+2440587.5 
    # t=ts.ut1_jd(jd)
    earth_loc = sf.wgs84.latlon(*coords)
    dH = 24/nbins
    # print(earth_loc.lst_hours_at(t))
    binned = [np.asarray([]).reshape(-1,2048)]*nbins
    for i,pol00 in enumerate(datpol00):
        tstamp = get_ts_from_name(data_subdirs[i])
        if(i%10==0):
            print(i+1,"files read.File timestamp is ", tstamp)
        if(pol00 is None):
            print("YO WTF, READ",fname, data_subdirs[i])
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

    loc='uapishka'
    ctime_start= 1661011607
    ctime_stop = 1666620593
    mytz=pytz.timezone('US/Eastern')
    nbins = 720 # 2 minute bins 
    coords = [51.4641932, -68.2348603,300]
    data_dir = '/project/s/sievers/mohanagr/uapishka_aug_oct_2022/data_auto_cross/'

    plot_type = 'median'
    
    sttime=get_localtime_from_UTC(ctime_start,mytz).strftime("%b-%d %H:%M")
    entime=get_localtime_from_UTC(ctime_stop,mytz).strftime("%b-%d %H:%M")

    t1=time.time()
    binned=get_binned('pol00',data_dir,coords,ctime_start,ctime_stop,nbins)
    t2=time.time()
    print("time taken for binning",t2-t1)
    t1=time.time()
    statsp00=reduce_binned(binned,nbins,2048)
    t2=time.time()
    print("time taken for reduction",t2-t1)

    binned=get_binned('pol11',data_dir,coords,ctime_start,ctime_stop,nbins)
    statsp11=reduce_binned(binned,nbins,2048)
    print("Done pol11")
    binned=get_binned('pol01r',data_dir,coords,ctime_start,ctime_stop,nbins)
    statsp01r=reduce_binned(binned,nbins,2048)
    print("Done pol01r")
    binned=get_binned('pol01i',data_dir,coords,ctime_start,ctime_stop,nbins)
    statsp01i=reduce_binned(binned,nbins,2048)
    print("Done pol01i")
    pol01 =statsp01r[plot_type] + 1J*statsp01i[plot_type]

    f=plt.gcf()
    f.set_size_inches(15,15)
    
    plt.suptitle(f'Plotting from: {sttime} to {entime}, plot type: {plot_type}')
    myext=[0, 125, 24, 0]
    plt.subplot(321)
    plt.title("Pol00")
    plt.imshow(np.log10(statsp00[plot_type]),vmin=7,vmax=8.2,extent=myext,aspect='auto')
    plt.colorbar()

    plt.subplot(323)
    plt.title("Pol11")
    plt.imshow(np.log10(statsp11[plot_type]),vmin=7,vmax=8.2,extent=myext,aspect='auto')
    plt.colorbar()
    
    plt.subplot(322)
    plt.title("Pol01 mag")
    plt.imshow(np.log10(np.abs(pol01)),vmin=3,vmax=8,extent=myext,aspect='auto')
    plt.colorbar()

    plt.subplot(324)
    plt.title("Pol01 phase")
    plt.imshow(np.angle(pol01),extent=myext,aspect='auto',cmap='RdBu')
    plt.colorbar()

    plt.subplot(313)
    plt.title('bin count')
    plt.plot(np.arange(0,nbins),statsp00['counts'])

    output_path = f'/project/s/sievers/mohanagr/lst_{nbins}_{plot_type}_{ctime_start}_{ctime_stop}_{loc}.png'
    plt.savefig(output_path)
    print(output_path)
    
    output_path = f'/project/s/sievers/mohanagr/lst_{nbins}_{ctime_start}_{ctime_stop}_{loc}.npz'
    np.savez_compressed(output_path,\
                        p00mean=statsp00['mean'],p00median=statsp00['median'],\
                        p11mean=statsp11['mean'],p11median=statsp11['median'],\
                        p01rmean=statsp01r['mean'],p01rmedian=statsp01r['median'],\
                        p01imean=statsp01i['mean'],p01imedian=statsp01i['median'],
                        counts=statsp00['counts'])
    


        

        


