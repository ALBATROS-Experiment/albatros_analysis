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

def reduce_binned(binned,nbins,nchan):
    counts=np.zeros(nbins)
    bmedian = np.zeros((nbins,nchan)) #bmedian = bin median
    bmean = np.zeros((nbins,nchan))
    for i in range(nbins):
        counts[i]=binned[i].shape[0]
        if(counts[i]>0):
            bmean[i,:]=np.mean(binned[i],axis=0)
            bmedian[i,:]=np.median(binned[i],axis=0)
        
    return {'counts':counts,'mean':bmean,'median':bmedian}

if __name__ == '__main__':
    
    # data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)
    data_subdirs = ['/home/mohan/Projects/direct/16272/1627202093']

    ts=sf.load.timescale()
    # jd = tstart/86400+2440587.5 
    # t=ts.ut1_jd(jd)
    earth_loc = sf.wgs84.latlon(51.4641932, -68.2348603,300)
    dH = 24/1440
    # print(earth_loc.lst_hours_at(t))
    binned = [np.asarray([]).reshape(-1,2048)]*1440
    for mydir in data_subdirs:
        tstamp = get_ts_from_name(mydir)
        print("File timestamp is ", tstamp)
        pol00 = scio.read(os.path.join(mydir, "pol00.scio.bz2"))
        mytstamps = tstamp+np.arange(0,pol00.shape[0])*6.44
        myjds = mytstamps/86400 + 2440587.5
        # print(ts.ut1_jd(myjds))
        lsthrs=earth_loc.lst_hours_at(ts.ut1_jd(myjds))
        print("LST hours are:",lsthrs )
        lstbins = np.round(lsthrs/dH).astype(int)
        # print(lstbins)
        branch_points=list(np.where(np.diff(lstbins)!=0)[0])
        branch_points.append(pol00.shape[0]-1)
        # print(branch_points)
        st=0
        for i in range(len(branch_points)):
            b=branch_points[i]
            en=b+1
            # print(st,en,lstbins[b])
            binned[lstbins[b]]=np.vstack([binned[lstbins[b]],pol00[st:en,:]]) #prealocated arrays would prolly speeden this up
            st=en
    
    stats=reduce_binned(binned,1440,2048)
        

        


