import glob, time, datetime, os, re
import numpy as np
import scio

#============================================================
def read_field_many_fast(dirs,tag,dtype='float64',return_missing=False):
    ndir=len(dirs)
    all_dat=[None]*ndir
    missing=[]
    ndat=0
    for i in range(ndir):
        try:
            fname=dirs[i]+'/'+tag
            all_dat[i]=np.fromfile(fname,dtype=dtype)
            ndat=ndat+len(all_dat[i])
        except:
            missing.append(fname)
    if ndat>0:
        dat=np.zeros(ndat,dtype=dtype)
        ii=0
        for i in range(ndir):
            if not(all_dat[i] is None):
                nn=len(all_dat[i])
                if nn>0:
                    dat[ii:ii+nn]=all_dat[i]
                    ii+=nn
        if return_missing:
            return dat,missing
        else:
            return dat
    else:
        if return_missing:
            return None,missing
        else:
            return None
    
#============================================================
def read_pol_fast(dirs,tag):
    ndir=len(dirs)
    fnames=[None]*ndir
    for i in range(ndir):
        fnames[i]=dirs[i]+'/'+tag
    t0=time.time()
    all_dat=scio.read_files(fnames)
    t1=time.time()
    print 'read files in ',t1-t0
    ndat=0
    for dat in all_dat:
        if not(dat is None):
            ndat=ndat+dat.shape[0]
            nchan=dat.shape[1]

    if ndat>0:
        big_dat=np.zeros([ndat,nchan])
        ii=0
        for dat in all_dat:
            if not(dat is None):
                nn=dat.shape[0]
                big_dat[ii:(ii+nn),:]=dat
                ii=ii+nn
    else:
        print 'no files found in read_pol_fast.'
        big_dat=None
    return big_dat
    
#============================================================
def ctime2timestamp(ctimes):
    """Given a (list of) ctime, convert to human friendly format.

    - ctime = ctime(s) in desired text format

    Returns the time stamps (or list of time stamps) in human friendly format.
    """

    if isinstance(ctimes, (int, float)):
        return str(datetime.datetime.utcfromtimestamp(ctimes))
    else:
        return [ str(datetime.datetime.utcfromtimestamp(c)) for c in ctimes ]

#============================================================
def timestamp2ctime(date_strings, time_format='%Y%m%d_%H%M%S'):
    """Given a string time stamp (or list of time stamps) in human-frendly
    format, with default being YYYYMMSS_HHMMSS, convert to datetime
    object and calculate ctime.

    - date_strings = time stamp(s) in desired text format
    - time_format = formatting string for datetime

    Returns the time stamps (or list of time stamps) in ctime.
     """

    t0 = datetime.datetime(1970, 1, 1)

    if isinstance(date_strings, basestring):
        return int((datetime.datetime.strptime(date_strings, time_format) - t0).total_seconds())
    else:
        return [ int((datetime.datetime.strptime(d, time_format) - t0).total_seconds()) for d in date_strings ]

#============================================================
def time2fnames(time_start, time_stop, dir_parent, fraglen=5):
    """Given a start and stop ctime, retrieve list of corresponding files.
    This function assumes that the parent directory has the directory
    structure <dir_parent>/<5-digit coarse time fragment>/<10-digit
    fine time stamp>.

    - time_start, time_stop = start/stop times in ctime
    - dir_parent = parent directory, e.g. /path/to/data_100MHz
    - fraglen = # digits in coarse time fragments

    Returns list of files in specified time range.
    """

    times_coarse = os.listdir(dir_parent)
    times_coarse.sort()
    s = re.compile(r'(\d{10})')  # We'll use this to search for 10-digit time strings

    fnames = []
    for time_coarse in times_coarse:
        try:
            # Include +-1 coarse directory on endpoints because
            # sometimes the fine time stamp rolls over to the coarse
            # time within the same directory
	    if int(time_coarse) < int(str(time_start)[:fraglen])-1 or int(time_coarse) > int(str(time_stop)[:fraglen])+1:
       	        continue
	    all_fnames = os.listdir('{}/{}'.format(dir_parent, time_coarse))
       	    all_fnames.sort()

            for f in all_fnames:
	        if s.search(f):
                    tstamp = int(s.search(f).groups()[0])
                    if tstamp >= time_start and tstamp <= time_stop:
                        fnames.append(dir_parent+'/'+time_coarse+'/'+f)
        except:
            pass
    fnames.sort()
    return fnames

#============================================================    
def ctime2data(dir_parent, ct_start, ct_stop, pols = [0,1], time_file='time_gps_start.raw', fraglen=5):
    """Given a parent directory containing all SNAP data (eg. data_auto_cross), 
    and start and stop timestamp in human-friendly format (default being
    YYYYMMDD_HHMMSS), returns all the data between those times.

    - parentdir = dirctory conatining all SNAP data (string)
    - ct_start(/stop) = start(/stop) timestamps in UNIX time
    - pols = array of polarizations to read
    - time_file = name of file with time stamp data
    
    Returns array of 2d arrays, arranged by polarization:
    auto, ..., cross_r, cross_i, ...
    """

    fnames = time2fnames(ct_start, ct_stop, dir_parent, fraglen=fraglen)

    time = read_field_many_fast(fnames, time_file)

    inds = np.where( (time >= ct_start) & (time <= ct_stop) )[0]
    time = time[inds]
    
    print("Requested start time was: "+str(ct_start))
    print("Requested stop time was: "+str(ct_stop))
    print("Actual start time is: "+str(time[0]))
    print("Actual stop time is: "+str(time[-1]))
        
    data = []
    for pol in pols:
        tag = 'pol{0}{0}.scio'.format(pol)
        poldata = read_pol_fast(fnames, tag)
        data.append(poldata)

    for i in xrange(len(pols)):
        for j in xrange(i+1, len(pols)):
            for reality in ['r', 'i']:
                tag = 'pol{}{}{}.scio'.format(i,j,reality)
                poldata = read_pol_fast(fnames, tag)
                data.append(poldata)

    data = np.asarray(data)
    data = data[:,inds]

    return time, data
