import time, datetime, os, re
import numpy as np
from scio import scio


def read_field_many_fast(dirs,tag,dtype='float64',return_missing=False):
    """Reads one baseline from multiple data dumps. 
    
    *Warning* Do both this function and read_pol_fast have different use 
    cases? The only difference I can see is that this one also has option 
    to return missing files, and is implemented slightly differently 
    (does not use scio [pbio]). 
    
    Parameters
    ----------
    dirs: list
        List of paths to baseline data files. 
    tag: str
        Identifies which polarization baseline. E.g. 'pol00'
    dtype: str
        Specifies numpy primitive data-type. 
    return_missing: bool
        If True, returns list of paths where data is missing as second 
        return parameter. Defaults to False. 
    
    
    Returns
    -------
    np.ndarray or tuple[np.ndarray, list]
        A numpy array containing all xcorr/autocorr baseline data 
        requested. If none was found, returns None. If return_missing 
        is True, returns a list of filepaths that point to non-existant
        data. 
    """
    ndir=len(dirs)
    all_dat=[None]*ndir
    missing=[]
    ndat=0
    for i in range(ndir):
        try:
            fname=dirs[i]+'/'+tag # todo: change to os.path.join(dirs[i], tag)
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
    

def read_pol_fast(dirs,tag):
    """Reads one baseline from multiple data dumps.
    
    A baseline is identified by tag, the data is identified by dirs. 
    Relevant data are read and returned as a 2d numpy array. 
    
    Parameters
    ----------
    dirs: list of str
        List of paths to baseline data files. 
    tag: str
        Identifies which polarization baseline. E.g. 'pol00'
        
    Returns
    -------
    big_dat: np.ndarray with shape (ndat, nchan)
        A 2-d numpy array that stores all relevant data. 
    """
    ndir=len(dirs)
    fnames=[None]*ndir
    for i in range(ndir):
        fnames[i]=dirs[i]+'/'+tag 
    # Once we have tests, above 4 lines can be condensed into: 
    # fnames = [os.path.join(d,tag) for d in dirs]
    t0=time.time()
    all_dat=scio.read_files(fnames) # all data files
    t1=time.time()
    print('read files in ',t1-t0)
    ndat=0
    for dat in all_dat:
        if not(dat is None): # if dat is not None
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
        print('No files found in read_pol_fast.')
        big_dat=None
    return big_dat
    

def ctime2timestamp(ctimes):
    """Given a (list of) ctime, convert to human friendly format.

    Parameters
    ----------
    ctime: int or float or list of int/float
        (List of) ctime(s).

    Returns 
    -------
    list
        The time stamps (or list of time stamps) in human friendly format.
    """

    if isinstance(ctimes, (int, float)):
        return str(datetime.datetime.utcfromtimestamp(ctimes))
    else:
        return [ str(datetime.datetime.utcfromtimestamp(c)) for c in ctimes ]


def timestamp2ctime(date_strings, time_format='%Y%m%d_%H%M%S'):
    """Converts list of date strings into ctime format.
    
    Given a string time stamp (or list of time stamps) in human-frendly
    format, with default being YYYYMMSS_HHMMSS, convert to datetime
    object and calculate ctime.

    Parameters
    ----------
    date_strings: str or list of str
        Time stamp(s) in desired text format
    time_format: str or list of str
        Formatting string for datetime

    Returns
    -------
    list 
        The time stamps (or list of time stamps) in ctime.
     """

    t0 = datetime.datetime(1970, 1, 1)

    if isinstance(date_strings, str):
        return int((datetime.datetime.strptime(date_strings, time_format) - t0).total_seconds())
    else:
        return [ int((datetime.datetime.strptime(d, time_format) - t0).total_seconds()) for d in date_strings ]


def time2fnames(time_start, time_stop, dir_parent, fraglen=5):
    """Gets a list of filenames within specified time-rage. 
    
    Given a start and stop ctime, retrieve list of corresponding files.
    This function assumes that the parent directory has the directory
    structure <dir_parent>/<5-digit coarse time fragment>/<10-digit
    fine time stamp>.

    Paramaters
    -----------
    time_start: int 
        start time in ctime 
    time_stop: int 
        stop time in ctime
    dir_parent: str
        parent directory, e.g. /path/to/data_100MHz
    fraglen: int 
        number of digits in coarse time fragments
    
    Returns 
    -------
    list of str
        List of files in specified time range.
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
            if ((int(time_coarse) < int(str(time_start)[:fraglen])-1) or (int(time_coarse) > int(str(time_stop)[:fraglen])+1)):
                continue 
            
            all_fnames = os.listdir('{}/{}'.format(dir_parent, time_coarse))
            all_fnames.sort()

            for f in all_fnames:
                if s.search(f):
                    tstamp = int(s.search(f).groups()[0])
                    if tstamp >= time_start and tstamp <= time_stop:
                        # fnames.append(dir_parent+'/'+time_coarse+'/'+f)
                        fnames.append(os.path.join(dir_parent,time_coarse,f))
        except:
            pass
    fnames.sort()
    return fnames


  
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

    for i in range(len(pols)):
        for j in range(i+1, len(pols)):
            for reality in ['r', 'i']:
                tag = 'pol{}{}{}.scio'.format(i,j,reality)
                poldata = read_pol_fast(fnames, tag)
                data.append(poldata)

    data = np.asarray(data)
    data = data[:,inds]

    return time, data



def callocdir(dir_name):
    """Allocate and initialize a directory. (think calloc/malloc in C)
    
    Make sure a directory specified exists and is empty. If it doesn't
    exist, create it; if it's not empty, empty it.
    
    Parameters
    ----------
    dir_name: str
        The path to the directory we want to allocate. 
    """
    if os.path.exists(dir_name) == False: # == False -> is False
        os.mkdir(dir_name)
    else: #empty it before writing into it (who knows wtf is in it)
        for  file_name in os.listdir(dir_name):
            temp_path = os.path.join(dir_name, file_name)
            try:
                os.unlink(temp_path)
            except Exception as error:
                print("failed to delete: " + str(temp_path) + " cause: " + str(error))
                return
    return

def mallocdir(dir_name):
    """Allocate a directory without initializing it. (think calloc/malloc)
    
    If a directory doesn't exist, create it; otherwise, leave it as it is. 
    
    Parameters
    ----------
    dir_name: str
        The path to the directory we want to allocate. 
    """
    if os.path.exists(dir_name) == False:
        os.mkdir(dir_name)
    return

def readin_computed(fname):
    """Read binary file into numpy array. 
    
    *Warning*, this may be depricated (with 
    functionality contained within `read_pol_fast` subroutine)
    
    Thin wrapper for np.load().
    
    Parameters
    ----------
    fname: str
        Path to binary file. 

    Returns
    -------
    np.ndarray
        The binary array at fname. 
    """
    with open(fname, 'rb') as f:
        out = np.load(f)
    return out


def readin_append(dir_names, base_file_path, file_name, function):
    """Read multiple files fast. 
    
    *Warning* This may be a duplicate of `read_field_many_fast`. 
    
    Looks through multiple specified sub-directories of base_file_path 
    for leaves (files) with one specific name. Applies function to each
    file. E.g. This method can be used to load multiple data from one 
    multiple files containing the same baseline (xcorr/autocorr) into 
    an array.
    
    Parameters
    ----------
    dir_names: list of str
        List of relative directory names. 
    base_file_path: str
        Path to the base file in which we search for sub-directories 
        'dir_names'. 
    file_name: str
        Name of the leaf node (file). A similar variable is refered to as 
        'tag' above. 
    function: callable
        Applied to filepath. Takes path-string and returns numpy array. 
    
    Returns
    -------
    data: np.ndarray
        Most likely this will be used to read 
    """
    for index, dir_name in enumerate(dir_names):
        file_path = os.path.join(base_file_path, dir_name, file_name)
        if index ==0:
            data = function(file_path)
        else:
            data = np.append(data,function(file_path), axis = 0)
            # print("append so shape is now")
            # print(np.shape(data))
    # print(np.shape(data))
    return data 
