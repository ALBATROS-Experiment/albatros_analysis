import numpy
import os
import bz2
import gzip
import multiprocessing
import time
    

class scio:
    def __init__(self,fname,arr=None,status='w',compress=None,diff=False):
        if not(compress is None):
            if len(compress)==0:
                compress=None
        self.fid=open(fname,status)
        self.fname=fname
        self.diff=diff
        self.last=None
        self.compress=compress
        self.closed=False

        if arr is None:
            self.dtype=None
            self.shape=None
            self.initialized=False
        else:
            self.dtype=arr.dtype
            self.shape=arr.shape
            self.initialized=True
            self.write_header(arr)
            self.append(arr)

            
    def __del__(self):
        if self.closed==False:
            print('closing scio file ' + self.fname)
            self.fid.flush()        
            self.fid.close()
            self.closed=True
            if not(self.compress is None):
                to_exec=self.compress + ' ' + self.fname
                os.system(to_exec)


    def close(self):
        self.__del__()
    def write_header(self,arr):
        sz=arr.shape
        myvec=numpy.zeros(len(sz)+2,dtype='int32')
        myvec[0]=len(sz)
        if self.diff:
            myvec[0]=-1*myvec[0]
        for i in range(len(sz)):
            myvec[i+1]=sz[i]
        myvec[-1]=dtype2int(arr)
        myvec.tofile(self.fid)

        
    def append(self,arr):
        if self.initialized==False:
            self.dtype=arr.dtype
            self.shape=arr.shape
            self.write_header(arr)
            self.initialized=True

        if (arr.shape==self.shape):
            pass
        else:
            print("shape mismatch in scio.append")
        if (arr.dtype==self.dtype):
            if (self.diff):
                if self.last is None:
                    arr_use=arr
                else:
                    arr_use=arr-self.last
                self.last=arr.copy()
            else:
                arr_use=arr
            arr_use.tofile(self.fid)
            self.fid.flush()
        else:
            print('dtype mismatch in scio.append on file ' + self.fname)
        
            
#def append(arr,fname,overwrite=False):
#    asdf='abc'
#    assert(type(fname)==type(asdf))
#    asdf=numpy.zeros(2)
#    assert(type(arr)==type(asdf))
#    if overwrite:
#        os.system('rm  ' + fname)
#        
#    if (os.path.isfile(fname)):
#        f=open(fname,'a')
#        arr.tofile(f)
#        f.close()
#    else:
#        print 'creating ' + fname
#        f=open(fname,'w')
#        sz=arr.shape
#        myvec=numpy.zeros(len(sz)+2,dtype='int32')
#        myvec[0]=len(sz)
#        for i in range(len(sz)):
#            myvec[i+1]=sz[i]
#        myvec[-1]=dtype2int(arr)
#        #print myvec
#        #print sz
#        #print type(myvec)
#        myvec.tofile(f)
#        arr.tofile(f)
#        f.close()

def _read_from_string(mystr):
    icur=0;
    ndim=numpy.fromstring(mystr[icur:icur+4],dtype='int32')[0]
    icur=icur+4
    if (ndim<0):
        diff=True
        ndim=-1*ndim
    else:
        diff=False        
    #print 'ndim is ',ndim
    sz=numpy.fromstring(mystr[icur:icur+4*ndim],'int32')
    icur=icur+4*ndim
    mytype=numpy.fromstring(mystr[icur:icur+4],'int32')[0]
    icur=icur+4

    #check for file size sanity
    bytes_per_frame=int2nbyte(mytype)*numpy.product(sz)
    cur_bytes=len(mystr)-icur
    n_to_cut=numpy.remainder(cur_bytes,bytes_per_frame)
    if n_to_cut>0:
        #print 'current len: ',len(mystr)
        print('We have a byte mismatch in reading scio file.  Truncating ' + repr(n_to_cut) + ' bytes.')
        mystr=mystr[:-n_to_cut]
        #print 'new len: ',len(mystr)
        
    vec=numpy.fromstring(mystr[icur:],dtype=int2dtype(mytype))

    nmat=vec.size/numpy.product(sz)
    new_sz=numpy.zeros(sz.size+1,dtype='int32')
    new_sz[0]=nmat
    new_sz[1:]=sz

    mat=numpy.reshape(vec,new_sz)
    if diff:
        mat=numpy.cumsum(mat,0)

    return mat
    

def _read_file_as_string(fname):
    if fname[-4:]=='.bz2':
        f=bz2.BZ2File(fname,'r')
        mystr=f.read()
        f.close()
        return mystr
    if fname[-3:]=='.gz':
        f=gzip.GzipFile(fname,'r')
        mystr=f.read()
        f.close()
        return mystr

    #if we get here, assume it's raw binary
    f=open(fname,'rb')
    mystr=f.read()
    f.close()
    return mystr

def read(fname,strict=False):
    if True:
        if strict:
            #only read the filename passed in
            mystr=_read_file_as_string(fname)
            return _read_from_string(mystr)
        else:
            #try some guesses about what other sane filenames might be based on the input filename
            fnames=[fname]
            if fname[-4:]=='.bz2':
                fnames.append(fname[:-4])
            if fname[-3:]=='.gz':
                fnames.append(fname[:-3])
            fnames.append(fname+'.bz2')
            fnames.append(fname+'.gz')
            
            for fname in fnames:
                try:
                    mystr=_read_file_as_string(fname)
                    if len(mystr)>0:
                        try:  #try/except loop added by JLS 11 June 2019 to catch cases where string length is unexpected
                            return _read_from_string(mystr)
                        except:
                            print('File ',fname,' appears to be garbled when parsing string of length ',len(mystr))
                            return None
                    else:
                        return None
                except:
                    pass
    return None
    if fname[-4:]=='.bz2':
        return read_bz2(fname)
    f=open(fname)
    ndim=numpy.fromfile(f,'int32',1)
    if (ndim<0):
        diff=True
        ndim=-1*ndim
    else:
        diff=False
        
    sz=numpy.fromfile(f,'int32',ndim)
    mytype=numpy.fromfile(f,'int32',1)
    vec=numpy.fromfile(f,dtype=int2dtype(mytype))
    nmat=vec.size/numpy.product(sz)
    new_sz=numpy.zeros(sz.size+1,dtype='int32')
    new_sz[0]=nmat
    new_sz[1:]=sz


    mat=numpy.reshape(vec,new_sz)
    if diff:
        mat=numpy.cumsum(mat,0)

    return mat

def read_files(fnames,ncpu=0):
    t1=time.time()
    if ncpu==0:
        ncpu=multiprocessing.cpu_count()
    p=multiprocessing.Pool(ncpu)
    data=p.map(read,fnames)
    #without the p.terminate, the pool seems to last, which can cause the system to run out of processes.
    #this isn't what the documentation says should happen (terminate is supposed to get called when p 
    #gets garbage collected), but oh well...
    p.terminate()      
    t2=time.time()
    #print 'took ',t2-t1, ' seconds to read files in scio.'
    return data



def int2dtype(myint):
    if (myint==8):
        return 'float64'
    if (myint==4):
        return 'float32'
    if (myint==-4):
        return 'int32'
    if (myint==-8):
        return 'int64'
    if (myint==-104):
        return 'uint32'
    if (myint==-108):
        return 'uint64'
    
def int2nbyte(myint):
    nbyte=numpy.abs(myint)
    if nbyte>100:
        nbyte=nbyte-100
    return nbyte

def dtype2int(dtype_str):
    
    if (type(dtype_str)!=numpy.dtype):
        dtype_str=dtype_str.dtype

    aa=numpy.zeros(1,dtype='float64')
    if (dtype_str==aa.dtype):
        return 8

    aa=numpy.zeros(1,dtype='float32')
    if (dtype_str==aa.dtype):
        return 4
    

    aa=numpy.zeros(1,dtype='int32')
    if (dtype_str==aa.dtype):
        return -4
    
    aa=numpy.zeros(1,dtype='int64')
    if (dtype_str==aa.dtype):
        return -8

    aa=numpy.zeros(1,dtype='uint32')
    if (dtype_str==aa.dtype):
        return -104

    aa=numpy.zeros(1,dtype='uint64')
    if (dtype_str==aa.dtype):
        return -108
    
    print('unknown dtype')
    return 0