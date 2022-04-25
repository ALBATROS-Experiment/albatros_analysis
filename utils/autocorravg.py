import numpy as np
from correlations_temp import baseband_data_classes as bdc
import glob
import time
print("imported autocorravg module successfully")

def get_init_info(path, init_timestamp):
    '''
    Returns the index of file in a folder and 
    the index of the spectra in that file corresponding to init_timestamp
    '''
    
    files = glob.glob(path)
#     print(f'Found data files {len(files)} in {path}')
    files.sort()
    fs = 250e6 #Hz sampling rate
    acclen = 393216 # accumulation length 
    # in case acclen is not available, average over time diff in timestamp file and divide by dt_spec

    speclen=4096 # length of each spectra
    dt_spec = speclen/fs # time taken to read one spectra
    dt_block = acclen*dt_spec # time taken to read one chunk of direct spectra
#     print(dt_block, 'seconds')
    # find which file to read first 
    filetstamps = [int(f.split('.')[0].split('/')[-1]) for f in files]
    filetstamps.sort()
    filetstamps = np.asarray(filetstamps)

    # ------ SKIP -------#
    # make sure the sorted order of tstamps is same as of files. so that indices we'll find below correspond to correct files
    # np.unique(filetstamps - np.asarray([int(f.split('.')[0].split('/')[-1]) for f in files])) should return [0]

    # we're looking for a file that has the start timestamp closest to what we want
    fileidx = np.where(filetstamps<=init_timestamp)[0][-1]
    #assumed that our init_t will most often lie inside some file. hardly ever a file will begin with our init timestamp

    # once we have a file, we seek to required position in time
    idxstart = int((init_timestamp-filetstamps[fileidx])/dt_spec)
    # check that starting index indeed corresponds to init_timestamp
    print("CHECK",init_timestamp,idxstart*dt_spec + filetstamps[fileidx])
    print("CHECK", filetstamps[fileidx], files[fileidx])
    
    return idxstart, fileidx, files

def get_avg_slow(path, init_timestamp, acclen, nchunks, bitmode=1, polcross=False):
    '''
    Utility for averaging autospectra for over baseband files of one antenna. S
    Slow because it uses the "float" (and not packed) version of reading files. 
    Autospectra multiplication is also carried out in python making it slower.
    '''
    
    idxstart, fileidx, files = get_init_info(path, init_timestamp)
    print("Starting at: ",idxstart, "in filenum: ",fileidx)
    print(files[fileidx])
    
    obj = bdc.baseband_data_float(files[fileidx])
    channels=obj.channels
    assert(obj.pol0.shape[0]==obj.pol1.shape[0])
    assert(obj.bit_mode==bitmode)
    nchan=obj.pol0.shape[1]
    objlen=obj.pol0.shape[0]
    psd=np.zeros((nchunks,nchan),dtype='complex128')
    fc=0 #file counter
    st=time.time()
    for i in range(nchunks):
        rem=acclen #remaining
        
        while(True):
            l=objlen-idxstart
            if(l<rem):
                psd[i,:]=psd[i,:]+np.sum(obj.pol0[idxstart:objlen,:]*\
                                         np.conj(obj.pol1[idxstart:objlen,:]), axis=0)
                fc+=1
                idxstart=0
                del obj
                obj = bdc.baseband_data_float(files[fileidx+fc])
                objlen=obj.pol0.shape[0]
                assert(obj.pol0.shape[0]==obj.pol1.shape[0])
                rem-=l
                
            else:
                psd[i,:]=psd[i,:]+np.sum(obj.pol0[idxstart:idxstart+rem,:]*\
                                         np.conj(obj.pol1[idxstart:idxstart+rem,:]), axis=0)
                idxstart+=rem
                break
        print(i+1," blocks read")
    et=time.time()
    print(f"time taken {et-st:4.2f}")
    return psd,channels
                
                
            
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    