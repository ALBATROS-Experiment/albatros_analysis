import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import glob
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr

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

def get_num_missing(s_idx, e_idx, missing_loc, missing_num):

    sum = 0
    for i, loc in enumerate(missing_loc):

        loc_end = loc + missing_num[i]
        
        if(loc>=s_idx):
            if(loc_end<=e_idx):
                sum+=missing_num[i]
            else:
                if(loc < e_idx):
                    sum += e_idx - loc
                else:
                    break
        else:
            if(loc_end>s_idx):
                if (loc_end <= e_idx):
                    sum+= loc_end - s_idx
                else:
                    sum+= e_idx - s_idx
                    break
    return sum
                

def get_avg_fast(path, init_timestamp, acclen, nchunks, bitmode=4, polcross=False):
    
    idxstart, fileidx, files = get_init_info(path, init_timestamp)
    print("Starting at: ",idxstart, "in filenum: ",fileidx)
    print(files[fileidx])
    
    obj = bdc.BasebandPacked(files[fileidx])
    channels=obj.channels
    assert(obj.pol0.shape[0]==obj.pol1.shape[0])
    assert(obj.bit_mode==bitmode)
    nchan=obj.pol0.shape[1]
    objlen=obj.pol0.shape[0]
    psd=np.zeros((nchunks,nchan),dtype='float64')
    fc=0 #file counter
    st=time.time()

    for i in range(nchunks):
        rem=acclen #remaining
        missing_spec_gap=0
        file_spec_gap=0
        while(True):
            l=objlen-idxstart
            if(l<rem):
                missing_spec_gap += get_num_missing(idxstart,idxstart+objlen,obj.missing_loc,obj.missing_num)
                psd[i,:]=psd[i,:] + cr.avg_autocorr_4bit(obj.pol0[idxstart:idxstart+objlen,:]) 
                #if the code is here another part of chunk will be read from next file. 
                # So it WILL go to the else block, and that's where we'll divide. Just adding here.

                fc+=1
                idxstart=0
                file_spec_gap = -(obj.spec_num[-1]+obj.spectra_per_packet) # file_spec_gap = first spec num of new file - (last specnum + spec_per_pack of old file)
                del obj
                obj = bdc.BasebandPacked(files[fileidx+fc])
                file_spec_gap += obj.spec_num[0]
                objlen=obj.pol0.shape[0]
                # assert(obj.pol0.shape[0]==obj.pol1.shape[0])
                rem-=l
                
            else:
                missing_spec_gap += get_num_missing(idxstart,idxstart+rem,obj.missing_loc,obj.missing_num)
                print(f"file spec gap: {file_spec_gap}, missing spec gap: {missing_spec_gap}")
                psd[i,:]=(psd[i,:] + cr.avg_autocorr_4bit(obj.pol0[idxstart:idxstart+rem,:]))/(acclen-file_spec_gap-missing_spec_gap)
                idxstart+=rem
                break
        print(i+1," blocks read")
    et=time.time()
    print(f"time taken {et-st:4.2f}")
    return psd,channels

if __name__=="__main__":

    init_path = '/project/s/sievers/albatros/uapishka/baseband/snap1/16272/16272*'
    init_t = 1627202094
    acclen = 393216
    nchunks = 560
    psd,channels = get_avg_fast(init_path,init_t,acclen,nchunks)
    print(channels)
    print(psd)
    np.savetxt('/scratch/s/sievers/mohanagr/autospec.txt',psd)
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(1,1)
    img=ax.imshow(np.log10(psd),aspect='auto')
    plt.colorbar(img)
    plt.savefig('/scratch/s/sievers/mohanagr/psd.png')




        

                
                
            
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    