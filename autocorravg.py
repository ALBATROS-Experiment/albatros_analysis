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
                

def get_avg_fast(path, init_timestamp, acclen, nchunks, bitmode=4):
    
    idxstart, fileidx, files = get_init_info(path, init_timestamp)
    print("Starting at: ",idxstart, "in filenum: ",fileidx)
    print(files[fileidx])
    
    obj = bdc.BasebandPacked(files[fileidx])
    channels=obj.channels
    assert(obj.pol0.shape[0]==obj.pol1.shape[0])
    assert(obj.bit_mode==bitmode)
    nchan=obj.pol0.shape[1]
    objlen=obj.pol0.shape[0]
    pol00=np.zeros((nchunks,nchan),dtype='float64',order='c')
    pol11=np.zeros((nchunks,nchan),dtype='float64',order='c')
    pol01=np.zeros((nchunks,nchan),dtype='complex64',order='c')
    obj.pol0[:]=obj.pol0-obj.pol0+254
    fc=0 #file counter
    st=time.time()
    print(type(obj.pol0))
    print(obj.pol0.dtype,obj.pol0.strides)
    for i in range(nchunks):
        rem=acclen #remaining
        missing_spec_gap=0
        file_spec_gap=0
        # print("MISSING:", obj.missing_loc, obj.missing_num)
        while(True):
            l=objlen-idxstart
            if(l<rem):
                # missing_spec_gap += get_num_missing(idxstart,idxstart+objlen,obj.missing_loc,obj.missing_num)
                # print(obj.pol0, "form while lop")
                p0 = obj.pol0[idxstart:idxstart+objlen,:].copy() # try making a copy here before passing
                pol00[i,:]=pol00[i,:] + cr.avg_autocorr_4bit(p0)
                # pol11[i,:]=pol11[i,:] + cr.avg_autocorr_4bit(obj.pol1[idxstart:idxstart+objlen,:])
                # pol01[i,:]=pol01[i,:] + cr.avg_xcorr_4bit(obj.pol0[idxstart:idxstart+objlen,:], obj.pol1[idxstart:idxstart+objlen,:])

                #if the code is here another part of chunk will be read from next file. 
                # So it WILL go to the else block, and that's where we'll divide. Just adding here.

                fc+=1
                idxstart=0
                file_spec_gap = -(obj.spec_num[-1]+obj.spectra_per_packet) # file_spec_gap = first spec num of new file - (last specnum + spec_per_pack of old file)
                del obj
                obj = bdc.BasebandPacked(files[fileidx+fc])
                obj.pol0[:]=obj.pol0-obj.pol0+254
                file_spec_gap += obj.spec_num[0]
                objlen=obj.pol0.shape[0]
                # assert(obj.pol0.shape[0]==obj.pol1.shape[0])
                rem-=l
            else:
                # print("Strides for pol00", pol00.strides)
                # missing_spec_gap += get_num_missing(idxstart,idxstart+rem,obj.missing_loc,obj.missing_num)
                # print(f"file spec gap: {file_spec_gap}, missing spec gap: {missing_spec_gap}")
                # print(obj.pol0,"while loop else")
                # p0 = obj.pol0[idxstart:idxstart+rem,:] #copy here before passing
                p0 = np.ones((rem,nchan),dtype='uint8',order='c')
                # print("Strides for p0", p0.strides)
                pol00[i,:]=(pol00[i,:] + cr.avg_autocorr_4bit(p0))#/(acclen-file_spec_gap-missing_spec_gap)
                # pol11[i,:]=(pol11[i,:] + cr.avg_autocorr_4bit(obj.pol1[idxstart:idxstart+rem,:]))/(acclen-file_spec_gap-missing_spec_gap)
                # pol01[i,:]=(pol01[i,:] + cr.avg_xcorr_4bit(obj.pol0[idxstart:idxstart+rem,:],obj.pol1[idxstart:idxstart+rem,:]))/(acclen-file_spec_gap-missing_spec_gap)
                idxstart+=rem
                break
        print(i+1," blocks read")
    et=time.time()
    print(f"time taken {et-st:4.2f}")
    return pol00,pol11,pol01,channels


if __name__=="__main__":

    init_path = '/project/s/sievers/albatros/uapishka/baseband/snap1/16272/16272*'
    init_t = 1627202094
    acclen = 393216
    nchunks = 1
    pol00_1,pol11_1,pol01_1,channels = get_avg_fast(init_path,init_t,acclen,nchunks)
    print("RUN 1 DONE")
    pol00_2,pol11_2,pol01_2,channels = get_avg_fast(init_path,init_t,acclen,nchunks)
    print("RUN 2 DONE")
    diff=np.sum(np.abs(pol00_1-pol00_2),axis=1)
    check=np.where(diff!=0)
    print(check)
    print(diff[check[0]])
    print(diff)

    # np.savetxt('/scratch/s/sievers/mohanagr/pol00_5.txt',pol00)
    # np.savetxt('/scratch/s/sievers/mohanagr/pol01_5.txt',pol01)

    # from matplotlib import pyplot as plt
    # fig,ax=plt.subplots(1,2)
    # fig.set_size_inches(10,4)
    # img1=ax[0].imshow(np.log10(np.abs(pol01)),aspect='auto')
    # img2=ax[1].imshow(np.angle(pol01),aspect='auto',vmin=-np.pi,vmax=np.pi)
    # plt.colorbar(img1,ax=ax[0])
    # plt.colorbar(img2,ax=ax[1])
    # plt.savefig('/scratch/s/sievers/mohanagr/pol01.png')




        

                
                
            
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    