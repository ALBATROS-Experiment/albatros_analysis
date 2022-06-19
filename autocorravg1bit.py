import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import glob
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
# from correlations import baseband_data_classes as bdc2

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

def get_rows_from_specnum(spec1,spec2, spec_num, spectra_per_packet):

    # print(f"received spec1 = {spec1} and spec2 = {spec2}")
    l = np.searchsorted(spec_num,spec1,side='left')
    r = np.searchsorted(spec_num,spec2,side='left')
    # print(f"spec1={spec1} spec2={spec2}")
    # print(f"spec_num len {spec_num.shape[0]}, last spec_num {spec_num[-1]}, l={l}, r={r}")
    diff1=-1
    diff2=-1
    if(l<spec_num.shape[0]):
        diff1 = spec1-spec_num[l]
    if(r<spec_num.shape[0]):
        diff2 = spec2-spec_num[r] # will give distance from right element in not equal cases

    if(diff1!=0 and diff2!=0):
        diff1 = spec1-spec_num[l-1]
        diff2 = spec2-spec_num[r-1]
        if(diff1<spectra_per_packet):
            idx1=(l-1)*spectra_per_packet+diff1
        else:
            idx1=l*spectra_per_packet
        if(diff2<spectra_per_packet):
            idx2=(r-1)*spectra_per_packet+diff2
        else:
            idx2=r*spectra_per_packet
    elif(diff1==0 and diff2!=0):
        idx1 = l*spectra_per_packet
        diff2 = spec2-spec_num[r-1]
        if(diff2<spectra_per_packet):
            idx2=(r-1)*spectra_per_packet+diff2
        else:
            idx2=r*spectra_per_packet
    elif(diff1!=0 and diff2==0):
        idx2 = r*spectra_per_packet
        diff1 = spec1-spec_num[l-1]
        if(diff1<spectra_per_packet):
            idx1=(l-1)*spectra_per_packet+diff1
        else:
            idx1=l*spectra_per_packet
    else:
        idx1 = l*spectra_per_packet
        idx2 = r*spectra_per_packet
    return int(idx1),int(idx2)

def get_avg_fast_1bit(path, init_timestamp, acclen, nchunks):
    
    idxstart, fileidx, files = get_init_info(path, init_timestamp)
    print("Starting at: ",idxstart, "in filenum: ",fileidx)
    print(files[fileidx])
    
    obj = bdc.BasebandPacked(files[fileidx])
    channels=obj.channels
    assert(obj.pol0.shape[0]==obj.pol1.shape[0])
    assert(obj.bit_mode==1)
    nchan=obj.length_channels

    objlen=obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet

    pol01=np.zeros((nchunks,nchan),dtype='complex64',order='c')

    fc=0 #file counter
    st=time.time()
    file_spec_gap=0
    for i in range(nchunks):
        rem=acclen-file_spec_gap #file_spec_gap will be non-zero if a chunk ended at the end of one file.
        # we only need (acclen-file_spec_gap) spectra from new file.
        missing_spec_gap=0
        # print("MISSING:", obj.missing_loc, obj.missing_num)
        while(True):
            l=objlen-idxstart
            if(l<rem):
                # print("less than rem:", "idxstart l objlen", idxstart, l, objlen)
                rowstart, rowend = get_rows_from_specnum(idxstart,idxstart+objlen,obj.spec_idx,obj.spectra_per_packet)
                missing_spec_gap += get_num_missing(idxstart,idxstart+objlen,obj.missing_loc,obj.missing_num)
                pol01[i,:]=pol01[i,:] + cr.avg_xcorr_1bit(obj.pol0[rowstart:rowend,:], obj.pol1[rowstart:rowend,:],obj.length_channels)

                #if the code is here another part of chunk will be read from next file. 
                # So it WILL go to the else block, and that's where we'll divide. Just adding here.

                fc+=1
                idxstart=0
                file_spec_gap = -(obj.spec_num[-1]+obj.spectra_per_packet) # file_spec_gap = first spec num of new file - (last specnum + spec_per_pack of old file)
                # del obj

                obj = bdc.BasebandPacked(files[fileidx+fc])

                file_spec_gap += obj.spec_num[0]
                objlen= obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet
                rem = rem-l-file_spec_gap
            elif(l==rem):
                # one chunk ends exactly at end of file
                missing_spec_gap += get_num_missing(idxstart,idxstart+rem,obj.missing_loc,obj.missing_num)
                rowstart, rowend = get_rows_from_specnum(idxstart,idxstart+rem,obj.spec_idx,obj.spectra_per_packet)
                pol01[i,:]=(pol01[i,:] + cr.avg_xcorr_1bit(obj.pol0[rowstart:rowend,:],obj.pol1[rowstart:rowend,:],obj.length_channels))/(acclen-file_spec_gap-missing_spec_gap)
                #file_spec_gap above shouldn't affect the avg since we're still in the same file. And it doesn't because it's zero until a new file is read.
                fc+=1
                idxstart=0
                file_spec_gap = -(obj.spec_num[-1]+obj.spectra_per_packet)
                obj = bdc.BasebandPacked(files[fileidx+fc])
                file_spec_gap += obj.spec_num[0]
                objlen= obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet
                #don't reset file_spec_gap because upcoming chunk will be read from new file.
                break
            else:
                # print(obj.missing_loc,obj.missing_num,obj.spec_idx)
                missing_spec_gap += get_num_missing(idxstart,idxstart+rem,obj.missing_loc,obj.missing_num)
                rowstart, rowend = get_rows_from_specnum(idxstart,idxstart+rem,obj.spec_idx,obj.spectra_per_packet)
                if(rowstart==rowend):
                    print("YO SOME INVALID SHIT")
                    pol01[i,:] = np.nan  # if the whole specnum block lies in missing area. will be masked later
                else:
                    pol01[i,:]=(pol01[i,:] + cr.avg_xcorr_1bit(obj.pol0[rowstart:rowend,:],obj.pol1[rowstart:rowend,:],obj.length_channels))/(acclen-file_spec_gap-missing_spec_gap)
                idxstart+=rem
                file_spec_gap=0 # reset this because we're continuing in same file for upcoming chunks
                break
        print(i+1," blocks read")
    et=time.time()
    print(f"time taken {et-st:4.2f}")
    pol01 = np.ma.masked_invalid(pol01)
    return pol01,channels


if __name__=="__main__":

    init_path = "/project/s/sievers/albatros/uapishka/baseband/snap1/16275/16275*.raw"
    # /project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627204039.raw
    init_t = 1627528549 #1627204039
    acclen = 20000
    nchunks = 10986
    pol01,channels = get_avg_fast_1bit(init_path,init_t,acclen,nchunks)
    print("RUN 1 DONE")

    # pol00_2,pol11_2,pol01_2,channels = get_avg_fast(init_path,init_t,acclen,nchunks)
    # print("RUN 2 DONE")
    # diff1=np.sum(np.abs(pol00_1-pol00_2),axis=1)
    # diff2=np.sum(np.abs(pol00_1-pol00_2),axis=1)
    # diff3=np.sum(np.abs(pol00_1-pol00_2),axis=1)
    # print(diff1,diff2,diff3)

    np.savez_compressed(f'/scratch/s/sievers/mohanagr/pol01_1bit_{str(init_t)}_{str(acclen)}_{str(nchunks)}.txt',data=pol01.data,mask=pol01.mask)
    # r = np.real(pol01_1)
    # im = np.imag(pol01_1)

    # from matplotlib import pyplot as plt
    # fig,ax=plt.subplots(1,2)
    # fig.set_size_inches(10,4)
    # # img1=ax[0].imshow(np.abs(pol01_1),aspect='auto',vmax=0.01)
    # # img2=ax[1].imshow(np.angle(pol01_1),aspect='auto',vmin=-np.pi,vmax=np.pi)
    # img1=ax[0].imshow(r,aspect='auto',vmin=-0.005,vmax=0.005)
    # ax[0].set_title('real part')
    # img2=ax[1].imshow(im,aspect='auto',vmin=-0.005,vmax=0.005)
    # ax[1].set_title('imag part')
    # plt.colorbar(img1,ax=ax[0])
    # plt.colorbar(img2,ax=ax[1])
    # plt.savefig(f'/scratch/s/sievers/mohanagr/pol01_1bit_{str(init_t)}_{str(acclen)}_{str(nchunks)}.png')
    # print(f'/scratch/s/sievers/mohanagr/pol01_1bit_{str(init_t)}_{str(acclen)}_{str(nchunks)}.png')




        

                
                
            
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    