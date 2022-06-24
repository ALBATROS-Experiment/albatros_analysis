import numpy as np
# from correlations_temp import baseband_data_classes as bdc
import glob
import time
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
from utils import baseband_utils as butils
import argparse


def get_avg_fast_1bit(path, init_t, end_t, acclen, nchunks, chanstart=0, chanend=-1):
    
    idxstart, fileidx, files = butils.get_init_info(init_t, end_t, path)
    print("Starting at: ",idxstart, "in filenum: ",fileidx)
    print(files[fileidx])
    
    obj = bdc.BasebandPacked(files[fileidx],chanstart,chanend)
    channels=obj.channels[chanstart:chanend]
    assert(obj.pol0.shape[0]==obj.pol1.shape[0])
    assert(obj.bit_mode==1)

    objlen=obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet

    if(chanend==-1):
        ncols=obj.length_channels
    else:
        ncols = chanend-chanstart

    pol01=np.zeros((nchunks,ncols),dtype='complex64',order='c')

    fc=0 #file counter
    st=time.time()
    file_spec_gap=0
    for i in range(nchunks):
        if(file_spec_gap>=acclen):
            # next chunk is not present. missing spec data.
            print("MASSIVE GAP BETWEEN TWO FILES. NEW CHUNK IN GAP.")
            pol01[i,:] = np.nan
            continue
        else:
            rem=acclen-file_spec_gap #file_spec_gap will be non-zero if a chunk ended at the end of one file.
            # we only need (acclen-file_spec_gap) spectra from new file.
        missing_spec_gap=0
        while(True):
            l=objlen-idxstart
            if(l<rem):
                # print("less than rem:", "idxstart l objlen", idxstart, l, objlen)
                rowstart, rowend = butils.get_rows_from_specnum(idxstart,idxstart+objlen,obj.spec_idx,obj.spectra_per_packet)
                missing_spec_gap += butils.get_num_missing(idxstart,idxstart+objlen,obj.missing_loc,obj.missing_num)
                pol01[i,:]=pol01[i,:] + cr.avg_xcorr_1bit(obj.pol0[rowstart:rowend,:], obj.pol1[rowstart:rowend,:],ncols)

                #if the code is here another part of chunk will be read from next file. 
                # So it WILL go to the else block, and that's where we'll divide. Just adding here.

                fc+=1
                idxstart=0
                file_spec_gap = -(obj.spec_num[-1]+obj.spectra_per_packet) # file_spec_gap = first spec num of new file - (last specnum + spec_per_pack of old file)
                # del obj
                obj = bdc.BasebandPacked(files[fileidx+fc],chanstart,chanend)

                file_spec_gap += obj.spec_num[0]
                file_spec_gap = int(file_spec_gap)
                print("FILE SPEC GAP IS ", file_spec_gap)
                if(file_spec_gap>0):
                    print("WARNING: SPEC GAP NOTICED BETWEEN FILES")
                objlen= obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet
                rem = rem-l #new remaining % of chunk left to read
                if(file_spec_gap>=rem):
                    print("MASSIVE GAP BETWEEN TWO FILES")
                    #if the spec gap b/w two files is bigger than what we had to read, the small part read earlier is the whole chunk
                    pol01[i,:]=pol01[i,:]/(l-missing_spec_gap)
                    file_spec_gap = file_spec_gap - rem # for the next chunk that'll be read from new file read above
                    break
                else:
                    rem = rem - file_spec_gap # continue the while loop and go to else block

            elif(l==rem):
                # one chunk ends exactly at end of file
                missing_spec_gap += butils.get_num_missing(idxstart,idxstart+rem,obj.missing_loc,obj.missing_num)
                rowstart, rowend = butils.get_rows_from_specnum(idxstart,idxstart+rem,obj.spec_idx,obj.spectra_per_packet)
                pol01[i,:]=(pol01[i,:] + cr.avg_xcorr_1bit(obj.pol0[rowstart:rowend,:],obj.pol1[rowstart:rowend,:],ncols))/(acclen-file_spec_gap-missing_spec_gap)
                #file_spec_gap above shouldn't affect the avg since we're still in the same file. And it doesn't because it's zero until a new file is read.
                fc+=1
                idxstart=0
                file_spec_gap = -(obj.spec_num[-1]+obj.spectra_per_packet)
                obj = bdc.BasebandPacked(files[fileidx+fc],chanstart,chanend)
                file_spec_gap += obj.spec_num[0]
                objlen= obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet
                #don't reset file_spec_gap because upcoming chunk will be read from new file.
                break
            else:
                # print(obj.missing_loc,obj.missing_num,obj.spec_idx)
                missing_spec_gap += butils.get_num_missing(idxstart,idxstart+rem,obj.missing_loc,obj.missing_num)
                rowstart, rowend = butils.get_rows_from_specnum(idxstart,idxstart+rem,obj.spec_idx,obj.spectra_per_packet)
                if(rowstart==rowend):
                    print("WHOLE CHUNK LIES IN MISSING REGION")
                    pol01[i,:] = np.nan  # if the whole specnum block lies in missing area. will be masked later
                else:
                    pol01[i,:]=(pol01[i,:] + cr.avg_xcorr_1bit(obj.pol0[rowstart:rowend,:],obj.pol1[rowstart:rowend,:],ncols))/(acclen-file_spec_gap-missing_spec_gap)
                idxstart+=rem
                file_spec_gap=0 # reset this because we're continuing in same file for upcoming chunks
                break
        print(i+1," blocks read")
    et=time.time()
    print(f"time taken {et-st:4.2f}")
    pol01 = np.ma.masked_invalid(pol01)
    return pol01,channels


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Average over pol01 of 1 bit baseband files with custom control over acclen/timestamps and channels.")
    parser.add_argument('data_dir', type=str,help='Parent data directory. Should have 5 digit time folders.')
    parser.add_argument("time_start",type=int, help="Start timestamp ctime")
    parser.add_argument("acclen", type=int, help="Accumulation length for averaging")
    parser.add_argument('-n', '--nchunks', dest='nchunks',type=int, default=560, help='Number of chunks in output file. If stop time is specfied this is overwritten. Default 560 ~ 1 hr.')
    parser.add_argument('-t', '--time_stop', dest='time_stop',type=int, default=False, help='Stop time. Overwrites nchunks if specified')
    parser.add_argument("-c", '--chans', type=int, nargs=2, help="Indices of start and end channels. Start channel index MUST be even.")
    parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='/scratch/s/sievers/mohanagr/',
              help='Output directory for data and plots')
    args = parser.parse_args()

    if(args.time_stop):
        args.nchunks = int(np.floor((args.time_stop-args.time_start)*250e6/4096/args.acclen))
    else:
        args.time_stop = args.time_start + int(np.ceil(args.nchunks*args.acclen*4096/250e6))
    if(not args.chans):
        args.chans=[0,-1]

    print("nchunks is: ", args.nchunks,"and stop time is ", args.time_stop)
    # assert(1==0)
    pol01,channels = get_avg_fast_1bit(args.data_dir, args.time_start, args.time_stop, args.acclen, args.nchunks, args.chans[0], args.chans[1])
    print("RUN 1 DONE")

    # pol01_2,channels = get_avg_fast_1bit(args.data_dir, args.time_start, args.time_stop, args.acclen, args.nchunks)
    # print("RUN 2 DONE")

    # diff1=np.sum(np.abs(pol01_1-pol01_2),axis=1)
    # print(diff1) checked that this is zero. 

    import os
    fname = f"pol01_1bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.npz"
    fpath = os.path.join(args.outdir,fname)
    np.savez_compressed(fpath,data=pol01.data,mask=pol01.mask,chans=channels)
    r = np.real(pol01)
    im = np.imag(pol01)

    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(10,4)
    # img1=ax[0].imshow(np.abs(pol01_1),aspect='auto',vmax=0.01)
    # img2=ax[1].imshow(np.angle(pol01_1),aspect='auto',vmin=-np.pi,vmax=np.pi)
    img1=ax[0].imshow(r,aspect='auto',vmin=-0.005,vmax=0.005)
    ax[0].set_title('real part')
    img2=ax[1].imshow(im,aspect='auto',vmin=-0.005,vmax=0.005)
    ax[1].set_title('imag part')
    plt.colorbar(img1,ax=ax[0])
    plt.colorbar(img2,ax=ax[1])
    fname=f'pol01_1bit_{str(args.time_start)}_{str(args.acclen)}_{str(args.nchunks)}_{args.chans[0]}_{args.chans[1]}.png'
    fpath=os.path.join(args.outdir,fname)
    plt.savefig(fpath)
    print(fpath)




        

                
                
            
                
                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    