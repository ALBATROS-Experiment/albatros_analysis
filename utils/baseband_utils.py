import numpy as np
import os, glob

def get_init_info(init_t, end_t, parent_dir):
    '''
    Returns the index of file in a folder and 
    the index of the spectra in that file corresponding to init_timestamp
    '''
    # create a big list of files from 5 digit subdirs. we might not need all of them, but I don't want to write regex. 
    # This may be faster, and I don't care about storing a few 100 more strings than I need to.
    frag1 = str(int(init_t/100000))
    frag2 = str(int(end_t/100000))
    path = os.path.join(parent_dir,frag1)
    files = glob.glob(path+'/*')
    if(frag1!=frag2):
        path = os.path.join(parent_dir,frag2)
        files.append(glob.glob(path))
    files.sort()
    speclen=4096 # length of each spectra
    fs=250e6
    dt_spec = speclen/fs # time taken to read one spectra

    # find which file to read first 
    filetstamps = [int(f.split('.')[0].split('/')[-1]) for f in files]
    filetstamps.sort()
    filetstamps = np.asarray(filetstamps)

    # ------ SKIP -------#
    # make sure the sorted order of tstamps is same as of files. so that indices we'll find below correspond to correct files
    # np.unique(filetstamps - np.asarray([int(f.split('.')[0].split('/')[-1]) for f in files])) should return [0]

    # we're looking for a file that has the start timestamp closest to what we want
    fileidx = np.where(filetstamps<=init_t)[0][-1]
    #assumed that our init_t will most often lie inside some file. hardly ever a file will begin with our init timestamp

    # once we have a file, we seek to required position in time
    idxstart = int((init_t-filetstamps[fileidx])/dt_spec)
    # check that starting index indeed corresponds to init_t
    print("CHECK",init_t,idxstart*dt_spec + filetstamps[fileidx])
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