import numpy as np
import os, glob
from matplotlib import pyplot as plt

def get_init_info(init_t, end_t, parent_dir):
    '''
    Returns the index of file in a folder and 
    the index of the spectra in that file corresponding to init_timestamp
    '''
    # create a big list of files from 5 digit subdirs. we might not need all of them, but I don't want to write regex. 
    # This may be faster, and I don't care about storing a few 100 more strings than I need to.
    print("HELLO")
    frag1 = str(int(init_t/100000))
    frag2 = str(int(end_t/100000))
    print(frag1,frag2)
    path = os.path.join(parent_dir,frag1)
    files = glob.glob(path+'/*')
    if(frag1!=frag2):
        path = os.path.join(parent_dir,frag2)
        files.append(glob.glob(path+'/*'))
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
    print("Fileidx:", fileidx)
    print("CHECK",init_t,idxstart*dt_spec + filetstamps[fileidx])
    print("CHECK", filetstamps[fileidx], files[fileidx])
    
    return idxstart, fileidx, files

def get_plot_lims(pol,acclen):

    # numpy percentile method ignores mask and may generate garbage with 0s (missing specs). 
    # Pivot to using mean if acclen too small.

    if(acclen>250000):
        med = np.mean(pol)
        xx=np.ravel(pol).copy()
        u=np.percentile(xx,99)
        b=np.percentile(xx,1)
        xx_clean=xx[(xx<=u)&(xx>=b)] # remove some outliers for better plotting
        stddev = np.std(xx_clean)
    else:
        med = np.mean(pol)
        stddev = np.std(pol)
    vmin= max(med - 2*stddev,1)
    vmax = med + 2*stddev
    print(med,vmin,vmax)
    return med,vmin,vmax

def plot_4bit(pol00,pol11,pol01,channels,acclen,time_start,opath,minutes=False,logplot=True):

    freq = channels*125/2048 #MHz
    pol00_med = np.median(pol00, axis=0)
    pol11_med = np.median(pol11, axis=0)
    pol00_mean = np.mean(pol00, axis=0)
    pol11_mean = np.mean(pol11, axis=0)
    pol00_max = np.max(pol00, axis=0)
    pol11_max = np.max(pol11, axis=0)
    pol00_min = np.min(pol00, axis=0)
    pol11_min = np.min(pol11, axis=0)
    med,vmin,vmax=get_plot_lims(pol00,acclen)
    med2,vmin2,vmax2=get_plot_lims(pol11,acclen)
    pol01_mag = np.abs(pol01)
    if(logplot):
        pol00 = np.log10(pol00)
        pol11 = np.log10(pol11)
        pol00_med = np.log10(pol00_med)
        pol11_med = np.log10(pol11_med)
        pol00_mean = np.log10(pol00_mean)
        pol11_mean = np.log10(pol11_mean)
        pol00_max = np.log10(pol00_max)
        pol11_max = np.log10(pol11_max)
        pol00_min = np.log10(pol00_min)
        pol11_min = np.log10(pol11_min)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
        vmin2 = np.log10(vmin2)
        vmax2 = np.log10(vmax2)
        pol01_mag = np.log10(pol01_mag)

    plt.figure(figsize=(18,10), dpi=200)
    t_acclen = acclen*2048/125e6 #seconds
    t_end = pol01.shape[0]*t_acclen
    tag='Seconds'
    if(minutes):
        t_end = t_end/60
        tag='Minutes'
    myext = np.array([np.min(channels)*125/2048,np.max(channels)*125/2048, t_end, 0])
    plt.suptitle(f"{tag} since {time_start}")
    plt.subplot(2,3,1)
    plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
    plt.title('pol00')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(tag)
    cb00 = plt.colorbar()
    cb00.ax.plot([0, 1], [7.0]*2, 'w')

    plt.subplot(2,3,4)
    plt.imshow(pol11, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
    plt.title('pol11')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(tag)
    plt.colorbar()

    plt.subplot(2,3,2)
    plt.title('Basic stats for frequency bins')
    plt.plot(freq, pol00_max, 'r-', label='Max')
    plt.plot(freq, pol00_min, 'b-', label='Min')
    plt.plot(freq, pol00_mean, 'k-', label='Mean')
    plt.plot(freq, pol00_med, color='#666666', linestyle='-', label='Median')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('pol00')

    plt.subplot(2,3,5)
    plt.plot(freq, pol11_max, 'r-', label='Max')
    plt.plot(freq, pol11_min, 'b-', label='Min')
    plt.plot(freq, pol11_mean, 'k-', label='Mean')
    plt.plot(freq, pol11_med, color='#666666', linestyle='-', label='Median')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('pol11')
    plt.legend(loc='lower right', fontsize='small')

    plt.subplot(2,3,3)
    plt.imshow(pol01_mag, aspect='auto', extent=myext)
    plt.title('pol01 magnitude')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(tag)
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.imshow(np.angle(pol01), vmin=-np.pi, vmax=np.pi, aspect='auto', extent=myext, cmap='RdBu')
    plt.ylabel(tag)
    plt.xlabel('Frequency (MHz)')
    plt.title('pol01 phase')
    plt.colorbar()
    plt.savefig(opath)

def plot_1bit(pol01,channels,acclen,time_start,opath,minutes=False,logplot=False):
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(10,4)
    t_acclen = acclen*2048/125e6 #seconds
    t_end = pol01.shape[0]*t_acclen
    tag='Seconds'
    if(minutes):
        t_end = t_end/60
        tag='Minutes'
    myext = np.array([np.min(channels)*125/2048,np.max(channels)*125/2048, t_end, 0])

    plt.suptitle(f'{tag} since {time_start}')
    img1=ax[0].imshow(np.real(pol01),aspect='auto',vmin=-0.005,vmax=0.005, extent=myext)
    ax[0].set_title('pol01 real part')
    img2=ax[1].imshow(np.imag(pol01),aspect='auto',vmin=-0.005,vmax=0.005, extent=myext)
    ax[1].set_title('pol01 imag part')
    plt.colorbar(img1,ax=ax[0])
    plt.colorbar(img2,ax=ax[1])
    plt.savefig(opath)
    