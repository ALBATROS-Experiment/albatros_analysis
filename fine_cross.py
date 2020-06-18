import albatrostools
import matplotlib
matplotlib.use("Agg")
import os, sys, datetime, pylab
import numpy as np
import SNAPfiletools as sft
from optparse import OptionParser
import time
import shutil

from multiprocessing import Pool, get_context
import pfb_helper as pfb

def downsample2d(dat, dsfac):
    nrow = np.shape(dat)[0]
    ncol = np.shape(dat)[1]
    if nrow%dsfac != 0:
        print('downsample2d: number of rows not divisible by downsample factor')
        exit(0)
    return np.mean( np.reshape(dat, (nrow/dsfac, dsfac, ncol)), axis=1 )

def spec_resolve(data, bins):
    """
    Takes in some frequency domain data from the snap and pfbs it there and back to get finer resolution
    Will generate one time chunk with bins number of bins
    """
    

    ##do the invers
    time_stream = np.clip(pfb.inverse_pfb(data, 4).ravel(), -1,1)



    print(np.log2(len(time_stream)))
    ##snip off some data to make it a power of 2
    ##should make it a hell of a lot faster
    ##hoping we dont loose too much data :)
    ##also might deal with the edges well (we will see)
    current_len = len(time_stream)
    good_len = 2**(np.floor(np.log2(current_len)))
    snip = current_len - good_len
    if snip%2 == 0:
        time_stream = time_stream[snip/2:-snip/2 - 1]
    else:
        snip = snip + 1
        time_stream = time_stream[snip/2:-snip/2 - 1]
    if np.abs(int(np.log2(len(time_stream))) - np.log2(len(time_stream))) > 0.01:
        print("simon is dumb")

    #do the actual PFB
    spec = pfb.pfb(time_stream, bins, ntap=4)

    return sepc

if __name__ == "__main__":
    """
    Plot spectra at arbitrary precision

    """

    parser = OptionParser()
    parser.set_usage('python fine_cross.py <start time as YYYYMMDD_HHMMSS> <stop time as YYYYMMDD_HHMMSS> [options]')
    parser.set_description(__doc__)
    parser.add_option('-o', '--outdir', dest='outdir',type='str', default='/project/s/sievers/simont/baseband_plots',
		      help='Output plot directory [default: %default]')
    parser.add_option('-d', '--datadir', dest='data_dir', type='str', default='/project/s/sievers/mars2019/MARS1/albatros_north_baseband',
                      help='Baseband data directory [default: %default]')
    #parser .add_option('-l', '--length', dest='readlen', type='int', default=1000, help='Length of data to read from each raw file [default: %default]')
    parser.add_option('-z', '--dsfac', dest='dsfac', type='int', default=True,
		      help='Downsampling factor for # time samples (if True, then all time samples are averaged together [default: %default]')
    parser.add_option('-c', '--ctime', dest='c_flag', action='store_true',
                    help='Use Ctime instead of real time [default: %default]')
    parser.add_option('-b', '--bins', dest='nbins', type='int', default=1000,
		      help='Number of output frequency bins [default: %default]')
    parser.add_option('-p', '--cores', dest='ncores', type='int', default=4,
		      help='Number of cores to use [default: %default]')
    opts, args = parser.parse_args(sys.argv[1:])



    if len(args) != 2:
        print 'Please specify start and stop times.  Run with -h for more usage info.'
        exit(0)
    time_start = args[0]
    time_stop = args[1]

    if opts.c_flag is True:
        ctime_start = int(time_start)
        ctime_stop = int(time_stop)
    else:
        ctime_start = sft.timestamp2ctime(time_start)
        ctime_stop = sft.timestamp2ctime(time_stop)

    fnames = sft.time2fnames(ctime_start, ctime_stop, opts.data_dir)
    ##expects baseband in <First5DigitsOfCtime/Ctime.raw> 
    ##opts.data_dir should point to parent dir of above structure
    if len(fnames) == 0:
        print 'No files found in time range'
        exit(0)



    pol00 = None
    pol11 = None
    pol01_mag = None
    pol01_phase = None
    tstamps = []    
    for fname in fnames:
        print("Reading", fname)
        tstamp = os.path.basename(fname).split(".")[0]
        tstamps.append(tstamp)
        header, data = albatrostools.get_data(fname, items=-1, unpack_fast=False, float=True)
        ##unpack using c code do not forget to compile that stuff first!

    
        
        ##some remedial stuff to get ready for graphing later
        if header['bit_mode'] == 1:
            plot_auto = False
        else:
            plot_auto = True
        fmin = header['channels'][0]*125.0/2048
        fmax = header['channels'][-1]*125.0/2048
        
        ##heavy duty rebinning here:
        ##get variables set up
        n_cores = opts.ncores
        lines_per_core = 2**10
        n_bins = opts.nbins
        n_channels = header['channels'][-1]

        fs = 2* n_channels * 125e6 / 2048
        time_bin_size = 2 * (n_bins +4) /fs
        #might be useful


        data_new = {
        'pol0' : [],
        'pol1' : []
        }

        ## the actual rebinning
        ##for both pollarizations
        with get_context("spawn").Pool() as pool:
            for key, pol in data.items():
                ##split up the job into lines per core
                job = [(pol[x:(x+lines_per_core)], n_bins, n_channels) for x in range(0,np.shape(pol)[0]-1,lines_per_core)]
                ##assign the job
                result = pool.starmap(spec_resolve, [(1,1),(2,2),(3,3),(4,4),(5,5)])
                ##collaps result
                data_new[key] = np.array(result).ravel()



        # print 'Data dimensions are:', np.shape(data['pol0']), 'and', np.shape(data['pol1'])
        ntime = np.shape(data_new['pol0'])[0]
        nchan = np.shape(data_new['pol0'])[1]
        freqs = np.linspace(fmin, fmax, nchan)

        # Calculate auto and cross correlations
        corr = data_new['pol0']*np.conj(data_new['pol1'])

        ##this bit will turn the entire data stream into one set of values
        
        if opts.dsfac is True:
            time_bin_size = time_bin_size * ntime
            mean_pol00 = np.mean(np.abs(data_new['pol0'])**2, axis=0)
            mean_pol11 = np.mean(np.abs(data_new['pol1'])**2, axis=0)
            mean_mag = np.mean(np.abs(corr), axis=0)
            mean_phase = np.mean(np.angle(corr), axis=0)
        else:
            mean_pol00 = downsample2d(np.abs(data_new['pol0'])**2, opts.dsfac)
            mean_pol11 = downsample2d(np.abs(data_new['pol1'])**2, opts.dsfac)
            mean_mag = downsample2d(np.abs(corr), opts.dsfac)
            mean_phase = downsample2d(np.angle(corr), opts.dsfac)
        if pol01_mag is None:
            pol00 = mean_pol00.copy()
            pol11 = mean_pol11.copy()
            pol01_mag = mean_mag.copy()
            pol01_phase = mean_phase.copy()
        else:
            pol00 = np.vstack((pol00, mean_pol00))
            pol11 = np.vstack((pol11, mean_pol11))
            pol01_mag = np.vstack((pol01_mag, mean_mag))
            pol01_phase = np.vstack((pol01_phase, mean_phase))
        print 'Total dimensions are now', np.shape(pol01_mag)
    
    
    
    # Plot cross spectra
    pylab.figure(figsize=(16,16))
    myext = np.array([fmin, fmax, time_bin_size*np.shape(pol01_mag)[0], 0])
    pylab.subplot(2,2,1)
    vmin = np.median(pol01_mag) - 2*np.std(pol01_mag)
    vmax = np.median(pol01_mag) + 2*np.std(pol01_mag)
    pylab.imshow(pol01_mag, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol01 magnitude')
    pylab.subplot(2,2,2)
    pylab.imshow(pol01_phase, vmin=-np.pi, vmax=np.pi, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol01 phase')
    pylab.subplot(2,2,3)
    pylab.plot(np.median(pol01_phase, axis=1), 'k-')
    pylab.axis([0, np.shape(pol01_mag)[0], -np.pi, np.pi])
    pylab.xlabel('(Down)samples -- time')
    pylab.ylabel('Median phase')
    pylab.title('Pol01 median phase versus time')
    pylab.subplot(2,2,4)
    pylab.plot(np.median(pol01_phase, axis=0), 'k-')
    pylab.axis([fmin, fmax, -np.pi, np.pi])
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('Median phase')
    pylab.title('Pol01 median phase versus freq')
    pylab.suptitle(str(datetime.datetime.utcfromtimestamp(int(tstamps[0])))+' - '+str(datetime.datetime.utcfromtimestamp(int(tstamps[-1]))), fontsize=24)
    outfile = opts.outdir+'/coarse_cross_'+tstamps[0]+'-'+tstamps[-1]+'.png'
    pylab.savefig(outfile)
    print 'wrote', outfile
    pylab.close()

    # Plot autospectra
    if plot_auto is False:
        print 'Skipping autospectrum plots for 1-bit data'
        exit(0)
    pylab.figure(figsize=(16,16))
    myext = np.array([fmin, fmax,time_bin_size* np.shape(pol01_mag)[0], 0])

    pylab.subplot(2,2,1)
    vmin00 = np.median(pol00) - 2*np.std(pol00)
    vmax00 = np.median(pol00) + 2*np.std(pol00)
    pylab.imshow(pol00, vmin=vmin00, vmax=vmax00, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol00 magnitude')

    pylab.subplot(2,2,2)
    vmin11 = np.median(pol11) - 2*np.std(pol11)
    vmax11 = np.median(pol11) + 2*np.std(pol11)
    pylab.imshow(pol11, vmin=vmin11, vmax=vmax11, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol11 magnitude')

    pylab.subplot(2,2,3)
    pylab.plot(freqs, np.mean(pol00, axis=0), 'k-')
    pylab.plot(freqs, np.median(pol00, axis=0), 'r-')
    pylab.axis([fmin, fmax, vmin00, vmax00])
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('Pol00 magnitude')

    pylab.subplot(2,2,4)
    pylab.plot(freqs, np.mean(pol11, axis=0), 'k-', label='Mean')
    pylab.plot(freqs, np.median(pol11, axis=0), 'r-', label='Median')
    pylab.axis([fmin, fmax, vmin11, vmax11])
    pylab.legend(loc='lower right')
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('Pol11 magnitude')

    pylab.suptitle(str(datetime.datetime.utcfromtimestamp(int(tstamps[0])))+' - '+str(datetime.datetime.utcfromtimestamp(int(tstamps[-1]))), fontsize=24)
    outfile = opts.outdir+'/coarse_auto_'+tstamps[0]+'-'+tstamps[-1]+'.png'
    pylab.savefig(outfile)
    print('wrote', outfile)
    pylab.close()    