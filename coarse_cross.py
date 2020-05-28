import albatrostools
import matplotlib
matplotlib.use("Agg")
import os, sys, datetime, pylab
import numpy as nm
import SNAPfiletools as sft
from optparse import OptionParser

#============================================================
def downsample2d(dat, dsfac):
    nrow = nm.shape(dat)[0]
    ncol = nm.shape(dat)[1]
    if nrow%dsfac != 0:
        print 'downsample2d: number of rows not divisible by downsample factor'
        exit(0)
    return nm.mean( nm.reshape(dat, (nrow/dsfac, dsfac, ncol)), axis=1 )

#============================================================
if __name__=="__main__":

    """
    Plot coarsely sampled auto- and cross-spectra calculated from
    baseband.  This script searches for baseband files within a
    specified time range, reads a small snippet of data from each,
    calculates auto- and cross-spectra, then averages those spectra by
    the specified downsampling factor, and plots the results.
    """
    
    parser = OptionParser()
    parser.set_usage('python coarse_cross.py <start time as YYYYMMDD_HHMMSS> <stop time as YYYYMMDD_HHMMSS> [options]')
    parser.set_description(__doc__)
    parser.add_option('-o', '--outdir', dest='outdir',type='str', default='baseband_plots',
		      help='Output plot directory [default: %default]')
    parser.add_option('-d', '--datadir', dest='data_dir', type='str', default='/media/cynthia/MARS2/albatros_north_baseband',
                      help='Baseband data directory [default: %default]')
    parser .add_option('-l', '--length', dest='readlen', type='int', default=1000, help='Length of data to read from each raw file [default: %default]')
    parser.add_option('-z', '--dsfac', dest='dsfac', type='int', default=True,
		      help='Downsampling factor for # time samples (if True, then all time samples are averaged together [default: %default]')
    opts, args = parser.parse_args(sys.argv[1:])

    if len(args) != 2:
        print 'Please specify start and stop times.  Run with -h for more usage info.'
        exit(0)
    time_start = args[0]
    time_stop = args[1]

    ctime_start = sft.timestamp2ctime(time_start)
    ctime_stop = sft.timestamp2ctime(time_stop)
    fnames = sft.time2fnames(ctime_start, ctime_stop, opts.data_dir)
    if len(fnames) == 0:
        print 'No files found in time range'
        exit(0)

    pol00 = None
    pol11 = None
    pol01_mag = None
    pol01_phase = None
    tstamps = []
    for fname in fnames:
        print 'Reading', fname
        fname_sub = fname.split('/')[-1]
        tstamp = fname_sub.split('.')[0]
        tstamps.append(tstamp)
        header, data = albatrostools.get_data(fname, items=opts.readlen)
        if header['bit_mode'] == 1:
            plot_auto = False
        else:
            plot_auto = True
        fmin = header['channels'][0]*125.0/2048
        fmax = header['channels'][-1]*125.0/2048
        # print 'Data dimensions are:', nm.shape(data['pol0']), 'and', nm.shape(data['pol1'])
        ntime = nm.shape(data['pol0'])[0]
        nchan = nm.shape(data['pol0'])[1]
        freqs = nm.linspace(fmin, fmax, nchan)

        # Calculate auto and cross correlations
        corr = data['pol0']*nm.conj(data['pol1'])
        # Append averaged data
        if opts.dsfac is True:
            mean_pol00 = nm.mean(nm.abs(data['pol0'])**2, axis=0)
            mean_pol11 = nm.mean(nm.abs(data['pol1'])**2, axis=0)
            mean_mag = nm.mean(nm.abs(corr), axis=0)
            mean_phase = nm.mean(nm.angle(corr), axis=0)
        else:
            mean_pol00 = downsample2d(nm.abs(data['pol0'])**2, opts.dsfac)
            mean_pol11 = downsample2d(nm.abs(data['pol1'])**2, opts.dsfac)
            mean_mag = downsample2d(nm.abs(corr), opts.dsfac)
            mean_phase = downsample2d(nm.angle(corr), opts.dsfac)
        if pol01_mag is None:
            pol00 = mean_pol00.copy()
            pol11 = mean_pol11.copy()
            pol01_mag = mean_mag.copy()
            pol01_phase = mean_phase.copy()
        else:
            pol00 = nm.vstack((pol00, mean_pol00))
            pol11 = nm.vstack((pol11, mean_pol11))
            pol01_mag = nm.vstack((pol01_mag, mean_mag))
            pol01_phase = nm.vstack((pol01_phase, mean_phase))
        print 'Total dimensions are now', nm.shape(pol01_mag)

    # Plot cross spectra
    pylab.figure(figsize=(16,16))
    myext = nm.array([fmin, fmax, nm.shape(pol01_mag)[0], 0])
    pylab.subplot(2,2,1)
    vmin = nm.median(pol01_mag) - 2*nm.std(pol01_mag)
    vmax = nm.median(pol01_mag) + 2*nm.std(pol01_mag)
    pylab.imshow(pol01_mag, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol01 magnitude')
    pylab.subplot(2,2,2)
    pylab.imshow(pol01_phase, vmin=-nm.pi, vmax=nm.pi, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol01 phase')
    pylab.subplot(2,2,3)
    pylab.plot(nm.median(pol01_phase, axis=1), 'k-')
    pylab.axis([0, nm.shape(pol01_mag)[0], -nm.pi, nm.pi])
    pylab.xlabel('(Down)samples -- time')
    pylab.ylabel('Median phase')
    pylab.title('Pol01 median phase versus time')
    pylab.subplot(2,2,4)
    pylab.plot(nm.median(pol01_phase, axis=0), 'k-')
    pylab.axis([fmin, fmax, -nm.pi, nm.pi])
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
    myext = nm.array([fmin, fmax, nm.shape(pol01_mag)[0], 0])

    pylab.subplot(2,2,1)
    vmin00 = nm.median(pol00) - 2*nm.std(pol00)
    vmax00 = nm.median(pol00) + 2*nm.std(pol00)
    pylab.imshow(pol00, vmin=vmin00, vmax=vmax00, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol00 magnitude')

    pylab.subplot(2,2,2)
    vmin11 = nm.median(pol11) - 2*nm.std(pol11)
    vmax11 = nm.median(pol11) + 2*nm.std(pol11)
    pylab.imshow(pol11, vmin=vmin11, vmax=vmax11, aspect='auto', extent=myext)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('(Down)samples')
    pylab.title('Pol11 magnitude')

    pylab.subplot(2,2,3)
    pylab.plot(freqs, nm.mean(pol00, axis=0), 'k-')
    pylab.plot(freqs, nm.median(pol00, axis=0), 'r-')
    pylab.axis([fmin, fmax, vmin00, vmax00])
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('Pol00 magnitude')

    pylab.subplot(2,2,4)
    pylab.plot(freqs, nm.mean(pol11, axis=0), 'k-', label='Mean')
    pylab.plot(freqs, nm.median(pol11, axis=0), 'r-', label='Median')
    pylab.axis([fmin, fmax, vmin11, vmax11])
    pylab.legend(loc='lower right')
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('Pol11 magnitude')

    pylab.suptitle(str(datetime.datetime.utcfromtimestamp(int(tstamps[0])))+' - '+str(datetime.datetime.utcfromtimestamp(int(tstamps[-1]))), fontsize=24)
    outfile = outdir+'/coarse_auto_'+tstamps[0]+'-'+tstamps[-1]+'.png'
    pylab.savefig(outfile)
    print 'wrote', outfile
    pylab.close()    
