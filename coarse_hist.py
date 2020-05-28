import albatrostools
import matplotlib
matplotlib.use("Agg")
import os, sys, datetime, pylab
import numpy as nm
import SNAPfiletools as sft
from optparse import OptionParser

if __name__=="__main__":

    """
    Plot coarsely sampled baseband histograms.  This script searches
    for baseband files within a specified time range, reads a small
    snippet of data from each, histograms each channel (for real/imag
    of each polarization separately, then plots the results in
    waterfall form.
    """
    
    parser = OptionParser()
    parser.set_usage('python coarse_hist.py <start time as YYYYMMDD_HHMMSS> <stop time as YYYYMMDD_HHMMSS> [options]')
    parser.set_description(__doc__)
    parser.add_option('-o', '--outdir', dest='outdir',type='str', default='baseband_plots',
		      help='Output plot directory [default: %default]')
    parser.add_option('-d', '--datadir', dest='data_dir', type='str', default='/media/cynthia/MARS2/albatros_north_baseband',
                      help='Baseband data directory [default: %default]')
    parser.add_option('-r', '--row', dest='nrow', type='int', default=12,
                      help='Number of subplot rows [default: %default]')
    parser.add_option('-c', '--col', dest='ncol', type='int', default=10,
                      help='Number of subplot columns [default: %default]')
    parser .add_option('-l', '--length', dest='readlen', type='int', default=2000, help='Length of data to read from each raw file [default: %default]')
    opts, args = parser.parse_args(sys.argv[1:])

    if len(args) != 2:
        print 'Please specify start and stop times.  Run with -h for more usage info.'
        exit(0)
    time_start = args[0]
    time_stop = args[1]

    ctime_start = sft.timestamp2ctime(time_start)
    ctime_stop = sft.timestamp2ctime(time_stop)
    fnames = sft.time2fnames(ctime_start, ctime_stop, opts.data_dir)

    allhist = {}
    tstamps = []
    for fname in fnames:
        print 'Reading', fname
        fname_sub = fname.split('/')[-1]
        tstamp = fname_sub.split('.')[0]
        tstamps.append(tstamp)
        header, data = albatrostools.get_data(fname, items=opts.readlen)
        fmin = header['channels'][0]*125.0/2048
        fmax = header['channels'][-1]*125.0/2048
        # print 'Data dimensions are:', nm.shape(data['pol0']), 'and', nm.shape(data['pol1'])
        ntime = nm.shape(data['pol0'])[0]
        nchan = nm.shape(data['pol0'])[1]
        freqs = nm.linspace(fmin, fmax, nchan)

        if header['bit_mode'] == 2:
            bins = nm.arange(-2.25, 3, 1.25)
            bmin = -2
            bmax = 2
        elif header['bit_mode'] == 4:
            bins = nm.arange(-7, 9, 1)
            bmin = -8
            bmax = 8
        else:
            print 'Bit mode is neither 2 nor 4, giving up'
            exit(0)

        # Histogram real and imaginary parts for each channel
        for index, chan in enumerate(header['channels']):
            pol0r, edges = nm.histogram(nm.real(data['pol0'][:,index]), bins, normed=False, weights=None)
            pol0i, edges = nm.histogram(nm.imag(data['pol0'][:,index]), bins, normed=False, weights=None)
            pol1r, edges = nm.histogram(nm.real(data['pol1'][:,index]), bins, normed=False, weights=None)
            pol1i, edges = nm.histogram(nm.imag(data['pol1'][:,index]), bins, normed=False, weights=None)

            if fname is fnames[0]:
                allhist['pol0r_'+str(chan)] = pol0r.copy()
                allhist['pol0i_'+str(chan)] = pol0i.copy()
                allhist['pol1r_'+str(chan)] = pol1r.copy()
                allhist['pol1i_'+str(chan)] = pol1i.copy()
            else:
                allhist['pol0r_'+str(chan)] = nm.vstack((allhist['pol0r_'+str(chan)], pol0r))
                allhist['pol0i_'+str(chan)] = nm.vstack((allhist['pol0i_'+str(chan)], pol0i))
                allhist['pol1r_'+str(chan)] = nm.vstack((allhist['pol1r_'+str(chan)], pol1r))
                allhist['pol1i_'+str(chan)] = nm.vstack((allhist['pol1i_'+str(chan)], pol1i))
    # End loop over files

    # Plot ALL the things
    for pol in range(2):
        for ri in ['r','i']:
            pylab.figure(figsize=(30,25))
            for index, chan in enumerate(header['channels']):
                dat = allhist['pol'+str(pol)+ri+'_'+str(chan)]
                pylab.subplot(opts.nrow, opts.ncol, index+1)
                myext = nm.array([bmin, bmax, nm.shape(dat)[0], 0])
                pylab.imshow(dat, aspect='auto', extent=myext)
                pylab.title(str(chan)+' | '+str(freqs[index]))
                pylab.subplots_adjust(hspace=0.75, wspace=0.5)
            pylab.suptitle('Coarse histograms : pol'+str(pol)+ri+' | '+str(datetime.datetime.utcfromtimestamp(int(tstamps[0])))+' - '+str(datetime.datetime.utcfromtimestamp(int(tstamps[-1]))), fontsize=24)
            outfile = opts.outdir+'/coarse_hist_pol'+str(pol)+ri+'_'+tstamps[0]+'-'+tstamps[-1]+'.png'
            pylab.savefig(outfile)
            print 'Wrote', outfile
