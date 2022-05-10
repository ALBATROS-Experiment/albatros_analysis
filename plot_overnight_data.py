import os, sys
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


import numpy as nm
import scio, pylab, datetime, time
import SNAPfiletools as sft
from optparse import OptionParser


#============================================================

if __name__ == '__main__':



    parser = OptionParser()
    parser.set_usage('python plot_overnight_data.py <start time as YYYYMMDD_HHMMSS> <stop time as YYYYMMDD_HHMMSS> [options]')
    parser.set_description(__doc__)
    parser.add_option('-o', '--outdir', dest='outdir',type='str', default='/project/s/sievers/mohanagr/',
		      help='Output plot directory [default: %default]')
    parser.add_option('-d', '--datadir', dest='data_dir', type='str', default='/project/s/sievers/simont/mars_computed',
                      help='Baseband data directory [default: %default]')
    parser .add_option('-l', '--length', dest='readlen', type='int', default=1000, help='length of integration time in secconds [default: %default]')
    parser.add_option('-c', '--computed', dest='c_flag', action='store_true',
                    help='Use computed values instead of snap auto/cross  [default: %default]')
    opts, args = parser.parse_args(sys.argv[1:])


    if len(args) != 2:
        print('Please specify start and stop times.  Run with -h for more usage info.')
        exit(0)
    time_start = args[0]
    time_stop = args[1]

    ##ADDING THING TO PLOT THE DATA AS COMPUTED FROM THE OFFLINE CROSS AND AUTOS (APPLES TO APPLES)
    
    #cary over 
    #time_start = "20190720_000000"
    #time_stop = "20190720_235959"
    #data_dir = '/project/s/sievers/mars2019/auto_cross/data_auto_cross'
    #plot_dir = '/project/s/sievers/simont/baseband_plots_snap'

    data_dir = opts.data_dir
    plot_dir = opts.outdir
    #print(data_dir)
    
    logplot = False
    
    ctime_start = sft.timestamp2ctime(time_start)
    ctime_stop = sft.timestamp2ctime(time_stop)
    print('In cTime from: ' + str(ctime_start) + ' to ' + str(ctime_stop))

    data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)
    #lets figure out how to group them
    time_interval = 0
    data_set_index = 0
    temp_data =[]
    data_set =[]
    print(data_subdirs, "data subdirs")
    for index, val in enumerate(data_subdirs):
        if index == 0:
            prev_val = int(os.path.basename(val))
        
        if time_interval < opts.readlen:
            time_interval += (int(os.path.basename(val)) - prev_val)
            prev_val = int(os.path.basename(val))
            temp_data.append(str(val))
            #print("here!!!!")
        else:
            data_set.append(temp_data)
            temp_data=[]
            time_interval = 0
        print(time_interval)
    if(temp_data):
        data_set.append(temp_data)

    print(data_set, "dataset")
    p00=sft.readin_append(data_set[0], opts.data_dir, 'pol00.scio.bz2', scio.read)
    sys.exit()
   
    for data_subdir in data_set:

        tstamp_ctime = int(os.path.basename(data_subdir[0]))
        tstamp = sft.ctime2timestamp(tstamp_ctime)
        print('Processing ' + str(tstamp) + 'ctime ' + str(tstamp_ctime))
        
        
        if opts.c_flag is True:
            pol00 = sft.readin_append(data_subdir, opts.data_dir, 'pol00.npy', sft.readin_computed)
            pol11 = sft.readin_append(data_subdir, opts.data_dir, 'pol11.npy', sft.readin_computed)
            pol01m = sft.readin_append(data_subdir, opts.data_dir, 'pol01_mag.npy', sft.readin_computed)
            pol01p = sft.readin_append(data_subdir, opts.data_dir, 'pol01_phase.npy', sft.readin_computed)
        else:
            pol00 = sft.readin_append(data_subdir, opts.data_dir, 'pol00.scio.bz2', scio.read)
            pol11 = sft.readin_append(data_subdir, opts.data_dir, 'pol11.scio.bz2', scio.read)
            pol01r = sft.readin_append(data_subdir, opts.data_dir, 'pol01r.scio.bz2', scio.read)
            pol01i = sft.readin_append(data_subdir, opts.data_dir, 'pol01i.scio.bz2', scio.read)
            pol01 = pol01r + 1J*pol01i

        freq = nm.linspace(0, 125, nm.shape(pol00)[1])

        pol00_med = nm.median(pol00, axis=0)
        pol11_med = nm.median(pol11, axis=0)
        pol00_mean = nm.mean(pol00, axis=0)
        pol11_mean = nm.mean(pol11, axis=0)
        pol00_max = nm.max(pol00, axis = 0)
        pol11_max = nm.max(pol11, axis = 0)
        pol00_min = nm.min(pol00, axis = 0)
        pol11_min = nm.min(pol11, axis = 0)
        if opts.c_flag is True:
            pol01_min = nm.median(pol01m) - 2*nm.std(pol01m)
            pol01_max = nm.median(pol01m) + 2*nm.std(pol01m)
        # print(nm.shape(pol00_med))

        # print(pol00_mean)
        vmin = nm.median(pol00) - 2*nm.std(pol00)
        vmax = nm.median(pol00) + 2*nm.std(pol00)
        axrange = [0, 125, 0, 1e11]
        if logplot is True:
            pol00 = nm.log10(pol00)
            pol11 = nm.log10(pol11)
            pol00_mean = nm.log10(pol00_mean)
            pol11_mean = nm.log10(pol11_mean)
            pol00_med = nm.log10(pol00_med)
            pol11_med = nm.log10(pol11_med)
            pol00_max = nm.log10(pol00_max)
            pol11_max = nm.log10(pol11_max)
            pol00_min = nm.log10(pol00_min)
            pol11_min = nm.log10(pol11_min)
            vmin = nm.log10(vmin)
            vmax = nm.log10(vmax)
            axrange = [0, 125, 8, 10]

        myext = nm.array([0,125,pol00.shape[0],0])
            
        pylab.figure(figsize=(18,10) , dpi=200)
        
        pylab.subplot(2,3,1)
        pylab.imshow(pol00, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
        pylab.title('pol00')

        pylab.subplot(2,3,4)
        pylab.imshow(pol11, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
        pylab.title('pol11')
        
        pylab.subplot(2,3,2)
        pylab.plot(freq, pol00_max, 'r-', label='Max')
        pylab.plot(freq, pol00_min, 'b-', label='Min')
        pylab.plot(freq, pol00_mean, 'k-', label='Mean')
        pylab.plot(freq, pol00_med, color='#666666', linestyle='-', label='Median')
        pylab.ylim(vmin,vmax)
        pylab.xlabel('Freq (MHz)')
        pylab.ylabel('pol00')
        pylab.axis(axrange)

        pylab.subplot(2,3,5)
        pylab.plot(freq, pol11_max, 'r-', label='Max')
        pylab.plot(freq, pol11_min, 'b-', label='Min')
        pylab.plot(freq, pol11_mean, 'k-', label='Mean')
        pylab.plot(freq, pol11_med, color='#666666', linestyle='-', label='Median')
        pylab.ylim(vmin,vmax)
        pylab.xlabel('Freq (MHz)')
        pylab.ylabel('pol11')
        pylab.axis(axrange)
        pylab.legend(loc='lower right', fontsize='small')

        pylab.subplot(2,3,3)
        if opts.c_flag is True:
            pylab.imshow(pol01m, vmin=pol01_min, vmax=pol01_max, aspect='auto', extent=myext)
        else:    
            pylab.imshow(nm.log10(nm.abs(pol01)), vmin=6, vmax=9, aspect='auto', extent=myext)

        pylab.title('pol01 magnitude')

        pylab.subplot(2,3,6)
        if opts.c_flag is True:
            pylab.imshow(pol01p, vmin=-nm.pi, vmax=nm.pi, aspect='auto', extent=myext)
        else:    
            pylab.imshow(nm.angle(pol01), vmin=-nm.pi, vmax=nm.pi, aspect='auto', extent=myext)

        pylab.title('pol01 phase')
        
        pylab.suptitle(tstamp+' | '+str(tstamp_ctime), fontsize='large')

        outfile = os.path.join(plot_dir,tstamp+'.png')
        pylab.savefig(outfile)
        print('Wrote ' + outfile)
        pylab.close()
