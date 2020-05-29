import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


import numpy as nm
import scio, pylab, datetime, time
import SNAPfiletools as sft


#============================================================

if __name__ == '__main__':

    time_start = "20190720_000000"
    time_stop = "20190720_235959"
    data_dir = '/project/s/sievers/mars2019/auto_cross/data_auto_cross'
    plot_dir = '/project/s/sievers/simont/baseband_plots_snap'
    logplot = True
    
    ctime_start = sft.timestamp2ctime(time_start)
    ctime_stop = sft.timestamp2ctime(time_stop)
    print('In cTime from: ' + str(ctime_start) + '. to: ' + str(ctime_stop))

    data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)

    freq = nm.linspace(0, 125, 2048)
   
    for data_subdir in data_subdirs:

        tstamp_ctime = int(data_subdir.split('/')[-1])
        tstamp = sft.ctime2timestamp(tstamp_ctime)
        print('Processing ' + str(tstamp) + 'ctime ' + str(tstamp_ctime))
        
        pol00 = scio.read(data_subdir+'/pol00.scio.bz2')
        pol11 = scio.read(data_subdir+'/pol11.scio.bz2')
        pol01r = scio.read(data_subdir+'/pol01r.scio.bz2')
        pol01i = scio.read(data_subdir+'/pol01i.scio.bz2')
        pol01 = pol01r + 1J*pol01i

        pol00_med = nm.median(pol00, axis=0)
        pol11_med = nm.median(pol11, axis=0)
        pol00_mean = nm.mean(pol00, axis=0)
        pol11_mean = nm.mean(pol11, axis=0)
        pol00_max = nm.max(pol00, axis=0)
        pol11_max = nm.max(pol11, axis=0)
        pol00_min = nm.min(pol00, axis=0)
        pol11_min = nm.min(pol11, axis=0)

        vmin = 0
        vmax = 1e11
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
            vmin = 8
            vmax = 9.5
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
        pylab.xlabel('Freq (MHz)')
        pylab.ylabel('pol00')
        pylab.axis(axrange)

        pylab.subplot(2,3,5)
        pylab.plot(freq, pol11_max, 'r-', label='Max')
        pylab.plot(freq, pol11_min, 'b-', label='Min')
        pylab.plot(freq, pol11_mean, 'k-', label='Mean')
        pylab.plot(freq, pol11_med, color='#666666', linestyle='-', label='Median')
        pylab.xlabel('Freq (MHz)')
        pylab.ylabel('pol11')
        pylab.axis(axrange)
        pylab.legend(loc='lower right', fontsize='small')

        pylab.subplot(2,3,3)
        pylab.imshow(nm.log10(nm.abs(pol01)), vmin=6, vmax=9, aspect='auto', extent=myext)
        pylab.title('pol01 magnitude')

        pylab.subplot(2,3,6)
        pylab.imshow(nm.angle(pol01), vmin=-nm.pi, vmax=nm.pi, aspect='auto', extent=myext)
        pylab.title('pol01 phase')
        
        pylab.suptitle(tstamp+' | '+str(tstamp_ctime), fontsize='large')

        outfile = plot_dir+'/'+tstamp+'.png'
        pylab.savefig(outfile)
        print('Wrote ' + outfile)
        pylab.close()
