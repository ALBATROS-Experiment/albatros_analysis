import numpy as nm
import SNAPfiletools as sft
import pylab

#============================================================

if __name__ == '__main__':

    time_start = '20190720_030000'
    time_stop = '20190720_123000'
    data_dir = '/home/cynthia/working/arctic/data/data_auto_cross'
    logplot = True

    ctime_start = sft.timestamp2ctime(time_start)
    ctime_stop = sft.timestamp2ctime(time_stop)
    time, dat = sft.ctime2data(data_dir, ctime_start, ctime_stop)
    pol00 = dat[0]
    pol11 = dat[1]
    pol01 = dat[2] + 1J*dat[3]

    freq = nm.linspace(0, 125, 2048)
    
    axrange = [0, 125, 6.5, 8.5]
    myext = nm.array([0,125,pol01.shape[0],0])
            
    pylab.figure(figsize=(16,8), dpi=300)
        
    pylab.subplot(1,2,1)
    pylab.imshow(nm.log10(pol00), vmin=8, vmax=9.5, aspect='auto', extent=myext)
    pylab.title('pol00 (LWA EW) magnitude')

    pylab.subplot(1,2,2)
    pylab.imshow(nm.log10(pol11), vmin=8, vmax=9.5, aspect='auto', extent=myext)
    pylab.title('pol11 (LWA NS) magnitude')
        
    outfile = 'lwa_overnight_auto.png'
    pylab.savefig(outfile)
    print(f'Wrote {outfile}')
    pylab.close()

    pylab.figure(figsize=(16,8))
        
    pylab.subplot(1,2,1)
    pylab.imshow(nm.log10(nm.abs(pol01)), vmin=6, vmax=9, aspect='auto', extent=myext)
    pylab.title('pol01 magnitude')

    pylab.subplot(1,2,2)
    pylab.imshow(nm.angle(pol01), vmin=-nm.pi, vmax=nm.pi, aspect='auto', extent=myext)
    pylab.title('pol01 phase')
        
    outfile = 'lwa_overnight_cross.png'
    pylab.savefig(outfile)
    print(f'Wrote {outfile}')
    pylab.close()
    
