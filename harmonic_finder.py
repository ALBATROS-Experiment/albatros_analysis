import numpy as np
from acoustics.cepstrum import complex_cepstrum
import os, sys, argparse
from  scio import scio
import matplotlib.pyplot as plt

from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve

from scipy.interpolate import interp1d
import scipy.signal as signal
from scipy.integrate import quad

from palettable.cartocolors.qualitative import Safe_10

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def simple_harm_sweep(x, freqs, fmin=None, fmax=None, numf = 1e5, harm_min = 1, harm_max = 5, window_size = None, interp = None):
    if fmin is None:
        fmin = min(freqs)
    if fmax is None:
        fmax = max(freqs) 
    numf = int(numf)

    if interp:
        fspace = np.linspace(fmin, fmax, numf)
        interp_x = interp1d(freqs, x, kind=interp)
        to_return = np.zeros(len(fspace))
        for i in range(len(to_return)-1):
            #Don't look at more than some number of  harmonics since it washes out the power 
            harm_freqs = np.arange(harm_min*fspace[i], min(max(freqs), harm_max*fspace[i]), fspace[i]) 

            if window_size is None: 
                to_return[i] = interp_x(harm_freqs).sum() / len(harm_freqs)
            else:
                gaus_window = lambda x: gaussian(x, harm_freqs[:, None], window_size).sum(axis=0)
                integrand = lambda x: interp_x(x)*gaus_window(x)
                to_return[i] = quad(integrand, min(freqs), max(freqs))[0] / quad(gaus_window, min(freqs), max(freqs))[0]
        return fspace, to_return /np.mean(x)

    else:
        to_return = np.zeros(len(x))
        index_fmin = int(np.floor(fmin/(freqs[1]-freqs[0])))
        index_fmax = int(np.floor(fmax/(freqs[1]-freqs[0])))
     
        
        
        #for i in range(10, int(len(x)/2)):
        for i in range(index_fmin, index_fmax):
            index_harm_min = int(np.floor((i+1)*harm_min))
            index_harm_max = min((i+1)*harm_max, len(x))  
            to_return[i] = x[i:index_harm_max:i].sum() / len(range(i, index_harm_max, i))
    

    return freqs, to_return/np.mean(x)
        
def _parse_slice(s):
    a = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*a)

def get_acctime(fpath):
    dat = np.fromfile(fpath,dtype='uint32')
    diff = np.diff(dat)
    acctime = np.mean(diff[(diff>0)&(diff<100)]) #sometimes timestamps are 0, which causes diff to be huge. could also use np. median
    return acctime

if __name__ == "__main__":
    "Example usage: python harmonic_finder.py -sl=14:15  ~/albatros_data/uapishka_april_23/data_auto_cross/16807/1680755638/" 
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Auto/cross-spectra location. Ex: ~/data_auto_cross/16171/161700000")
    parser.add_argument("-o", "--output_dir", type=str, default="./", help="Output directory for plots")
    parser.add_argument("-sl", "--tslice", type=_parse_slice, help="Slice on time axis to restrict plot to. Format: -sl=tmin:tmax for timin, tmax in minutes")
    parser.add_argument("-st", "--stattype", type=str, default="mean", help="Statisitcal method for reducing the data long time axis. Options are median or mean")
    parser.add_argument("-si", "--sim", type=int, default=0, help="sim")
    parser.add_argument("-fr", "--freqrange", type=_parse_slice, default=slice(5e5, 1e7, 1e5), help="Slice of freqeucny space over which to perform the comb")
    parser.add_argument("-hr", "--harmrange", type=_parse_slice, default=slice(1, 10, 1), help="First and last harmonic to consider in the harmonics comb")
    args = parser.parse_args()

    pol00 = scio.read(os.path.join(args.data_dir, "pol00.scio.bz2"))
    pol11 = scio.read(os.path.join(args.data_dir, "pol11.scio.bz2"))
    pol01r = scio.read(os.path.join(args.data_dir, "pol01r.scio.bz2"))
    pol01i = scio.read(os.path.join(args.data_dir, "pol01i.scio.bz2"))
    acctime = get_acctime(os.path.join(args.data_dir, "time_gps_start.raw"))

    pol00 = pol00[1:,:]
    pol11 = pol11[1:,:]
    pol01r = pol01r[1:,:]
    pol01i = pol01i[1:,:] 
    #fs = 16.384 #us

    if args.tslice:
        #convert tslice in minutes to samps
        tstart, tstop, tstep = args.tslice.start, args.tslice.stop, args.tslice.step

        if tstart is not None:
                tstart = int(np.floor(tstart*60/acctime))
        if tstop is not None:
                tstop = int(np.floor(tstop*60/acctime))
        if tstep is not None:
                tstep = int(np.floor(tstep*60/acctime))

        tslice = slice(tstart, tstop, tstep)

        pol00 = pol00[tslice, :]
        pol11 = pol11[tslice, :]
        pol01r = pol01r[tslice, :]
        pol01i = pol01i[tslice, :] 

    pol01 = pol01r + 1J*pol01i
    
    if args.stattype == "mean":
        pol00_stat = np.mean(pol00, axis=0)
        pol11_stat = np.mean(pol11, axis=0)

    
    fmin, fmax = args.freqrange.start, args.freqrange.stop
    hmin, hmax = args.harmrange.start, args.harmrange.stop
    
    t = np.arange(pol00.shape[1]) / 250e6
    freqs = np.arange(0, len(pol00_stat))*61035.15
    f00, harm00 = simple_harm_sweep(pol00_stat, freqs, fmin = fmin, fmax = fmax, numf = 500, harm_min = hmin, harm_max = hmax, window_size = None, interp = None)
    f11, harm11 = simple_harm_sweep(pol11_stat, freqs, fmin = fmin, fmax = fmax, numf = 500, harm_min = hmin, harm_max = hmax, window_size = None, interp = None) 

    f00_interp, harm00_interp = simple_harm_sweep(pol00_stat, freqs, fmin = fmin, fmax = fmax, numf = 500, harm_min = hmin, harm_max = hmax, window_size = None, interp = 'linear')
    f11_interp, harm11_interp = simple_harm_sweep(pol11_stat, freqs, fmin = fmin, fmax = fmax, numf = 500, harm_min = hmin, harm_max = hmax, window_size = None, interp = 'linear')

    kernel = Gaussian1DKernel(4)
    harm00 = convolve(harm00, kernel)
    harm11 = convolve(harm11, kernel)

    harm00_interp = convolve(harm00_interp, kernel)
    harm11_interp = convolve(harm11_interp, kernel) 

    peaks00, peak00_dict = signal.find_peaks(harm00, height = 1e-2, prominence=1e-2, threshold=1e-2)
    print("Peaks pol00: ", (f00[peaks00])/1e6,"MHz")

    peaks11, peak11_dict = signal.find_peaks(harm11, height = 1e-2, prominence=1e-2, threshold=1e-2)
    print("Peaks pol11: ", (f11[peaks11]/1e6),"MHz")

    peaks00_interp, peak00_dict_interp = signal.find_peaks(harm00_interp, height = 1e-2, prominence=1e-2, threshold=1e-2)
    print("Peaks pol00 interp: ", (f00_interp[peaks00_interp])/1e6,"MHz")

    peaks11_interp, peak11_dict_interp = signal.find_peaks(harm11_interp, height = 1e-2, prominence=1e-2, threshold=1e-2)
    print("Peaks pol11 interp: ", (f11_interp[peaks11_interp]/1e6),"MHz")

    f00_max = np.where((harm00 == np.amax(harm00)))[0]
    f11_max = np.where((harm11 == np.amax(harm11)))[0]
    f00_max_interp = np.where((harm00_interp == np.amax(harm00_interp)))[0]
    f11_max_interp = np.where((harm11_interp == np.amax(harm11_interp)))[0]

    print("Tmax pol00: ", f00[f00_max]/1e6)
    print("Tmax pol11: ", f11[f11_max]/1e6)
    print("Tmax pol00 interp: ", f00_interp[f00_max_interp]/1e6)
    print("Tmax pol11 interp: ", f11_interp[f11_max_interp]/1e6)

    colors = np.array(Safe_10.colors)/256

    timestamp = args.data_dir.split('/')[-1]

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(f00/1e6, harm00)
    ax0.scatter(f00[f00_max]/1e6, harm00[f00_max], marker='x', color='red', zorder = 1)
    ax0.scatter(f00[peaks00]/1e6, harm00[peaks00], marker='x', color='black', zorder = 0)
    ax0.set_xlabel('MHz')
    ax0.set_yscale('log')
    ax0.set_title('pol00')
    ax0.set_xlim(0, 10)

    ax1 = fig.add_subplot(212)
    ax1.plot(f11/1e6, harm11)
    ax1.scatter(f11[f11_max]/1e6, harm11[f11_max], marker='x', color='red', zorder = 1)
    ax1.scatter(f11[peaks11]/1e6, harm11[peaks11], marker='x', color='black', zorder = 0)
    ax1.set_xlabel('MHz')
    ax1.set_yscale('log')
    ax1.set_title('pol11')
    ax1.set_xlim(0,10)
 
    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_{}_{}'.format(tstart, tstop) + '.png')
    print(outfile)
    plt.savefig(outfile)
    plt.close() 

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(f00_interp/1e6, harm00_interp)
    ax0.scatter(f00_interp[f00_max_interp]/1e6, harm00_interp[f00_max_interp], marker='x', color='red', zorder = 1)
    ax0.scatter(f00_interp[peaks00_interp]/1e6, harm00_interp[peaks00_interp], marker='x', color='black', zorder = 0) 
    ax0.set_xlabel('MHz')
    ax0.set_yscale('log')
    ax0.set_title('pol00')
    ax0.set_xlim(0,10)

    ax1 = fig.add_subplot(212)
    ax1.plot(f11_interp/1e6, harm11_interp) 
    ax1.scatter(f11_interp[f11_max_interp]/1e6, harm11_interp[f11_max_interp], marker='x', color='red', zorder = 1)    
    ax1.scatter(f11_interp[peaks11_interp]/1e6, harm11_interp[peaks11_interp], marker='x', color='black', zorder = 0)
    ax1.set_xlabel('MHz')
    ax1.set_yscale('log')
    ax1.set_title('pol11')
    ax1.set_xlim(0,10)
    
    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_{}_{}_interp'.format(tstart, tstop) + '.png')
    plt.savefig(outfile)
    plt.close()
    print(outfile)
  
    fig = plt.figure()
    ax0 = fig.add_subplot(211) 
    ax0.plot(freqs/1e6, pol00_stat)
    ax0.vlines(range(1, 10)*f00_interp[f00_max_interp]/1e6, 0, 1e14, color='black')
    ax0.set_xlabel('MHz') 
    ax0.set_yscale('log')
    ax0.set_title('pol00')
    ax0.set_xlim(0,30)
    ax0.set_ylim(1e7,1e12)

    spectrum_peaks, _ = signal.find_peaks(pol11_stat, height = 1e10, prominence=1e10, threshold=1e10) 
    
    ax1 = fig.add_subplot(212)
    ax1.plot(freqs/1e6, pol11_stat)
    ax1.scatter(freqs[spectrum_peaks]/1e6, pol11_stat[spectrum_peaks], marker='x', color = 'black')
    #ax1.vlines(range(1, 10)*f11_interp[f11_max_interp]/1e6, 0, 1e14, color='black')
    #ax1.vlines(range(1, 20)*(f11_interp[peaks11_interp[1]])/1e6, 0, 1e14, color='black')
    ax1.set_xlabel('MHz')
    ax1.set_yscale('log')
    ax1.set_title('pol11')
    ax1.set_xlim(0,30) 
    ax1.set_ylim(1e7, 1e13)
    
    #for i in range(min(3, len(peaks00_interp))):
    #    ax1.vlines(range(1,10)*(f00_interp[peaks00_interp[i]])/1e6, 0, 1e14, color = colors[i])  

    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_{}_{}_spectrum'.format(tstart, tstop) + '.png')
    plt.savefig(outfile) 
    plt.close()
    print(outfile)

    print('Spectrum peaks', freqs[spectrum_peaks]/1e6)  



