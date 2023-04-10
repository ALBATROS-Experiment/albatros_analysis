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
import scipy.fft as fft

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
    parser.add_argument("-nf", "--numf", type = int, default=500, help = "number of interpolated points over which to comb")
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
    
    numf = args.numf

    t = np.arange(pol00.shape[1]) / 250e6
    freqs = np.arange(0, len(pol00_stat))*61035.15

    kernel = Gaussian1DKernel(2)
    pol00_stat = convolve(pol00_stat, kernel)
    pol11_stat = convolve(pol11_stat, kernel) 

    f00, harm00 = simple_harm_sweep(pol00_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = None)
    f11, harm11 = simple_harm_sweep(pol11_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = None) 

    f00_interp, harm00_interp = simple_harm_sweep(pol00_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = 'linear')
    f11_interp, harm11_interp = simple_harm_sweep(pol11_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = 'linear')

    #kernel = Gaussian1DKernel((4/500)*numf)
    #harm00 = convolve(harm00, kernel)
    #harm11 = convolve(harm11, kernel)

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

    fig = plt.figure(figsize=(14, 6))
    ax0 = fig.add_subplot(221)
    ax0.plot(f00_interp/1e6, harm00_interp)
    ax0.scatter(f00_interp[f00_max_interp]/1e6, harm00_interp[f00_max_interp], marker='x', color='red', zorder = 1)
    ax0.scatter(f00_interp[peaks00_interp]/1e6, harm00_interp[peaks00_interp], marker='x', color='black', zorder = 0) 
    ax0.set_xlabel('MHz')
    ax0.set_yscale('log')
    ax0.set_title('pol00 harmonics')
    ax0.set_xlim(0,15)

    ax1 = fig.add_subplot(223)
    ax1.plot(f11_interp/1e6, harm11_interp) 
    ax1.scatter(f11_interp[f11_max_interp]/1e6, harm11_interp[f11_max_interp], marker='x', color='red', zorder = 1)    
    ax1.scatter(f11_interp[peaks11_interp]/1e6, harm11_interp[peaks11_interp], marker='x', color='black', zorder = 0)
    ax1.set_xlabel('MHz')
    ax1.set_yscale('log')
    ax1.set_title('pol11 harmonics')
    ax1.set_xlim(0,15)
    
    #outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_{}_{}_interp'.format(tstart, tstop) + '.png')
    #plt.savefig(outfile)
    #plt.close()
    #print(outfile)
  
    #kernel = Gaussian1DKernel(2)
    #pol00_stat = convolve(pol00_stat, kernel)
    #pol11_stat = convolve(pol11_stat, kernel) 

    spectrum00_peaks, _ = signal.find_peaks(pol00_stat, height = 1e7, prominence=1e7, threshold=1e7) 
    spectrum11_peaks, _ = signal.find_peaks(pol11_stat, height = 1e7, prominence=1e7, threshold=1e7)

    
    ax2 = fig.add_subplot(222)  
    ax2.plot(freqs/1e6, pol00_stat)
    ax2.scatter(freqs[spectrum00_peaks]/1e6, pol00_stat[spectrum00_peaks], marker='x', color = 'black')
    ax2.set_xlabel('MHz') 
    ax2.set_yscale('log')
    ax2.set_title('pol00 spectrum')
    ax2.set_xlim(0,15)
    ax2.set_ylim(1e7,1e13)
    
    ax3 = fig.add_subplot(224)
    ax3.plot(freqs/1e6, pol11_stat)
    ax3.scatter(freqs[spectrum11_peaks]/1e6, pol11_stat[spectrum11_peaks], marker='x', color = 'black')
    #ax1.vlines(range(1, 10)*f11_interp[f11_max_interp]/1e6, 0, 1e14, color='black')
    #ax1.vlines(range(1, 20)*(f11_interp[peaks11_interp[1]])/1e6, 0, 1e14, color='black')
    ax3.set_xlabel('MHz')
    ax3.set_yscale('log')
    ax3.set_title('pol11 spectrum')
    ax3.set_xlim(0,15) 
    ax3.set_ylim(1e7, 1e13)
    
    #for i in range(min(3, len(peaks00_interp))):
    #    ax1.vlines(range(1,10)*(f00_interp[peaks00_interp[i]])/1e6, 0, 1e14, color = colors[i])  

    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_{}_{}_combined'.format(tstart, tstop) + '.png')

    plt.savefig(outfile) 
    plt.close()
    print(outfile)

    print('pol00')
    print('Spectrum peaks', freqs[spectrum00_peaks]/1e6)  
    print('Peak diffs: ', np.ediff1d(freqs[spectrum00_peaks]/1e6))
    print('Peaks div fundamental: ', freqs[spectrum00_peaks]/(f00_interp[peaks00_interp][0]))
    print('Peaks div max: ', freqs[spectrum00_peaks]/f00_interp[f00_max_interp])
    print('\n')
    print('pol11')
    print('Spectrum peaks', freqs[spectrum11_peaks]/1e6)
    print('Peak diffs: ', np.ediff1d(freqs[spectrum11_peaks]/1e6))
    print('Peaks div fundamental: ', freqs[spectrum11_peaks]/(f11_interp[peaks11_interp][0]))
    print('Peaks div max: ', freqs[spectrum11_peaks]/f11_interp[f11_max_interp])

    #pol00_fft = fft.fftshift(fft.fft(pol00_stat))
    #plt.plot(pol00_fft)

    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_{}_{}_fft'.format(tstart, tstop) + '.png')
    plt.savefig(outfile)


#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16807/1680766468
#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16808/1680851401
#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16809/1680937220
#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16810/1681023554



