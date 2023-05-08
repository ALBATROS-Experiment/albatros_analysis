import numpy as np
#from acoustics.cepstrum import complex_cepstrum
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

def triangle(start, stop, freqs):
    '''Function which returns a single triangle window with designated start and stop.
       Input: start, the start frequency
              stop, the stop frequency
              freqs, an array of the freqs over which the triangle window will be applies
    '''
    mid = (stop+start)/2
    to_return = np.zeros(len(freqs))
    for i in range(len(freqs)):
        if freqs[i]<start or freqs[i] > stop:
            continue
        elif freqs[i] <= mid:
            to_return[i] = (freqs[i] - start) / (mid - start)
        elif freqs[i] > mid:
            to_return[i] = (freqs[i] - stop) / (mid-stop)

    return to_return    

def get_Hb(freqs, nb=30):
    '''Function for generating the power response function Hb from klapuri06
       Inputs: freqs, the frequencies associated with x
               nb, the number of subbands
       Outputs: Hb, the responses
    '''
    bs = np.arange(0,nb+2,1) 
    cb =  229 * (10**((1+bs)/(21.4))-1)*1.80e4
    Hb = np.zeros((len(freqs), nb)) 
    for i in range(1, nb+1):
        Hb[...,i-1] = triangle(cb[i-1], cb[i+1], freqs)
    
    return Hb, cb[1:-1]

def whittener(x, freqs, nu=0.33, nb = 30):
    ''' Spectral whittening function from Klapuri06: https://www.ee.columbia.edu/~dpwe/papers/Klap06-multif0.pdf
        Input: x, the spectrum to be whitened
               nu, a whittening scaling parameter
        Reutrns: gamma(f), the whittening function as a function of frequency.
    '''
    Hb, cb = get_Hb(freqs = freqs, nb = nb)
    sigma_bs = np.zeros(len(cb))
    test = x/ 1e10
    for i in range(len(sigma_bs)):  
        sigma_bs[i] = np.sqrt(1/len(x) * sum(Hb[...,i] * np.abs(x)**2))  
        
    yb = sigma_bs**(nu-1)
    
    return interp1d(cb, yb, fill_value = 'extrapolate') 


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def simple_harm_sweep(x, freqs, fmin=None, fmax=None, numf = 1e5, harm_min = 1, harm_max = 5, window_size = None, interp = None):
    '''This function takes a spectrum and applies a harmonic comb to it, identifying which frequencies f are present in the spectrum at 1*f, 2*f, 3*f, etc. Useful when you suspect that there is a harmonic series present in a spectrum, and would like to identify what the fundamentals are

        Input: x, the spectrum to be considered
               freqs, the coresponding frequencies of the spectrum
               fmin, the minimum fundamental frequency to consider
               fmax, the maximum fundamental frequency to consider 
               numf, the number of points in fundamental frequency space to sample
               harm_min, the lowest harmonic on the frequency comb. harm_min=1 gives the simplest case where we begin the comb at the fundamental, harm_min = 2 starts with the second harmonic, etc.
               harm_max, the highest harmonic of the comb. Including too many harmonics washes out the comb power
               window, depreciated, type of window by which the spectrum is smoothed. Superceded by whittening, but kept in case we want to return to smoothing 
               interp, how to interpolate the spectrum for applying the comb. If none, no interpolation is performed
       Returns: fspace if interp, else freqs. The fundamental frequencies. When no interpolating, this is the same as the input frequencies.
                to_return, the comb response at fspace/freqs, normalized by the average spectrum.    '''

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
    if args.stattype == "max":
        pol00_stat = np.amax(pol00, axis=0)
        pol11_stat = np.amax(pol11, axis=0)
        
        pol00_stat = np.array(pol00_stat, dtype=float)
        pol11_stat = np.array(pol11_stat, dtype=float)
 
    fmin, fmax = args.freqrange.start, args.freqrange.stop
    hmin, hmax = args.harmrange.start, args.harmrange.stop
    
    numf = args.numf

    t = np.arange(pol00.shape[1]) / 250e6
    freqs = np.arange(0, len(pol00_stat))*61035.15       
       
    whittener00 = whittener(pol00_stat, freqs, nu = 0.3)
    whittener11 = whittener(pol11_stat, freqs, nu = 0.3) 

    kernel = Gaussian1DKernel(2)
    #uncomment to smooth the spectrum by kernel before whittening
    #pol00_stat = convolve(pol00_stat, kernel)
    #pol11_stat = convolve(pol11_stat, kernel) 
    pol00_unwhite = pol00_stat
    pol11_unwhite = pol11_stat 

    pol00_stat = whittener00(freqs)*pol00_stat
    pol11_stat = whittener11(freqs)*pol11_stat
    
    #Code returns both interpolated and no-interpolated harmonic combs as of now, but they generally give the same results and interpolated is much easier to parse so may just remove the non-interp in the future
    f00, harm00 = simple_harm_sweep(pol00_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = None)
    f11, harm11 = simple_harm_sweep(pol11_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = None) 

    f00_interp, harm00_interp = simple_harm_sweep(pol00_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = 'linear')
    f11_interp, harm11_interp = simple_harm_sweep(pol11_stat, freqs, fmin = fmin, fmax = fmax, numf = numf, harm_min = hmin, harm_max = hmax, window_size = None, interp = 'linear')

    #Uncomment to smooth the resulting harmonic comb
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
   
    #Kernel is same as above but can uncomment here to change 
    #kernel = Gaussian1DKernel(2)
    pol00_unwhite = convolve(pol00_unwhite, kernel)
    pol11_unwhite = convolve(pol11_unwhite, kernel) 

    spectrum00_peaks, _ = signal.find_peaks(pol00_unwhite, height = 1e7, prominence=1e7, threshold=1e7) 
    spectrum11_peaks, _ = signal.find_peaks(pol11_unwhite, height = 1e7, prominence=1e7, threshold=1e7)

    
    ax2 = fig.add_subplot(222)  
    ax2.plot(freqs/1e6, pol00_unwhite)
    ax2.scatter(freqs[spectrum00_peaks]/1e6, pol00_unwhite[spectrum00_peaks], marker='x', color = 'black')
    ax2.set_xlabel('MHz') 
    ax2.set_yscale('log')
    ax2.set_title('pol00 spectrum')
    ax2.set_xlim(0,15)
    ax2.set_ylim(1e7,1e13)
    
    ax3 = fig.add_subplot(224)
    ax3.plot(freqs/1e6, pol11_unwhite)
    ax3.scatter(freqs[spectrum11_peaks]/1e6, pol11_unwhite[spectrum11_peaks], marker='x', color = 'black')
    
    ax3.set_xlabel('MHz')
    ax3.set_yscale('log')
    ax3.set_title('pol11 spectrum')
    ax3.set_xlim(0,15) 
    ax3.set_ylim(1e7, 1e13)
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

    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_{}_{}_fft'.format(tstart, tstop) + '.png')
    plt.savefig(outfile)


#Some example usage
#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16807/1680766468
#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16808/1680851401
#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16809/1680937220
#python harmonic_finder.py -hr=2:10 -sl=0:60 -o=./plots ~/albatros_data/uapishka_april_23/data_auto_cross/16810/1681023554



