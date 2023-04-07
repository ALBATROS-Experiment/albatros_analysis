import numpy as np
from acoustics.cepstrum import complex_cepstrum
import os, argparse
from  scio import scio
import matplotlib.pyplot as plt
import scipy.signal as signal

def complex_cepstrum_from_spectrum(spectrum, n=None):
    r"""Compute the complex cepstrum of a spectrum.
    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    ndelay : int
        The amount of samples of circular delay added to `x`.
    The complex cepstrum is given by
    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}
    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    """

    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay
    
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase

    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay

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
    t = np.arange(pol00.shape[1]) / 250e6
    
    #if args.sim:
    #    pol00=[1 if i%int(args.sim) else 2 for i in range(pol00.shape[1])]
    #    pol00 = np.array(pol00)
    #    pol00 = np.repeat([pol00], pol11.shape[0],axis=0) 

    pol01 = pol01r + 1J*pol01i
    
    if args.stattype == "mean":
        pol00_stat = np.mean(pol00, axis=0)
        pol11_stat = np.mean(pol11, axis=0)

    if args.sim:
        #fundamental = 100.0
        harmonics = np.arange(1, 30)
        pol00_stat = 2+np.sin(2.0*np.pi*args.sim*t*harmonics[:, None]).sum(axis=0)
    
#    ceps00, _ = complex_cepstrum(pol00_stat)
#    ceps11, _ = complex_cepstrum(pol11_stat)

    ceps00, _ = complex_cepstrum_from_spectrum(pol00_stat)
    ceps11, _ = complex_cepstrum_from_spectrum(pol11_stat)
    print(ceps00)
#    ceps00 = np.fft.ifft(np.log(np.abs(pol00_stat))).real
#    ceps11 = np.fft.ifft(np.log(np.abs(pol11_stat))).real 
   
    peaks00, peak00_dict = signal.find_peaks(ceps00, height = 1e-3, prominence=1e-1, threshold=1e-1)
    print("Peaks pol00: ", (1/t[peaks00])/1e6,"MHz")

    peaks11, peak11_dict = signal.find_peaks(ceps11, height = 1e-3, prominence=1e-1, threshold=1e-1)
    print("Peaks pol11: ", (1/t[peaks11]/1e6),"MHz")

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(t, ceps00)
    ax0.scatter(t[peaks00], ceps00[peaks00], marker='x', color='red')
    ax0.set_xlabel('quefrency in seconds')
    ax0.set_yscale('log')
    ax0.set_title('pol00')
 
    ax1 = fig.add_subplot(212)
    ax1.plot(t, ceps11)
    ax1.scatter(t[peaks11], ceps11[peaks11], marker='x', color='red')
    ax1.set_xlabel('quefrency in seconds')
    ax1.set_yscale('log')
    ax1.set_title('pol11')

    timestamp = args.data_dir.split('/')[-1]
    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_cepstrum' + '.png')
    plt.savefig(outfile)
    plt.close()

    print("Saved to ", outfile)
    print("Prominence over peak height, pol00: ", peak00_dict['prominences']/peak00_dict['peak_heights'])
    print("Prominence over peak height, pol11: ", peak11_dict['prominences']/peak11_dict['peak_heights'])


    
