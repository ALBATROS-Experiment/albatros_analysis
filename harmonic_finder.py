import numpy as np
from acoustics.cepstrum import complex_cepstrum
import os, argparse
from  scio import scio
import matplotlib.pyplot as plt
import scipy.signal as signal
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
    
    if args.sim:
        pol00=[0 if i%int(args.sim) else 1 for i in range(pol00.shape[1])]
        pol00 = np.array(pol00)
        pol00 = np.repeat([pol00], pol11.shape[0],axis=0) 

    pol01 = pol01r + 1J*pol01i
    
    if args.stattype == "mean":
        pol00_stat = np.mean(pol00, axis=0)
        pol11_stat = np.mean(pol11, axis=0)

    ceps00, _ = complex_cepstrum(pol00_stat)
    ceps11, _ = complex_cepstrum(pol11_stat)
   
    peaks, peak_dict = signal.find_peaks(ceps00, threshold=1e-1)
    print(1/t[peaks])

    plt.plot(t, ceps00)
    plt.scatter(t[peaks], ceps00[peaks], marker='x', color='red')
    plt.xlabel('quefrency in seconds')
    plt.yscale('log')

    timestamp = args.data_dir.split('/')[-1]
    outfile = os.path.normpath(args.output_dir + '/' + timestamp + '_cepstrum' + '.png')
    plt.savefig(outfile)
    plt.close()

    print("Saved to ", outfile)
    




