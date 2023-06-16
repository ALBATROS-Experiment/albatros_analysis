#usage: python missing_frac.py ~/Projects/baseband/SNAP1/16272

from correlations import baseband_data_classes as bdc
from glob import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_missing_frac(fname):
    obj=bdc.Baseband(fname)
    missing_frac = np.sum(obj.missing_num)/(obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet)
    return missing_frac


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirpath', type=str,help='Path to a 5-digit timestamp folder whose files you want to look at.')
    # parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='./',
    #           help='Output directory for data and plots')
    args = parser.parse_args()

    print("INPUT DIRPATH:", os.path.abspath(args.dirpath)+'/*')
    files = glob(os.path.abspath(args.dirpath)+'/*.raw')
    # print(files)
    files.sort()
    fracs = np.zeros(len(files))

    for i,file in enumerate(files):
        tag=file.split('/')[-1]
        fracs[i] = get_missing_frac(file)
        print(f"File {tag}, missing frac = {fracs[i]*100:5.3f}%")
    # print("Missing fracs are:", fracs)
    plt.plot(fracs)
    plt.show()




