#usage: python missing_frac.py ~/Projects/baseband/SNAP1/16272

from correlations import baseband_data_classes as bdc
from glob import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_missing_frac(fname):
    '''
        E.g. say we write 100 spectra per file (could be discontinuous if stuff goes missing); baseband files are size limited.
        old way :   file says I should have 130 spectra (last specnum - first specnum), but 30 of them are missing (discontinuties)
                    so missing frac is 30/130
        new way :   I know we always write 100 spectra per file. file says I should have 130 spectra (apparent total)
                    which means 30 of them went missing. so missing frac is 30/100.
        Essentially, new way will be slightly higher than the old way.
    '''
    obj=bdc.Baseband(fname)
    missing_frac_old = np.sum(obj.missing_num)/(obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet)
    missing_frac_new = (obj.spec_num[-1]-obj.spec_num[0]+obj.spectra_per_packet)/(obj.num_packets*obj.spectra_per_packet) - 1
    return missing_frac_old, missing_frac_new


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
    fracs = np.zeros((len(files),2))
    try:
        for i,file in enumerate(files):
            tag=file.split('/')[-1]
            fracs[i] = get_missing_frac(file)
            print(f"File {tag}, missing frac old = {fracs[i][0]*100:5.3f}% | missing frac new = {fracs[i][1]*100:5.3f}%")
        # print("Missing fracs are:", fracs)
    except KeyboardInterrupt:
        print("exiting...")
    finally:
        plt.plot(fracs[:i],label=['old','new'])
        plt.legend()
        plt.show()




