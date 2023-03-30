import numpy as np
from scio import scio
import matplotlib.pyplot as plt

import argparse, os, glob

parser = argparse.ArgumentParser()

parser.add_argument("ifiles", nargs="+")
parser.add_argument("oname")
parser.add_argument("-o", "--odir", type=str, default = "./plots/")
parser.add_argument("-p", "--ptype", type=str, default="flux")
parser.add_argument("-s", "--slices", default = -1)

args = parser.parse_args()

#ifiles  = sum([sorted(glob(ifile)) for ifile in args.ifiles],[])
fnames = args.ifiles
dat=scio.read_files(fnames)
try:
    dat=np.vstack(dat)
except:
    tmp=[]
    for i in range(len(dat)):
        if not(dat[i] is None):
            tmp.append(dat[i])
    dat=np.vstack(tmp)

print(dat.shape)

median_data=np.median(dat,axis=0)
      
#Norms data
dat_norm=np.repeat([median_data],dat.shape[0],axis=0)

plt.clf()
plt.imshow(dat/dat_norm,vmin=0.5,vmax=2)

plt.axis('auto')
plt.savefig(args.odir+str(args.oname)+'.png')


#Example usage
#python plot_one_file_scaled.py  /project/s/sievers/mohanagr/uapishka_aug_oct_2022/data_auto_cross/16610/1661011607/pol11.scio.bz2 test

