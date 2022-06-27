from asyncio import as_completed
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import scio

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bbfilepath', type=str,help='Path to npz file. E.g. ~/pol01_4bit.npz')
    parser.add_argument('direct_dir', type=str,help='Path to direct data directory of relevant timestamp. E.g. ~/16272/1627202094/')
    parser.add_argument('acclen', type=int,help='Accumulation length that was used to generate direct and baseband npz file')
    parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='/scratch/s/sievers/mohanagr/',
              help='Output directory for data and plots')
    args = parser.parse_args()

    pol01r = scio.read(os.path.join(args.direct_dir,'pol01r.scio.bz2'))
    pol01i = scio.read(os.path.join(args.direct_dir,'pol01i.scio.bz2'))
    with np.load(args.bbfilepath) as npz:
        bbpol01 = np.ma.MaskedArray(npz['data'],npz['mask'])
        # bbpol01 = np.ma.MaskedArray(npz['data'],npz['mask'])
        # bbpol01 = np.ma.MaskedArray(npz['data'],npz['mask'])
        channels = npz['channels'].copy()

    dat2 = (pol01r[:,channels]+1J*pol01i[:,channels])/args.acclen/2**7
    fig,ax=plt.subplots(3,2)
    fig.set_size_inches(10,12)

    fig,ax=plt.subplots(3,2)
fig.set_size_inches(10,12)

m=np.mean(pr[:,channels])
s=np.std(pr[:,channels])
ax[0][0].set_title('Direct real')
im1=ax[0][0].imshow(pr[:,channels],aspect='auto',vmin=m-2*s,vmax=m+2*s)
plt.colorbar(im1,ax=ax[0][0])

ax[0][1].set_title('Baseband real')
im3=ax[0][1].imshow(r,aspect='auto',vmin=m-2*s,vmax=m+2*s)
plt.colorbar(im3,ax=ax[0][1])

m=np.mean(pi[:,channels])
s=np.std(pi[:,channels])
ax[1][0].set_title('Direct imag')
im2=ax[1][0].imshow(pi[:,channels],aspect='auto',vmin=m-2*s,vmax=m+2*s)
plt.colorbar(im2,ax=ax[1][0])

ax[1][1].set_title('Baseband imag')
im4=ax[1][1].imshow(im,aspect='auto',vmin=m-2*s,vmax=m+2*s)
plt.colorbar(im4,ax=ax[1][1])

m=np.mean(res_real)
s=np.std(res_real)
ax[2][0].set_title('Real residuals')
img=ax[2][0].imshow(res_real,aspect='auto',vmin=-0.005,vmax=0.005)
plt.colorbar(img,ax=ax[2][0])

m=np.mean(res_imag)
s=np.std(res_imag)
ax[2][1].set_title('Imag residuals')
img=ax[2][1].imshow(res_imag,aspect='auto',vmin=-0.005,vmax=0.005)
plt.colorbar(img,ax=ax[2][1])


plt.tight_layout()

    tstamp = os.path.realpath(args.direct_dir).split('/')[-1]
    plt.savefig(os.path.join(args.outdir,f'pol01_{tstamp}_comparison.png'))