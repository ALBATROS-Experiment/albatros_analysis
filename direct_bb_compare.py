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
        channels = npz['channels'].copy()

    dat2 = (pol01r[:,channels]+1J*pol01i[:,channels])/args.acclen/2**7
    fig,ax=plt.subplots(3,2)
    fig.set_size_inches(10,12)

    ax[0][0].set_title('direct pol01 mag')
    img=ax[0][0].imshow(np.log10(np.abs(dat2)),aspect='auto',vmin=-1.4,vmax=-0.5)
    plt.colorbar(img,ax=ax[0][0])
    ax[0][1].set_title('direct pol01 phase')
    img=ax[0][1].imshow(np.angle(dat2),aspect='auto',cmap='RdBu',vmin=-np.pi,vmax=np.pi)
    plt.colorbar(img,ax=ax[0][1])

    ax[1][0].set_title('baseband pol01 mag')
    img=ax[1][0].imshow(np.log10(np.abs(bbpol01)),aspect='auto',vmin=-1.4,vmax=-0.5)
    plt.colorbar(img,ax=ax[1][0])
    ax[1][1].set_title('baseband pol01 phase')
    img=ax[1][1].imshow(np.angle(bbpol01),aspect='auto',cmap='RdBu',vmin=-np.pi,vmax=np.pi)
    plt.colorbar(img,ax=ax[1][1])

    err=np.real(dat2)-np.real(bbpol01)
    m=np.mean(err)
    std=np.std(err)
    ax[2][0].set_title('real residuals')
    img=ax[2][0].imshow(err,aspect='auto',vmin=m-2*std,vmax=m+2*std)
    plt.colorbar(img,ax=ax[2][0])
                
    err=np.imag(dat2)-np.imag(bbpol01)
    m=np.mean(err)
    std=np.std(err)
    ax[2][1].set_title('imag residuals')
    img=ax[2][1].imshow(err,aspect='auto',vmin=m-2*std,vmax=m+2*std)
    plt.colorbar(img,ax=ax[2][1])

    tstamp = os.path.realpath(args.direct_dir).split('/')[-1]
    plt.savefig(os.path.join(args.outdir,f'pol01_{tstamp}_comparison.png'))