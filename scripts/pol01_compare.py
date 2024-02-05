"""This script is meant to compare pol01 real and imaginary parts produced from baseband averaging and corresopnding direct file for quick sanity check. 

For 4 bit data, it plots magnitude, phase, and real/imag residuals. For 1 bit data, it plots real/imag parts and real/imag residuals. 

Accumulation length is inferred from file name, so don't mess with default file names. 

Example usage:

`pol01_compare.py ~/path_to_baseband_npz/pol01_1bit_1627528594_393216_560_0_None.npz ~/path_to_direct_folder/16275/1627528594/`
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scio import scio  # previously: from utils import scio

# The name of the package containing scio module
# & class is actually called pbio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bbfilepath", type=str, help="Path to npz file. E.g. ~/pol01_4bit.npz"
    )
    parser.add_argument(
        "direct_dir",
        type=str,
        help="Path to direct data directory of relevant timestamp. E.g. ~/16272/1627202094/",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        type=str,
        default="./",
        help="Output directory for data and plots",
    )
    args = parser.parse_args()

    pol01r = scio.read(os.path.join(args.direct_dir, "pol01r.scio.bz2"))
    pol01i = scio.read(os.path.join(args.direct_dir, "pol01i.scio.bz2"))
    with np.load(args.bbfilepath) as npz:
        bbpol01 = np.ma.MaskedArray(npz["datap01"], npz["maskp01"])
        # bbpol01 = np.ma.MaskedArray(npz['data'],npz['mask'])
        # bbpol01 = np.ma.MaskedArray(npz['data'],npz['mask'])
        channels = npz["chans"].copy()

    bitmode = int(args.bbfilepath.split("/")[-1].split("_")[1][0])
    print("Inferred bit mode: ", bitmode)
    acclen = int(args.bbfilepath.split("/")[-1].split("_")[3])
    print("Inferred acclen: ", acclen)

    fig, ax = plt.subplots(3, 2)
    fig.set_size_inches(10, 12)
    if bitmode == 1:
        pol00 = scio.read(os.path.join(args.direct_dir, "pol00.scio.bz2"))
        pol11 = scio.read(os.path.join(args.direct_dir, "pol11.scio.bz2"))
        norm = np.sqrt(pol11) * np.sqrt(pol00)
        pr = pol01r / norm
        pi = pol01i / norm
        r = np.real(bbpol01)
        imag = np.imag(bbpol01)
        vmin = -0.005
        vmax = 0.005

        img = ax[0][0].imshow(pr[:, channels], aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(img, ax=ax[0][0])

        ax[0][1].set_title("Baseband real")
        img = ax[0][1].imshow(r, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(img, ax=ax[0][1])

        ax[1][0].set_title("Direct imag")
        img = ax[1][0].imshow(pi[:, channels], aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(img, ax=ax[1][0])

        ax[1][1].set_title("Baseband imag")
        img = ax[1][1].imshow(imag, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(img, ax=ax[1][1])

        res_real = r - pr[:, channels]
        ax[2][0].set_title("Real residuals")
        img = ax[2][0].imshow(res_real, aspect="auto", vmin=-0.005, vmax=0.005)
        plt.colorbar(img, ax=ax[2][0])

        res_imag = imag - pi[:, channels]
        ax[2][1].set_title("Imag residuals")
        img = ax[2][1].imshow(res_imag, aspect="auto", vmin=-0.005, vmax=0.005)
        plt.colorbar(img, ax=ax[2][1])

        plt.tight_layout()
    elif bitmode == 4:
        dat2 = (pol01r[:, channels] + 1j * pol01i[:, channels]) / acclen / 2**7
        ax[0][0].set_title("Direct pol01 mag")
        img = ax[0][0].imshow(
            np.log10(np.abs(dat2)), aspect="auto", vmin=-1.4, vmax=-0.5
        )
        plt.colorbar(img, ax=ax[0][0])
        ax[0][1].set_title("Direct pol01 phase")
        img = ax[0][1].imshow(
            np.angle(dat2), aspect="auto", cmap="RdBu", vmin=-np.pi, vmax=np.pi
        )
        plt.colorbar(img, ax=ax[0][1])

        ax[1][0].set_title("Baseband pol01 mag")
        img = ax[1][0].imshow(
            np.log10(np.abs(bbpol01)), aspect="auto", vmin=-1.4, vmax=-0.5
        )
        plt.colorbar(img, ax=ax[1][0])
        ax[1][1].set_title("Baseband pol01 phase")
        img = ax[1][1].imshow(
            np.angle(bbpol01), aspect="auto", cmap="RdBu", vmin=-np.pi, vmax=np.pi
        )
        plt.colorbar(img, ax=ax[1][1])

        res_real = np.real(dat2) - np.real(bbpol01)
        m = np.mean(res_real)
        std = np.std(res_real)
        ax[2][0].set_title("real residuals")
        img = ax[2][0].imshow(
            res_real, aspect="auto", vmin=m - 2 * std, vmax=m + 2 * std
        )
        plt.colorbar(img, ax=ax[2][0])

        res_imag = np.imag(dat2) - np.imag(bbpol01)
        m = np.mean(res_imag)
        std = np.std(res_imag)
        ax[2][1].set_title("imag residuals")
        img = ax[2][1].imshow(
            res_imag, aspect="auto", vmin=m - 2 * std, vmax=m + 2 * std
        )
        plt.colorbar(img, ax=ax[2][1])
        plt.tight_layout()

    tstamp = os.path.realpath(args.direct_dir).split("/")[-1]
    fpath = os.path.join(args.outdir, f"pol01_{tstamp}_comparison.png")
    plt.savefig(fpath)
    print(fpath)
