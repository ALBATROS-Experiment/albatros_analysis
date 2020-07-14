
##it is time to leave the caves of python 2 and write this in python3 

import numpy as np
import scipy.linalg as la


def sinc_window(ntap, lblock):
    """Sinc window function.
    
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
        
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    coeff_length = np.pi * ntap
    coeff_num_samples = ntap * lblock
    
    # Sampling locations of sinc function
    X = np.arange(-coeff_length / 2.0, coeff_length / 2.0,
                  coeff_length / coeff_num_samples)
    
    # np.sinc function is sin(pi*x)/pi*x, not sin(x)/x, so use X/pi
    return np.sinc(X / np.pi)


def sinc_hanning(ntap, lblock):
    """Hanning-sinc window function.
    
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
        
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """
    
    return sinc_window(ntap, lblock) * np.hanning(ntap * lblock)

def sinc_hamming(ntap, lblock):
    """Hamming-sinc window function.
    Parameters
    ----------
    ntaps : integer
        Number of taps.
    lblock: integer
        Length of block.
    Returns
    -------
    window : np.ndarray[ntaps * lblock]
    """

    return sinc_window(ntap, lblock) * np.hamming(ntap * lblock)


def pfb(timestream, nfreq, ntap=4, window=sinc_hamming):
    """Perform the CHIME PFB on a timestream.
    
    Parameters
    ----------
    timestream : np.ndarray
        Timestream to process
    nfreq : int
        Number of frequencies we want out (probably should be odd
        number because of Nyquist)
    ntaps : int
        Number of taps.

    Returns
    -------
    pfb : np.ndarray[:, nfreq]
        Array of PFB frequencies.
    """
    
    # Number of samples in a sub block
    lblock = 2 * (nfreq - 1)
    
    # Number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    
    # Initialise array for spectrum
    # if np.abs(int(nblock) - nblock) > 0.0001:
    #     print("Nblock non integer")
    #     print(nblock)
    
    nblock = int(nblock)        
    spec = np.zeros((nblock, nfreq), dtype=np.complex128)
    
    # Window function
    w = window(ntap, lblock)
    
    # Iterate over blocks and perform the PFB
    for bi in range(nblock):
        # Cut out the correct timestream section
        ts_sec = timestream[(bi*lblock):((bi+ntap)*lblock)].copy()
        
        # Perform a real FFT (with applied window function)
        ft = np.fft.rfft(ts_sec * w)
        
        # Choose every n-th frequency
        spec[bi] = ft[::ntap]
        
    return spec




# Routine wrapping Lapack dgbmv
def band_mv(A, kl, ku, n, m, x, trans = False):
    
    #y = np.zeros(n if trans else m, dtype=np.float64)
    lda = kl + ku + 1

    if lda != A.shape[0]:
        print(lda)
        print(A.shape)
        raise Exception('A does not match the number of diagonals specified.')
    if trans is True:
        T = 1
        add = n -len(x)
        if add > 0:
            #print("n added:", add)
            x = np.append(x,np.zeros(add + 1))
    elif trans is False:
        T =0
        add = m -len(x)
        if add > 0:
            #print("m added:", add)
            x = np.append(x,np.zeros(add + 1))
    

    yout = la.blas.dgbmv(m, n , kl, ku, 1.0, A, x, offx=0, incx=1, trans= T)

    
    return yout


def inverse_pfb(ts_pfb, ntap, window=sinc_hamming, no_nyquist=False):
    """Invert the CHIME PFB timestream.
    Parameters
    ----------
    ts_pfb : np.ndarray[nsamp, nfreq]
        The PFB timestream.
    ntap : integer
        The number of number of blocks combined into the final timestream.
    window : function (ntap, lblock) -> np.ndarray[lblock * ntap]
        The window function to apply to each block.
    no_nyquist : boolean, optional
        If True, we are missing the Nyquist frequency (i.e. CHIME PFB), and we
        should add it back in (with zero amplitude).
    """

    # If we are missing the Nyquist freq (default for CHIME), add it back in
    if no_nyquist:
        new_shape = ts_pfb.shape[:-1] + (ts_pfb.shape[-1] + 1,)
        pts2 = np.zeros(new_shape, dtype=np.float64)
        pts2[..., :-1] = ts_pfb
        ts_pfb = pts2

    # Inverse fourier transform to get the pseudo-timestream
    pseudo_ts = np.fft.irfft(ts_pfb, axis=-1)

    # Transpose timestream
    pseudo_ts = pseudo_ts.T.copy()

    # Pull out the number of blocks and their length
    lblock, nblock = pseudo_ts.shape
    ntsblock = nblock + ntap - 1

    # Coefficients for the P matrix
    # Create the window array.
    coeff_P = window(ntap, lblock).reshape(ntap, lblock)

    # Coefficients for the PP^T matrix
    coeff_PPT = np.array([(coeff_P[:, np.newaxis, :] *
                           coeff_P[np.newaxis, :, :])
                          .diagonal(offset=k).sum(axis=-1)
                          for k in range(ntap)])

    rec_ts = np.zeros((lblock, ntsblock), dtype=np.float64)

    for i_off in range(lblock):

        # Create band matrix representation of P
        band_P = np.zeros((ntap, ntsblock), dtype=np.float64)
        band_P[:] = coeff_P[::-1, i_off, np.newaxis]

        # Create band matrix representation of PP^T (symmetric)
        band_PPT = np.zeros((ntap, nblock), dtype=np.float64)
        band_PPT[:] = coeff_PPT[::-1, i_off, np.newaxis]

        # Solve for intermediate vector
        yh = la.solveh_banded(band_PPT, pseudo_ts[i_off])

        # Project into timestream estimate
        rec_ts[i_off] = band_mv(band_P, 0, 3, ntsblock, nblock, yh, trans=True)

    # Transpose timestream back
    rec_ts = rec_ts.T.copy()

    return rec_ts
