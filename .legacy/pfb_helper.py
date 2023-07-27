
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


def get_pfb_mat_sinc(nchan,ntap,window=sinc_hanning):
    x=np.arange(-(nchan-1)*ntap,(nchan-1)*ntap)
    x=x/(nchan-1)/2
    #mysinc2=np.sinc(x)
    mysinc=window(ntap,2*(nchan-1))
    #print(np.std(mysinc-mysinc2))
    mymat=np.zeros([nchan,len(mysinc)],dtype='complex')
    for i in range(nchan):
        mymat[i,:]=mysinc*np.exp(-2J*np.pi*x*i)
    return mymat


def apply_pfb_filter_patches(mypfb,patches):
#def filter_pfb_patches(mypfb,patches):
    """Assume the filtering is stationary and so apply it with a convolution.  Each patch is the filter
    kernel for a single frequency, with highest value assumed in the middle."""
    mypfb_out=0*mypfb
    mypfb_ft=np.fft.fft2(mypfb)
    nchan=mypfb.shape[1]
    nblock=patches[0].shape[0]
    assert(patches[0].shape[1]==nchan)
    for i in range(nchan):
        tmp=0*mypfb_out
        patch=patches[i]
        patch=np.roll(patch,-i,1)
        tmp[:nblock//2,:]=patch[nblock//2:,:]
        tmp[-nblock//2:,:]=patch[:nblock//2,:]
        #print(np.sum(np.abs(tmp)),np.sum(np.abs(patch)))
        tmpft=np.fft.fft2(tmp)
        if True:
            tmpft=tmpft*mypfb_ft
            tmpft=np.fft.ifft(tmpft,axis=1)
            mypfb_out[:,i]=np.fft.ifft(tmpft[:,i])
        else:
            myconv=np.fft.ifft2((tmpft)*mypfb_ft)
            mypfb_out[:,i]=myconv[:,i]
    return mypfb_out

def make_large_pfb_mat(nchan,ntap,nblock,window=sinc_hamming):
    block=get_pfb_mat_sinc(nchan,ntap,window=window)
    shift=2*(nchan-1)
    nx=nblock*block.shape[0]
    ny=shift*(nblock-1)+block.shape[1]
    mat=np.zeros([nx,ny],dtype='complex')
    
    for i in range(nblock):
        ix=i*nchan
        iy=i*shift
        mat[ix:ix+nchan,iy:iy+block.shape[1]]=block
    return mat

def get_pfb_filter_mat(nchan,ntap,nblock,window=sinc_hamming,frac=0.85):
    mat=make_large_pfb_mat(nchan,ntap,nblock,window=window)
    u,s,v=np.linalg.svd(mat)
    ss=s[np.int(frac*len(s))]
    s_filt=s/ss
    s_filt[s_filt>1]=1
    s_filt=s_filt**2
    filt_mat=np.conj(u)@(np.diag(s_filt)@(u.T))
    return filt_mat
    

def make_conv_patches(nchan,nblock,mat,offset=0):
    patches=[0]*nchan
    if False:
        for i in range(nchan):
            for j in range(nblock):
                ind=j*nchan
                patch=np.reshape(mat[ind+i,:],[nblock,nchan]).copy()
                patches[i]=patches[i]+patch
            patches[i]=patches[i]/nblock
    else:
        ind=(nblock//2+offset)*nchan
        patches=[0]*nchan
        for i in range(nchan):
            patch=np.reshape(mat[ind+i,:],[nblock,nchan]).copy()
            if not(offset==0):
                patch=np.roll(patch,-offset,axis=0)
            patches[i]=patch
    return patches


def filter_pfb_patches(mypfb,patches=None,ntap=4,nblock=16,window=sinc_hamming,frac=0.85,return_patches=False):
    if patches is None:
        nchan=mypfb.shape[1]
        filt_mat=get_pfb_filter_mat(nchan,ntap,nblock,window=window,frac=frac)
        patches=make_conv_patches(nchan,nblock,filt_mat)

    mypfb_filt=apply_pfb_filter_patches(mypfb,patches)
    if return_patches:
        return mypfb_filt,patches
    else:
        return mypfb_filt




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


# def pfb_timestream_fullmatrix(ntime, nfreq, ntap=4, window=sinc_hanning):
    
#     # Number of samples in a sub-block
#     lblock = 2*(nfreq - 1)
    
#     # Number of blocks in timestream
#     nblocks = ntime / lblock
    
#     # Number of blocks in PFB
#     npfb = nblocks - ntap + 1
    
#     # Initialise matrix
#     mat = np.zeros((npfb, lblock, nblocks, lblock))
    
#     # Window function
#     w = window(ntap, lblock)
    
#     # Iterate over PFB blocks setting the elements
#     for bi in range(npfb):
#         for si in range(lblock):
#             for ai in range(ntap):
#                 mat[bi, si, bi+ai, si] = w[si + ai * lblock]
    
#     return mat


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
