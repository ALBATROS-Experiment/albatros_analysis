from os import path
import sys
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
from albatros_analysis.src.utils import orbcomm_utils as outils
import numpy as np
import cupy as cp
import time

def phase2timenoise(sigma_phase, nchan, chanwidth):
    return np.sqrt(12) * sigma_phase / chanwidth / np.sqrt(nchan * (nchan**2-1))

def average_rows(x,nblock=100):
    ''' 
    Averages some data array x into blocks of length nblock. 
    Remaining spectra that are left over are ignored.
    
    Parameters
    ----------
    x: numpy array 
        Shape (nspectra, nchans) over which we want to average the spectra into blocks

    nblock: int
        The length of the blocks we want to average x into

    Returns
    -------
    y: numpy array
        Shape (nspectra//nblock, nchans). The averaged block.
    '''
    if x.ndim == 1:
        x = x[:, None]
    nr=x.shape[0]//nblock
    print(x.shape[0],nr)
    nc=x.shape[1]
    y=np.zeros((nr,nc),dtype=x.dtype)
    for i in range(nr):
        y[i,:]=np.mean(x[i*nblock:(i+1)*nblock],axis=0)
    return y



def get_adev(x,tau,stidx=0,endidx=None, T_SPECTRA = 4096/250e6):
    ''' 
    Computes Allan Deviation for array of data x
    '''


    delta=int(tau/T_SPECTRA) #amount of time in units of spectra
    sl=slice(stidx,endidx,delta) #picks indices in interval of delta (integer value)
    samps=x[sl] #samples corresponding to those indices
    #get Allan deviation
    adev=T_SPECTRA*np.sqrt(np.mean((samps[2:]-2*samps[1:-1]+samps[:-2])**2)/(2*tau**2))
    return adev


def newton(n,xc,alpha,chan,scale=1):
    ''' 
    single iteration of newton-gauss algorithm to optimize for alpha

    Parameters
    ----------
    n: numpy array
        just the spectrum number index, shape (len(xc),) from zero to len(xc)-1.
        helpful because tells your phase where it's at in the beamform

    x: numpy array
        Data array of complex correlated data

    alpha: float
        Initial guess of alpha

    chan: float (or int?)
        Channel index

    scale: int
        fixed learning rate, like for gradient descent.
    
    Returns
    -------
    alpha2: float
        Improves fitting parameter

    '''
    #set basic parameters
    c=2*np.pi*chan
    N=len(xc)
    #using initial alpha, calculate all required quantities for update
    xc_phased = xc*np.exp(1j*c*n*alpha)
    S0conj=np.conj(np.mean(xc_phased))
    S1=np.mean(xc_phased*n*c)
    S2=np.mean(xc_phased*n**2*c**2)
    df=-np.imag(S0conj*S1)
    ddf=-np.real(S0conj*S2) + np.abs(S1)**2
    #apply update to alpha
    alpha2 = alpha - scale*df/ddf
    print(f"old alpha {alpha:5.3e}", 
          f"df {df:5.3e}", 
          f"ddf {ddf:5.3e}", 
          f"step size {df/ddf:5.3e}" 
          f"new alpha {alpha2:5.3e}")
    return alpha2


def lmsolver(xc,alpha,chan,lamda=16,xtol=1e-6,ftol=1e-6,niter=10,debug=False):
    '''
    For one chunk, for one channel, solve for alpha using LM algorithm.

    Parameters
    ----------
    xc: numpy array
        array shape (nspectra,), with the complex cross-correlation data

    alpha: float
        initial guess for the alpha parameter

    chan: float
        channel index that we are fitting

    lamda: float
        initial damping parameter. will get updated for each iteration

    xtol: float
        minimum relative difference in alpha tolerated to terminate iterations
        i.e. if |(a2-a)/a| < xtol, then we are safe to terminate

    ftol: float
        minimum relative difference in objective function value to terminate iterations
        i.e. if |(f2-f)/f| < ftol, then we are safe to terminate

    niter: int
        maximum number of iterations of alpha update algorithm

    Returns
    -------
    alpha: float
        fitted alpha parameter. is only returned if convergence is achieved
    '''
    c=2*np.pi*chan
    N=len(xc)
    n = np.arange(N)
    for ii in range(niter):
        if debug: print(f"------------------------------ LM iter {ii} -------------------------------")
        #set up the initial phased data and objective function.
        #this is what we get using our initial alpha_k
        xc_phased = 1e-5*xc*np.exp(1j*c*n*alpha)
        f = -np.abs(np.mean(xc_phased))**2

        #set up the conjugate versions and calculate first and second derivatives
        S0conj=np.conj(np.mean(xc_phased))
        S1=np.mean(xc_phased*n*c)
        S2=np.mean(xc_phased*n**2*c**2)
        df = np.imag(S0conj*S1)
        ddf = np.real(S0conj*S2) - np.abs(S1)**2

        #apply LM algorithm, update alpha to its next value, alpha_(k+1)
        step = - df/(ddf + lamda*np.abs(ddf))
        alpha2 = alpha + step

        #get the function value using new alpha
        xc_phased = 1e-5*xc*np.exp(1j*c*n*alpha2)
        f2 = -np.abs(np.mean(xc_phased))**2
        
        if debug: print(f"lamda: {lamda:5.3e}\n",
                        f'alpha: {alpha:5.3e}\n',
                        f'f: {f:5.3e}\n',
                        f'df: {df:5.3e}\n',
                        f'ddf: {ddf:5.3e}\n',
                        f'step: {step:5.3e}\n'
                        f'alpha2: {alpha2:5.3e}\n' 
                        f'f2: {f2:5.3e}')
            
        # if proceeding towards minimization, i.e. new alpha reduces cost
        if f2 < f: 
            if debug: print("accepting new alpha, reduced cost function")
            #get relative changes in alpha and cost function
            rel_alpha = (alpha2-alpha)/alpha
            rel_f = (f2-f)/f
            if debug: print(f"rel alpha: {np.abs(rel_alpha):5.3e}\n"
                            f"rel f: {np.abs(rel_f):5.3e}")
            #update starting alpha and reduce damping parameter
            alpha = alpha2
            lamda/=2
            #check for convergence, exit if satisfies conditions
            if np.abs(rel_alpha) < xtol or np.abs(rel_f) < ftol:
                if debug: print("converged.")
                return alpha
        #if not moving in correct direction, i.e. new alpha increases cost
        else:
            lamda*=2
    raise Exception(f"Failed to converge in {niter} iterations.")



def solver(xc,alpha,chan,scale=1,atol=1e-10,rtol=1e-6,niter=10):
    niter=niter
    success=0
    N=len(xc)
    n = np.arange(N)
    for i in range(niter):
        alpha2=newton(n,xc,alpha,chan,scale=scale)
        if np.abs(alpha2) < atol:
            print("atol hit")
            success=1
            break
        if np.abs((alpha2-alpha)/alpha) < rtol:
            print("rtol hit")
            success=1
            break
        alpha=alpha2
    if not success:
        print("Max iterations, unsuccessful.")
    return alpha2

def objective_func(xc,alpha,chan):
    c=2*np.pi*chan
    N=len(xc)
    n=np.arange(N)
    xc_phased = xc*np.exp(1j*c*n*alpha)
    return 

def objective_func(xc,alpha,chan,derivs=True):
    c=2*np.pi*chan
    N=len(xc)
    n=np.arange(N)
    xc_phased = xc*np.exp(1j*c*n*alpha)
    f=np.abs(np.mean(xc_phased))**2
    if derivs == True:
        S0conj=np.conj(np.mean(xc_phased))
        S1=np.mean(xc_phased*n*c)
        S2=np.mean(xc_phased*n**2*c**2)
        df=-np.imag(S0conj*S1)
        ddf=-np.real(S0conj*S2) + np.abs(S1)**2
        return f, df, ddf
    return f




def get_beamformed_vis_whole_pulse(files1, 
                                   files2, 
                                   idxstart1, 
                                   idxstart2,
                                   t1, 
                                   t2, 
                                   coords,
                                   sat,
                                   tle_path,
                                   chan_b_idx,
                                   acclen, 
                                   nchunks, 
                                   chanstart=0, 
                                   chanend=None,
                                   T_SPECTRA = 4096/250e6):
    
    freq = 250e6 * (1 - chan_b_idx / 4096)
    delays = np.zeros(acclen)
    niter = t2-t1 + 1
    nspec = int(niter/T_SPECTRA)
    d = outils.get_sat_delay(coords[0],
                             coords[1],
                             tle_path,
                             t1,
                             niter,
                             sat)
    delays = np.interp(np.arange(0, nspec) * T_SPECTRA, 
                                np.arange(0, niter), 
                                d)
    

    print("Starting at: ", idxstart1, "in filenum: ", files1[0], "for antenna 1")
    print("Starting at: ", idxstart2, "in filenum: ", files2[0], "for antenna 2")
    # print(files[fileidx])
    fileidx1 = 0
    fileidx2 = 0
    ant1 = bdc.BasebandFileIterator(
        files1,
        fileidx1,
        idxstart1,
        acclen,
        nchunks=nchunks,
        chanstart=chanstart,
        chanend=chanend,
    )
    ant2 = bdc.BasebandFileIterator(
        files2,
        fileidx2,
        idxstart2,
        acclen,
        nchunks=nchunks,
        chanstart=chanstart,
        chanend=chanend,
    )
    ncols = ant1.obj.chanend - ant1.obj.chanstart
    npols = 2
    polmap = {0: ["pol0", "pol0"], 1: ["pol1", "pol1"]}
    pols = np.zeros((npols, nchunks, ncols), dtype="complex64", order="c")
    rowcounts = np.empty(nchunks, dtype="int64")
    m1 = ant1.spec_num_start
    m2 = ant2.spec_num_start
    st = time.time()
    for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
        for spec in range(acclen):
            chunk2[spec, :] = chunk2[spec, :] * np.exp(2j * np.pi * freq * delays[i*acclen + spec])
    
        
        for pp in range(npols):
            xcorr, rowcount = cr.avg_xcorr_1bit_vanvleck_2ant(
                chunk1[polmap[pp][0]],
                chunk2[polmap[pp][1]],
                ncols,
                chunk1["specnums"],
                chunk2["specnums"],
                m1 + i * acclen,
                m2 + i * acclen,
            )
            if rowcount < 100:
                pols[pp, i, :] = np.nan
            else:
                pols[pp, i, :] = cr.van_vleck_correction(
                    *xcorr, rowcount
                )  # Van Vleck needs unpacked R0,R1,I0,I1
            rowcounts[i] = rowcount
        # t2=time.time()
        # print("time taken for one loop", t2-t1)
        j = ant1.spec_num_start
        # print("After a loop spec_num start at:", j, "Expected at", m1+(i+1)*acclen)
        if i % 1000 == 0:
            print(i + 1, "CHUNK READ")
    print("Time taken final:", time.time() - st)
    pols = np.ma.masked_invalid(pols)
    return pols, rowcounts, ant1.obj.channels
