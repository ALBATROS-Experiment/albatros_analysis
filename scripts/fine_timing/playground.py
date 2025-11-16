from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy.stats import median_abs_deviation as mad
import cg

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

if __name__ == '__main__':

    with np.load("./data/spectra_1830_1840_4096_24000_0.05_clock_1overf_5e-9.npz") as f:
        spec1=f['spectra1']
        spec2=f['spectra2']
        delays=f['delays']
    with np.load("/home/mohan/Downloads/raw_16_0.08.npz") as f:
        new_spec1 = f['spec1']
        new_spec2 = f['spec2']
        new_channels = f['channels']
    osamp=16
    # Global font size settings
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 22
    })
    # fig=plt.gcf()
    # fig.set_size_inches(10,4)
    # plt.title(r"Simulated 1/f drift [25 $\mu$Hz to 250 MHz] ")
    # plt.plot(np.arange(len(delays))*16e-6, -delays*4, lw=2)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Drift (ns)")
    # plt.tight_layout()
    # # plt.savefig("sim_drift.png",dpi=300)
    # plt.show()
    # sys.exit()
    avglen=3200
    # auto = np.abs(spec1[:,4:5])**2-np.abs(spec1[:,0:1])**2
    # plt.clf()
    # fig=plt.gcf()
    # fig.set_size_inches(10,6)
    # auto = np.abs(spec1[:,:])**2
    # auto_avg1 = average_rows(auto,nblock=10000)
    # df=0.061
    # myext = [1829.5*df, 1839.5*df, spec1.shape[0]*16e-6, 0]
    # plt.title(f"METEOR M2-4\nBW = 100 kHz, S/N -6 dB, int. time {10000*16e-6:4.2f}s")
    # im2=plt.imshow(np.log10(auto_avg1), aspect='auto',interpolation='none',extent=myext)
    # plt.xlabel("Freq (MHz)")
    # plt.ylabel("Time (s)")
    # ax=plt.gca()
    # plt.axvline(1834.1888*df, ls='dashed',c='black')
    # cbar = fig.colorbar(im2, ax=ax, orientation="vertical")
    # cbar.ax.tick_params(labelsize=14)
    # cbar.set_label("log(Power)", fontsize=16)
    # plt.savefig("/scratch/thomasb/mohan/meteor.png",dpi=300)
    # # auto = np.abs(spec2[:,4:5])**2-np.abs(spec2[:,0:1])**2
    # # auto_avg2 = average_rows(auto,nblock=avglen)
    
    # xc=spec1[:,:]*np.conj(spec2[:,:])
    # # xc_avg=average_rows(xc,nblock=avglen)
    # xc_true=xc[:,:]*np.exp(2j*np.pi*np.arange(1830,1840)*delays[:,None]/4096)
    # # xc_true=xc[:,4:5]*np.exp(2j*np.pi*1834.1888*delays/4096).reshape(-1,1)
    # # xc_true_avg = average_rows(xc_true,nblock=avglen)
    # xc_true_avg = average_rows(xc_true,nblock=avglen)
    # xc_avg = average_rows(xc,nblock=avglen)
    # plt.plot(auto_avg1[:,0],label='autos1')
    # # plt.plot(auto_avg2[:,0],label='autos2')
    # plt.plot(np.abs(xc_true_avg[:,4]),label='cross corrected true')
    # plt.plot(np.real(xc_true_avg[:,4]),label='cross corrected true real')
    # plt.plot(np.abs(xc_avg[:,4]),label='cross uncorrected true')
    # # plt.plot(np.abs(average_rows(spec1*np.conj(spec2),nblock=avglen))[:,4])
    # plt.legend()
    # plt.show()
    
    # xc_avg = xc_true_avg

    # noiser = np.std(xc_avg[:,[0,1,2,6,7,8,9]].real)
    # noiseim = np.std(xc_avg[:,[0,1,2,6,7,8,9]].imag)
    
    # # noiser = np.std(xc_avg[:,4].real)
    # # noiseim = np.std(xc_avg[:,4].imag)
    # signalr = np.mean(xc_avg[:,4].real)
    # signalim = np.mean(xc_avg[:,4].imag)
    # print("snr:",signalr/noiser/np.sqrt(avglen), signalim/noiseim)
    # ph_noise = np.std(np.angle(xc_avg[:,4]))
    # print(f"phase noise : {ph_noise:5.3f} rad")
    
    # sigma_tau = np.sqrt(12) * ph_noise * np.sqrt(61e3) / np.sqrt(avglen) / (100e3)**1.5
    # sigma_tau = phase2timenoise(ph_noise, 2, 61e3)

    # print(f"expected slope noise : {sigma_tau/4e-9:5.3e} samp")
    # # sys.exit()

    #iterate over several blocks and fit a line
    for mult in range(5):
        blocksize=avglen
        st=0
        nblocks=spec1.shape[0]//blocksize
        # nblocks=20
        print("Num blocks are", nblocks)
        fit_delta=np.zeros(nblocks,dtype='float64')
        actual_delta = np.zeros(nblocks,dtype='float64')
        r_val = np.zeros(nblocks,dtype='float64')
        for i in range(nblocks):
            #fft and get a coarse guess
            ix=st+i*blocksize
            # print("taking data from ", ix, ix+blocksize)
            y1=spec1[ix:ix+blocksize,4]
            y2=spec2[ix:ix+blocksize,4]
            xc_small = y1*np.conj(y2)
            xc_fft = np.fft.fftshift(np.abs(np.fft.fft(xc_small)))
            mm=np.argmax(xc_fft)
            M=len(xc_fft)
            # print(M)
            expected_delta = -(mm-M/2)/M /1834
            if np.abs(expected_delta) < 1e-15: expected_delta = 1e-15
            # print("Starting point", expected_delta)
            # print(f"Block {i} Expected delta={expected_delta}")
            alpha=lmsolver(xc_small,expected_delta,1834.1888,lamda=10,ftol=1e-8,xtol=1e-8,niter=100)
            fit_delta[i]=alpha
            # plt.title(f"Block {i}")
            # plt.plot(delays[ix:ix+blocksize])
            # plt.show()
            xx=np.arange(ix,ix+blocksize)
            yy=delays[ix:ix+blocksize]
            yymean=np.mean(yy)
            m,c=np.polyfit(xx,yy,1)
            yypred=np.polyval([m,c],xx)
            SSreg=np.sum((yypred-yymean)**2)
            SStot=np.sum((yy-yymean)**2)
            # print(f"SSreg, SStot for {i}", SSreg,SStot)
            r_val[i] = np.sqrt(SSreg/SStot)
            # m,c=np.polyfit(np.arange(0,en-st),delays[st:en]-delays[st],1)
            # print(f"block {i} slope", m/4096, "pred slope", alpha)
            actual_delta[i] = m/4096

        
        # xc=new_spec1[:,:]*np.conj(new_spec2[:,:])
        # xc_true=xc[:,:]*np.exp(2j*np.pi*new_channels*delays[:osamp*new_spec1.shape[0]:osamp,None]/4096/osamp)
        # xc_true_avg = average_rows(xc_true,nblock=1000)
        # xc_avg = average_rows(xc,nblock=1000)

        # # plt.plot(auto_avg1[:,0],label='autos1')
        # plt.plot(np.abs(xc_true_avg[:,4]),label='cross corrected true')
        # plt.plot(np.real(xc_true_avg[:,4]),label='cross corrected true real')
        # plt.plot(np.abs(xc_avg[:,4]),label='cross uncorrected true')
        # plt.legend()
        # plt.show()
        # sys.exit()
        
        st=0
        blocksize = avglen // osamp
        
        print("new channels are", new_channels/osamp)
        nblocks=new_spec1.shape[0]//blocksize - 5
        print("new blocksize=",blocksize)
        n = np.arange(0,blocksize)
        xc_avg_corrected = np.zeros((nblocks,len(new_channels)),dtype='complex128')
        xc_avg_uncorrected = np.zeros((nblocks,len(new_channels)),dtype='complex128')
        slopes = np.zeros(nblocks,dtype='float64')
        ph_noises = np.zeros(nblocks,dtype='float64')
        phases = np.zeros((nblocks,len(new_channels)),dtype='float64')
        for i in range(nblocks):
            #fft and get a coarse guess
            ix=st+i*blocksize
            y1=new_spec1[ix:ix+blocksize,:]
            y2=new_spec2[ix:ix+blocksize,:]
            xc_small = y1*np.conj(y2)
            # xc_corrected1 = xc_small * np.exp(2j*np.pi*1834*n*fit_delta[i])
            xc_corrected1 = xc_small * np.exp(2j*np.pi*new_channels*n[:,None]*fit_delta[i])
            # print(xc_corrected1.shape)
            # xc_corrected2 = xc_small * np.exp(2j*np.pi*1834*n*actual_delta[i])
            # print(np.abs(np.mean(xc_corrected1))>np.abs(np.mean(xc_corrected2)))
            # ph_noises[i] = np.std(np.angle(xc_corrected1)[:,0],axis=0) # phase noise is around 1.8 rad as expected from low S/N
            xc_avg_corrected[i,:] = np.mean(xc_corrected1,axis=0)
            xc_avg_uncorrected[i,:] = np.mean(xc_small,axis=0)
            ph = np.unwrap(np.angle(xc_avg_corrected[i,:]))
            phases[i, :] = ph
            slope, const = np.polyfit(2*np.pi*new_channels/4096/osamp,ph,1)
            slopes[i] = slope
            ph_noises[i] = np.std(ph-(2*slope*np.pi*new_channels/4096/osamp+const))
        
        #TRY DOING CONJUGATE GRADIENT HERE
        adev=1e-8
        noise = np.median(ph_noises)
        ddsigma = avglen*4096*adev*np.sqrt(2)
        print("ddsigma", ddsigma, "phnoise estimate", noise)
        Ninv = 1/(np.ones(len(slopes))*np.median(ph_noises)**2) #start with a constant noise for all time blocks
        x = 2*np.pi*new_channels/4096/osamp
        params = np.zeros(2*len(slopes),dtype='float64')
        params[:len(slopes)]=1
        params[len(slopes):]=1
        cg.cg_linear_model_adev(params, x, np.ravel(phases), ddsigma, noise, eps=1e-10, niter=100) #verified that without a prior the solution lines up with polyfit
        # plt.plot(slopes)
        # plt.plot(params[:len(slopes)])
        # plt.show()

        # plt.plot(np.abs(xc_avg[:,4]))
        # plt.plot(np.abs(xc_avg_uncorrected[:,0]),label='uncorrected')
        # plt.plot(np.abs(xc_avg_corrected[:,0]),label='corrected per-block')
        # # plt.axhline(np.mean(auto_avg1[:,0]),label='autos',c='red', lw=3, ls='dashed')
        # plt.legend()
        # # plt.plot(np.abs(average_rows(spec1*np.conj(spec2),nblock=avglen))[:,4])
        # plt.plot(np.abs(xc_avg_uncorrected[:,1]),label='uncorrected')
        # plt.plot(np.abs(xc_avg_corrected[:,1]),label='corrected per-block')
        # # plt.axhline(np.mean(auto_avg1[:,0]),label='autos',c='red', lw=3, ls='dashed')
        # plt.legend()
        # # ax=plt.gca()
        # # ax2=plt.twinx()
        # # ax2.plot(r_val,label='R value',c='red',alpha=0.5)
        # # ax2.axhline(0.5,ls='dashed',c='black',lw=3,label='Drift ISNT too linear below')
        # plt.show()
        times=np.arange(len(slopes))*16e-6*avglen
        fig=plt.gcf()
        fig.set_size_inches(10,4)
        plt.clf()
        plt.title(rf"Int. time {avglen*16e-6:4.2f}s, channel $\Delta \nu = {61/osamp:4.2f}$ kHz")
        plt.plot(times, slopes*4, label='Fitted drift')
        plt.plot(times, params[:len(slopes)]*4, label='Fit w/ ADEV prior')
        maderr = mad(slopes+delays[::blocksize*osamp][:len(slopes)])*4
        maderr2 = mad(params[:len(slopes)]+delays[::blocksize*osamp][:len(slopes)])*4
        stddev = np.std(slopes+delays[::blocksize*osamp][:len(slopes)])*4
        print(maderr, maderr2, stddev)
        err = maderr
        plt.plot(times, -delays[::blocksize*osamp][:len(slopes)]*4, label='True drift',ls='dashed',lw='2',c='purple')
        plt.text(0.5, 0.1, f"MAD {int(err)} ns (no prior) vs {int(maderr2)} (w/ prior) ns", transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right')
        # plt.ylim(-0.5,0.5)
        plt.ylim(-1000,2000)
        plt.xlabel("Time (s)")
        plt.ylabel("Drift (ns)")
        plt.legend(loc=4)
        plt.tight_layout()
        plt.savefig(f"./images/drift_fitted{mult}_adev.png",dpi=300)
        # plt.show()
        print("Done mult", mult)
        avglen*=2
    sys.exit()

    channum = 5
    ph = np.angle(xc_avg_corrected[:,channum])
    phdiff = np.diff(ph)
    howmany = np.where((np.abs(phdiff)<3.5)&(np.abs(phdiff)>np.pi))[0]
    print("Howmany in risky zone:", len(howmany))
    ph0 = np.unwrap(ph)-ph[0]
    plt.plot(ph0)
    plt.show()
    # plt.title("delay residual wrt block 0")
    # plt.plot(np.angle(np.exp(-1j*ph0)),marker='*')
    # plt.plot(np.angle(np.exp(2j*np.pi*new_channels[channum]*delays[::blocksize]/4096)),marker='o')
    # for h in howmany:
    #     plt.axvline(h,c='black')
    # ax=plt.gca()
    # ax2=ax.twinx()
    unwrapped_ph_noise = 2*np.pi*new_channels[channum]*delays[::blocksize*osamp][:len(slopes)]/4096/osamp + ph0
    print("unwraped phase noise", np.std(unwrapped_ph_noise))
    # sigma_tau = phase2timenoise(unwrapped_ph_noise, 2, 61e3)
    # print(f"expected slope noise : {sigma_tau/4e-9:5.3e} samp")
    plt.plot(unwrapped_ph_noise,c='red',marker='*')
    plt.show()

    plt.plot(2*np.pi*new_channels[channum]*delays[::blocksize*osamp][:len(slopes)]/4096/osamp)
    plt.plot(-ph0)
    # plt.plot(r_val)
    # plt.plot(delays[::5000])

    # plt.plot(np.abs(xc_avg_corrected),label='corrected per-block')
    # plt.axhline(np.abs(np.mean(xc_avg_corrected2)),label='global mean')
    plt.show()
    # plt.plot(np.unwrap(np.angle(xc_avg_corrected)))
    # plt.plot()
    # plt.show()


