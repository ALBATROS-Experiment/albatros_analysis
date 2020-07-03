import numpy as np
import pylab
import pfb_helper as pfb

def xcorr_channels(ts1, ts2, lblock=1024, skip=5):
    
    w = np.hanning(lblock)

    # Skip a number of blocks at the beginning and end, as we know that these are very noisy
    skip = 1000

    # FFT the timestreams to get freq channels
    ch1 = np.fft.rfft(ts1.reshape(-1, lblock) * w, axis=1)
    ch2 = np.fft.rfft(ts2.reshape(-1, lblock) * w, axis=1)

    # Perform the various correlations
    c_11 = (ch1 * ch1.conj())[skip:-skip].mean(axis=0)
    c_12 = (ch1 * ch2.conj())[skip:-skip].mean(axis=0)
    c_22 = (ch2 * ch2.conj())[skip:-skip].mean(axis=0)
    
    return c_11, c_22, c_12

def quantise(arr, levels, ref):
    """Quantise a timestream.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to quantise.
    levels : integer
        Number of levels in output timestream.
    ref : scalar
        The size of the level spacing (i.e. the value for the first level).
        
    Returns
    -------
    ts : np.ndarray
        Quantised array.
    """
    
    return np.maximum(np.minimum(np.rint(arr / ref), levels), -levels) * ref


# ##inverse pfb example
# ntime = 2**20

# # Generate a white noise timestream
# ts = np.random.standard_normal(ntime)

# # Perform the PFB
# spec_pfb = pfb.pfb(ts, 65)

# # Perform the inverse
# rts = pfb.inverse_pfb(spec_pfb, 4)


# pylab.imshow(spec_pfb[:300].real, interpolation='nearest', aspect='auto', cmap='RdBu')
# pylab.colorbar()
# pylab.show()


# pylab.subplot(121)
# pylab.plot((ts[1000:20000] - rts.ravel()[1000:20000]))
# pylab.title("Near start")

# pylab.subplot(122)
# pylab.plot((ts[-20000:-1000] - rts.ravel()[-20000:-1000]))
# pylab.title("Near end")
# pylab.show()


# block_res = ts.reshape(-1, 128) - rts

# pylab.title("Standard deviation of residuals")
# pylab.semilogy(block_res.std(axis=1))
# pylab.xlabel("Blocks")
# pylab.show()

# ##missing freq
# print("missing freq")
# ntime = 2**22

# # Generate a white noise timestream
# ts = np.random.standard_normal(ntime)

# qts = quantise(ts, levels=8, ref=0.5)

# c_ii, c_oo, c_oi = xcorr_channels(ts, qts, lblock=1024, skip=10)

# pylab.plot(c_oi / c_ii, label='Cross-corr')
# pylab.plot(c_oo / c_ii, label='Auto-corr', ls=':')

# pylab.legend()

# pylab.xlabel('Hi-res channel')

# pylab.ylim(0, 1.5)

# pylab.show()
# #axvline(8*30)

##bellow is last quantizaiton example 
print("quant last")
ntime = 2**22

# Generate a white noise timestream
ts = np.random.standard_normal(ntime)

# Define new window function
#f = lambda nt, lb: np.hanning(nt * lb)

# Perform the PFB
spec_pfb = pfb.pfb(ts, 65)
spec_fft = np.fft.fft(np.reshape(ts,(-1,64)))

# Quantise data
spec_q = quantise(spec_pfb, 32, 4.0)
pfb_delta = spec_pfb - spec_q

spec_fft_q = quantise(spec_fft, 8, 4.0)
fft_delta = spec_fft - spec_fft_q


# Perform the inverse
rts = pfb.inverse_pfb(spec_q, 4).ravel()
delta_rts = pfb.inverse_pfb(spec_q, 4).ravel()

print(np.shape(rts))


rts_fft = np.fft.irfft(spec_q).ravel()

some_zeros = np.zeros(4194304)

some_zeros[:len(rts_fft)] = rts_fft

rts_fft= some_zeros

print(np.shape(rts_fft))

c_ii, c_oo, c_oi = xcorr_channels(ts, rts, lblock=1024, skip=1000)

pylab.plot(c_oi / c_ii, label='Cross-corr')
pylab.plot(c_oo / c_ii, label='Auto-corr', ls=':')

tchan = 30

pylab.axvline(8*(tchan-0.5), color='k', ls='--')
pylab.legend()

pylab.xlabel('Hi-res channel')

pylab.ylim(0, 3.5)
#axvline(8*30)

pylab.show()


def bin_spec(spec,bin):
    nn=len(spec)//bin
    tmp=spec[:nn*bin]
    tmp=np.reshape(tmp,[nn,bin])
    return np.sum(tmp,axis=1)
    

x=np.ravel(delta_rts)[1000:-100]
xft=np.fft.rfft(x)

spec=bin_spec(np.real(xft*np.conj(xft)),32)
pylab.plot(spec)
pylab.show()

spec, _, _ = xcorr_channels(delta_rts, delta_rts, lblock=1024)
pylab.plot(spec)
pylab.show()
