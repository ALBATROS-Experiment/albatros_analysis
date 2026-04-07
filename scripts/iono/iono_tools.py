import cupy as cp
import numpy as np
from scipy.signal import butter, sosfilt
from albatros_analysis.src.utils import pycufft

ddc_kernel = cp.ElementwiseKernel(
    "float32 x, float32 f, float32 fs",
    "complex64 y",
    """
    float samples_per_cycle = fs / f;
    float t = i % samples_per_cycle;
    float phase = 2 * M_PI * t / samples_per_cycle;
    y = complex<float>(x * cos(phase), -x * sin(phase));
    """,
    "ddc_kernel",
)


# steps
class Mixer:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def mix(self, timestream, frequency):
        pass


class Filter:
    def __init__(self, cutoff_freq, sampling_rate, order=5):
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        self.filter_len = 1024 * int(
            20 * self.sampling_rate / self.cutoff_freq / 1024
        )  # closest multiple of 1024 so mem movement is fast
        self.block_size = self.filter_len * 16
        self.impulse_response = None
        self.filter_sos = None
        self.hf = None

    @property
    def hf(self):
        """Filter frequency response

        Returns:
            cp.ndarray: complex64 array of filter spectrum
        """
        if self.hf is None:
            self._hf = cp.fft.fft(
                cp.hstack(
                    [
                        self.impulse_response,
                        cp.zeros(self.block_size - self.filter_len, dtype="complex64"),
                    ]
                )
            )
        return self._hf

    @property
    def filter_sos(self):
        if self.filter_sos is None:
            nyquist = 0.5 * self.sampling_rate
            normal_cutoff = self.cutoff_freq / nyquist
            self._sos = butter(self.order, normal_cutoff, btype="low", output="sos")
        return self._sos

    @property
    def impulse_response(self):
        if self.impulse_response is None:
            self.sos = self._design_filter()
            x = np.zeros(self.filter_len)
            x[0] = 1
            self._impulse_reponse = cp.asarray(sosfilt(self.sos, x), dtype='complex64')
        return self._impulse_reponse

    def apply_filter(self, x):
        chop = len(x)//self.block_size
        x = x.astype('complex64') #because pycufft wants complex64 right now
        x2d = x[:self.block_size*chop].reshape(-1,self.block_size)
        x2df = pycufft.fft(x2d,axis=1)
        x2d_filt = pycufft.ifft(x2df * self.hf[None, :], axis=1)
        #now we stich the ends together
        temp = cp.empty((x2d.shape[0]-1,2*self.filter_len),dtype='complex64')
        small_hf = cp.zeros(2*self.filter_len,dtype='complex64')
        small_hf[:self.filter_len] = self.impulse_response
        small_hf = cp.fft.fft(small_hf)
        temp[:, :self.filter_len] = x2d[:-1,-self.filter_len:] #last few bits of previous
        temp[:, self.filter_len:] = x2d[1:,:self.filter_len] #first few bits of next
        tempf = pycufft.fft(temp,axis=1)
        patch = pycufft.ifft(tempf * small_hf[None,:], axis=1)
        x2d_filt[1:, :self.filter_len] = patch[self.filter_len:] #put the correctly stiched values
        return x2d_filt
