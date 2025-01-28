import numpy
import ctypes
import time
import os
from .. import xp
import cupy as cp

mylib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/lib_unpacking.so"
)
_unpack_4bit_float_c = mylib.unpack_4bit_float
_unpack_4bit_float_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]
unpack_1bit_float_c = mylib.unpack_1bit_float
unpack_1bit_float_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
sortpols_c = mylib.sortpols
sortpols_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_short,
    ctypes.c_int,
    ctypes.c_int,
]
hist_4bit_c = mylib.hist_4bit
hist_4bit_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
hist_1bit_c = mylib.hist_1bit
hist_1bit_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def hist(data, rowstart, rowend, length_channels, bit_depth, mode):
    """Gets list of values ready for plotting in histogram.

    Parameters
    ----------
    data: ??
        ?? What shape??
    rowstart: ??
        ??
    rowend: ??
        ??
    length_channels: ??
        ??
    bit_depth: int
        Either 1 or 4
    mode: int
        mode=0 for pol0, 1 for pol1, -1 for both.

    Returns
    -------
    histvals: array-like
        Binned frequency-of-occurence for each quantization level.
    """
    nbins = 2**bit_depth - 1
    histvals = numpy.empty((nbins + 1, length_channels), dtype="uint64", order="c")
    assert bit_depth in (
        1,
        4,
    ), f"bit_depth set to {bit_depth}, only takes values 1 or 4"
    assert mode in (0, 1, -1), f"mode set to {mode}, only takes values 0,1,-1"
    if bit_depth == 4:
        t1 = time.time()
        hist_4bit_c(
            data.ctypes.data,
            histvals.ctypes.data,
            rowstart,
            rowend,
            length_channels,
            nbins,
            mode,
        )
        t2 = time.time()
        print("time taken for histogramming", t2 - t1)
    if bit_depth == 1:
        t1 = time.time()
        hist_1bit_c(
            data.ctypes.data,
            histvals.ctypes.data,
            rowstart,
            rowend,
            length_channels,
            nbins,
            mode,
        )
        t2 = time.time()
        print("time taken for histogramming", t2 - t1)
    return histvals

def _unpack_4bit_float_gpu(data, rowstart, rowend, channels, length_channels):
    nrows = int(rowend-rowstart)
    ncols = len(channels)
    bytes_per_spectra = 2*length_channels
    d_data = data.reshape(-1,bytes_per_spectra) #data should already be on the GPU
    p0re = xp.right_shift(xp.bitwise_and(d_data[rowstart:rowend, 2*channels], 0xf0), 4, dtype='int8')
    p0re[p0re>8] = p0re[p0re>8] - 16
    p0im = xp.bitwise_and(d_data[rowstart:rowend, 2*channels], 0x0f, dtype='int8')
    p0im[p0im>8] = p0im[p0im>8] - 16
    p1re = xp.right_shift(xp.bitwise_and(d_data[rowstart:rowend, 2*channels+1], 0xf0), 4, dtype='int8')
    p1re[p1re>8] = p1re[p1re>8] - 16
    p1im = xp.bitwise_and(d_data[rowstart:rowend, 2*channels+1], 0x0f, dtype='int8')
    p1im[p1im>8] = p1im[p1im>8] - 16
    d_pol0 = xp.array(p0re+1j*p0im,dtype='complex64')
    d_pol1 = xp.array(p1re+1j*p1im,dtype='complex64')
    return d_pol0, d_pol1

_fill_1bit_float = cp.ElementwiseKernel(
    'int8 re, int8 im',
    'complex64 pol',
    'pol = complex<float>(static_cast<float>(re), static_cast<float>(im))',
    '_fill_1bit_float'
)

def _unpack_1bit_float_gpu(data, rowstart, rowend, channels, length_channels):
    nrows = int(rowend-rowstart)
    ncols = len(channels)
    print(nrows, ncols, "from unpack 1 bit gpu")
    bytes_per_spectra = length_channels//2
    cols=channels[::2]//2
    d_pol0 = xp.empty((nrows,ncols),dtype='complex64')
    d_pol1 = xp.empty((nrows,ncols),dtype='complex64')
    # print("allocated mem for pols", 2*nrows*ncols*64/8/1024**3, "GB")
    # print("allocated mem for raw data", data.shape[0]*data.shape[1]/8/1024**3, "GB")
    d_data = data.reshape(-1,bytes_per_spectra) #data should already be on the GPU
    
    r0c0 = 2*xp.bitwise_and(xp.right_shift(d_data[rowstart:rowend, cols], 7),1,dtype='int8')-1
    i0c0 = 2*xp.bitwise_and(xp.right_shift(d_data[rowstart:rowend, cols], 6),1,dtype='int8')-1
    r1c0 = 2*xp.bitwise_and(xp.right_shift(d_data[rowstart:rowend, cols], 5),1,dtype='int8')-1
    i1c0 = 2*xp.bitwise_and(xp.right_shift(d_data[rowstart:rowend, cols], 4),1,dtype='int8')-1
    r0c1 = 2*xp.bitwise_and(xp.right_shift(d_data[rowstart:rowend, cols], 3),1,dtype='int8')-1
    i0c1 = 2*xp.bitwise_and(xp.right_shift(d_data[rowstart:rowend, cols], 2),1,dtype='int8')-1
    r1c1 = 2*xp.bitwise_and(xp.right_shift(d_data[rowstart:rowend, cols], 1),1,dtype='int8')-1
    i1c1 = 2*xp.bitwise_and(d_data[rowstart:rowend, cols],1,dtype='int8')-1

    d_pol0[:,::2] = _fill_1bit_float(r0c0, i0c0) #all even channels, pol0
    d_pol0[:,1::2] = _fill_1bit_float(r0c1, i0c1) #all odd channels, pol0
    d_pol1[:,::2] = _fill_1bit_float(r1c0, i1c0) #all even channels, pol1
    d_pol1[:,1::2] = _fill_1bit_float(r1c1, i1c1) #all odd channels, pol1

    return d_pol0, d_pol1

def unpack_4bit(data, rowstart, rowend, channels, length_channels):
    """Unpacks the raw baseband data file in 4-bit mode 
    and inflates array of electric field as 64-bit complex numbers (fp32, fp32)..
    Returns pol0 and pol1 separately.

    Parameters
    ----------
    data : np.ndarray or cp.ndarray
        Raw data file returned from Baseband class. Normally, n_packets x n_bytes_per_packet
    length_channels : int
        Number of channels in the raw baseband file
    rowstart : int
        Row number corresponding to first spectrum (numpy index convention)
    rowend : int
        Row number corresponding to last spectrum (numpy index convention)
        rowstart, rowend = 10, 20 unpacks 10 spectra starting at 11th spectrum (index 10).
    channels : list
        Channel INDICES in numpy convention (not channel numbers) you'd like to unpack. 
        E.g. channels = [0, 2, 4] corresponds to channels at index 0, 2, and 4.

    Returns
    -------
    (xp.ndarray, xp.ndarray)
        Arrays of nrows x len(channels) corresponding to pol0 and pol1.
        Cupy arrays on device, if GPU in use.
    """
    # nspec = no. of spectra = no. of rows
    # print("num spec being unpacked is", nrows)
    if xp.__name__=='numpy':
        nrows = rowend - rowstart
        ncols = len(channels)
        pol0 = xp.empty([nrows, ncols], dtype="complex64")
        pol1 = xp.empty([nrows, ncols], dtype="complex64")
        t1 = time.time()
        # uint8_t *data, float *pol0, float *pol1, int rowstart, int rowend, int * channels, int ncols, int nchan
        _unpack_4bit_float_c(
            data.ctypes.data,
            pol0.ctypes.data,
            pol1.ctypes.data,
            rowstart,
            rowend,
            channels.ctypes.data,
            ncols,
            length_channels,
        )
        t2 = time.time()
    elif xp.__name__=='cupy':
        assert isinstance(data, xp.ndarray)
        t1=time.time()
        pol0,pol1 = _unpack_4bit_float_gpu(data, rowstart, rowend, channels, length_channels)
        t2=time.time()
    # print("Took " + str(t2 - t1) + " to unpack")
    return pol0, pol1


def unpack_1bit(data, rowstart, rowend, channels, length_channels):
    """Unpacks the raw baseband data file in 1-bit mode 
    and inflates array of electric field as 64-bit complex numbers (fp32, fp32).
    Returns pol0 and pol1 separately.

    Parameters
    ----------
    data : np.ndarray
        Raw data file returned from Baseband class. Normally, n_packets x n_bytes_per_packet
    length_channels : int
        Number of channels in the raw baseband file
    rowstart : int
        Row number corresponding to first spectrum (numpy index convention)
    rowend : int
        Row number corresponding to last spectrum (numpy index convention)
        rowstart, rowend = 10, 20 unpacks 10 spectra starting at 11th spectrum (index 10).
    channels : list
        Channel INDICES in numpy convention (not channel numbers) you'd like to unpack. 
        E.g. channels = [0, 2, 4] corresponds to channels at index 0, 2, and 4.
        In this mode, channel indices must be continuous, and must start at an even channel index.
        Total number of channels should also be a multiple of 2.

    Returns
    -------
    (xp.ndarray, xp.ndarray)
        Arrays of nrows x len(channels) corresponding to pol0 and pol1.
        Cupy arrays on device, if GPU in use.
    """
    if xp.__name__=='numpy':
        chanstart = channels[0]
        chanend = channels[-1]+1
        nrows = rowend - rowstart
        ncols = chanend - chanstart
        print("print shape of my pols", nrows, ncols)
        pol0 = numpy.empty([nrows, ncols], dtype="complex64")
        pol1 = numpy.empty([nrows, ncols], dtype="complex64")
        t1 = time.time()
        unpack_1bit_float_c(
            data.ctypes.data,
            pol0.ctypes.data,
            pol1.ctypes.data,
            rowstart,
            rowend,
            chanstart,
            chanend,
            length_channels,
        )
        t2 = time.time()
    elif xp.__name__=='cupy':
        assert isinstance(data, xp.ndarray)
        t1=time.time()
        pol0,pol1 = _unpack_1bit_float_gpu(data, rowstart, rowend, channels, length_channels)
        t2=time.time()
    print("Took " + str(t2 - t1) + " to unpack")
    return pol0, pol1


def sortpols(data, length_channels, bit_mode, rowstart, rowend, chanstart, chanend):
    """??

    ??

    Parameters
    ----------
    data: ??
        ??
    length_channels: ??
        ??
    bit_mode: ??
        ??
    rowstart: ??
        ??
    rowend: ??
        ??
    chanstart: ??
        ??
    chanend: ??
        ??

    Returns
    -------
    pol0: ??
        ??
    pol1: ??
        ??
    """
    # For packed data we don't need to unpack bytes. But re-arrange the raw data in npsec x () form and separate the two pols.
    # number of rows should be nspec because we want to iterate over spectra while corr averaging in python

    # spec_num is a min subtracted array of specnums (starts at 0).

    # removed a data.copy() here. be careful.

    if chanend is None:
        chanstart = 0
        chanend = length_channels
    nrows = rowend - rowstart
    if bit_mode == 4:
        ncols = (
            chanend - chanstart
        )  # gotta be careful with this for 1 bit and 2 bit. for 4 bits, ncols = nchans
    elif bit_mode == 1:
        if chanstart % 2 > 0:
            raise ValueError("ERROR: Start channel index must be even.")
        ncols = numpy.ceil((chanend - chanstart) / 4).astype(
            int
        )  # if num channels is not 4x, there will be a fractional byte at the end

    pol0 = numpy.empty([nrows, ncols], dtype="uint8", order="c")
    pol1 = numpy.empty([nrows, ncols], dtype="uint8", order="c")
    # we're passing ncols because ncols not always chanend-chanstart. although could do it on C side.
    t1 = time.time()
    sortpols_c(
        data.ctypes.data,
        pol0.ctypes.data,
        pol1.ctypes.data,
        rowstart,
        rowend,
        ncols,
        length_channels,
        bit_mode,
        chanstart,
        chanend,
    )
    t2 = time.time()
    # print(f"Took {(t2 - t1):5.3f} to unpack")

    return pol0, pol1
