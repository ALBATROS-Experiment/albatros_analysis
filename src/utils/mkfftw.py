import ctypes
import os
import numpy as np
from contextlib import contextmanager
import atexit

NUM_CPU = os.cpu_count()
mylib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/libfftw.so"
)

many_fft_c2c_1d_c = mylib.many_fft_c2c_1d
many_fft_c2c_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

@contextmanager
def parallelize_fft(nthreads=None):
    if not nthreads:
        nthreads=NUM_CPU
    mylib.set_threads(nthreads)
    yield
    mylib.cleanup_threads()

def many_fft_c2c_1d(dat, axis=1, backward=False):
    datft = np.empty(dat.shape,dtype=dat.dtype)
    # print("datft start addr", -datft.ctypes.data%16)
    sign=2*int(backward==True)-1
    # fftw_complex *dat, fftw_complex *datft, int nrows, int ncols, int axis, int sign
    many_fft_c2c_1d_c(dat.ctypes.data, datft.ctypes.data, dat.shape[0], dat.shape[1], axis, sign)
    return datft


read_wisdom_c = mylib.read_wisdom
read_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

write_wisdom_c = mylib.write_wisdom
write_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

mylib.set_threads(os.cpu_count())
atexit.register(mylib.cleanup_threads)