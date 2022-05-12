import ctypes
import numpy as np
import os
import time

mylib=ctypes.cdll.LoadLibrary(os.path.realpath(__file__+r"/..")+"/lib_correlations_cpu.so")
# mylib.average_cross_correlations.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short]
# mylib.average_cross_correlations.restype = None
# mylib.average_auto.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short]
# mylib.average_auto.restype = None

mylib.autocorr_4bit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
mylib.avg_autocorr_4bit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
autocorr_4bit_c = mylib.autocorr_4bit
avg_autocorr_4bit_c = mylib.avg_autocorr_4bit

def autocorr_4bit(pol):

	data = pol.copy()
	print(data.shape)
	corr = np.zeros(data.shape,dtype='uint8',order='c') # ncols = nchan for 4 bit
	t1=time.time()
	autocorr_4bit_c(data.ctypes.data, corr.ctypes.data, data.shape[0], data.shape[1])
	t2=time.time()
	print(f"time taken for corr {t2-t1:5.3f}s")
	return corr

def avg_autocorr_4bit(pol):

	data=pol.copy()
	corr = np.zeros(data.shape[1],dtype='uint64',order='c')
	t1=time.time()
	avg_autocorr_4bit_c(data.ctypes.data, corr.ctypes.data, data.shape[0], data.shape[1])
	t2=time.time()
	print(f"time taken for avg_corr {t2-t1:5.3f}s")
	return corr





