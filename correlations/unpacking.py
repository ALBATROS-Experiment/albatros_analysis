import numpy
import ctypes
import time
import os
import sys

mylib=ctypes.cdll.LoadLibrary(os.path.realpath(__file__+r"/..")+"/lib_unpacking.so")
unpack_4bit_float_c = mylib.unpack_4bit_float
unpack_4bit_float_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
unpack_1bit_float_c = mylib.unpack_1bit_float
unpack_1bit_float_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
sortpols_c = mylib.sortpols
sortpols_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short, ctypes.c_int, ctypes.c_int]
hist_4bit_c = mylib.hist_4bit
hist_4bit_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

def hist(data, length_channels, bit_depth, mode):

    nbins = 2**bit_depth - 1
    histvals = numpy.empty((nbins+1,length_channels), dtype='uint64',order='c')

    if(bit_depth==4):
        nspec = (data.shape[0]*data.shape[1]//length_channels//2)
        print("nspec is ", nspec)
        t1=time.time()
        hist_4bit_c(data.ctypes.data, histvals.ctypes.data, nspec, length_channels, nbins, mode)
        t2=time.time()
        print("time taken for histogramming", t2-t1)
    return histvals

        
def unpack_4bit(data, length_channels, rowstart, rowend, chanstart,chanend):
	#nspec = no. of spectra = no. of rows
	nrows = rowend-rowstart
	ncols = chanend-chanstart
	pol0 = numpy.empty([nrows,ncols],dtype='complex64')
	pol1 = numpy.empty([nrows,ncols],dtype='complex64')
	print("num spec being unpacked is", nrows)
	t1 = time.time()
	unpack_4bit_float_c(data.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,rowstart, rowend, chanstart,chanend, length_channels)
	t2 = time.time()
	print("Took " + str(t2 - t1) + " to unpack")
	return pol0, pol1

def unpack_1bit(data, length_channels, isfloat):
	if isfloat:
		nspec = 2*data.shape[0]*data.shape[1]//length_channels
		pol0 = numpy.zeros([nspec,length_channels],dtype='complex64')
		pol1 = numpy.zeros([nspec,length_channels],dtype='complex64')
		t1 = time.time()
		unpack_1bit_float_c(data.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,nspec,length_channels)
		t2 = time.time()
		print("Took " + str(t2 - t1) + " to unpack")
	else:
		print("not float")
	return pol0, pol1


def sortpols(data, length_channels, bit_mode, spec_num, rowstart, rowend, chanstart, chanend):
	# For packed data we don't need to unpack bytes. But re-arrange the raw data in npsec x () form and separate the two pols.
	# number of rows should be nspec because we want to iterate over spectra while corr averaging in python

	# spec_num is a min subtracted array of specnums (starts at 0).

	#removed a data.copy() here. be careful.

	if(chanend is None):
		chanstart=0
		chanend=length_channels
	nrows = rowend-rowstart
	if bit_mode == 4:
		ncols = chanend-chanstart # gotta be careful with this for 1 bit and 2 bit. for 4 bits, ncols = nchans
	elif bit_mode == 1:
		if(chanstart%2>0):
			raise ValueError("ERROR: Start channel index must be even.")
		ncols = numpy.ceil((chanend-chanstart)/4).astype(int) # if num channels is not 4x, there will be a fractional byte at the end

	pol0 = numpy.empty([nrows,ncols],dtype='uint8', order = 'c')
	pol1 = numpy.empty([nrows,ncols],dtype='uint8', order = 'c')
	# we're passing ncols because ncols not always chanend-chanstart. although could do it on C side.
	t1 = time.time()
	sortpols_c(data.ctypes.data, pol0.ctypes.data, pol1.ctypes.data, rowstart, rowend, ncols, length_channels, bit_mode, chanstart, chanend)
	t2 = time.time()
	print(f"Took {(t2 - t1):5.3f} to unpack")
	
	return pol0, pol1