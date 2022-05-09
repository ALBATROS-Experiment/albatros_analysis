import numpy
import ctypes
import time
import os
import sys

mylib=ctypes.cdll.LoadLibrary(os.path.realpath(__file__+r"/..")+"/lib_unpacking.so")
unpack_4bit_float_c = mylib.unpack_4bit_float
unpack_4bit_float_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
unpack_2bit_float_c = mylib.unpack_2bit_float
unpack_2bit_float_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
unpack_1bit_float_c = mylib.unpack_1bit_float
unpack_1bit_float_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
sortpols_c = mylib.sortpols
sortpols_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,\
	 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short]
hist_4bit_c = mylib.hist_4bit
hist_4bit_c.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

def hist(data, length_channels, bit_depth, mode):

    nbins = 2**bit_depth - 1
    histvals = numpy.zeros(nbins+1, dtype='uint64')

    if(bit_depth==4):
        nspec = (data.shape[0]*data.shape[1]//length_channels//2)
        print(sys.getsizeof(nspec),"bytes")
        t1=time.time()
        hist_4bit_c(data.ctypes.data, nspec*length_channels, histvals.ctypes.data, nbins, mode)
        t2=time.time()
        print("time taken for histogramming", t2-t1)
    
    return histvals

        
def unpack_4bit(data, length_channels, isfloat):
	if isfloat:
		#nspec = no. of spectra = no. of rows
		nspec = data.shape[0]*data.shape[1]//length_channels//2
		pol0 = numpy.zeros([nspec,length_channels],dtype='complex64')
		pol1 = numpy.zeros([nspec,length_channels],dtype='complex64')
		print("num spec is", nspec)
		data1=data.copy() #important to make sure no extra bytes leak in
		t1 = time.time()
		unpack_4bit_float_c(data1.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,nspec,length_channels)
		t2 = time.time()
		print("Took " + str(t2 - t1) + " to unpack")
	else:
		print("not float")
	return pol0, pol1

def unpack_2bit(data, length_channels, isfloat):
	if isfloat:
		pol0 = numpy.zeros([data.shape[0] * 2,length_channels],dtype='complex64')
		pol1 = numpy.zeros([data.shape[0] * 2,length_channels],dtype='complex64')
			
		t1 = time.time()
		unpack_2bit_float_c(data.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,data.shape[0],data.shape[1])
		t2 = time.time()
		print("Took " + str(t2 - t1) + " to unpack")
	else:
		print("not float")
	return pol0, pol1

def unpack_1bit(data, length_channels, isfloat):
	if isfloat:
		pol0 = numpy.zeros([data.shape[0],length_channels],dtype='complex64')
		pol1 = numpy.zeros([data.shape[0],length_channels],dtype='complex64')
			
		t1 = time.time()
		unpack_1bit_float_c(data.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,data.shape[0],data.shape[1])
		t2 = time.time()
		print("Took " + str(t2 - t1) + " to unpack")
	else:
		print("not float")
	return pol0, pol1


def sortpols(data, length_channels, bit_mode, missing_loc, missing_num):
	# For packed data we don't need to unpack bytes. But re-arrange the raw data in npsec x () form and separate the two pols.
	# number of rows should be nspec because we want to iterate over spectra while corr averaging in python
	assert(missing_loc.shape[0]==missing_num.shape[0])
	data1 = data.copy()
	if bit_mode == 4:
		nspec = data1.shape[0]*data1.shape[1]//length_channels//2
		nrows =  int(nspec + missing_num.sum()) #nrows is nspec + missing spectra that'll be added as zeros
		ncols = length_channels # gotta be careful with this for 1 bit and 2 bit. for 4 bits, ncols = nchans
		# print(type(nspec), type(nrows),type(ncols))
		pol0 = numpy.zeros([nrows,ncols],dtype='uint8', order = 'c')
		pol1 = numpy.zeros([nrows,ncols],dtype='uint8', order = 'c')
	elif bit_mode == 2:
		pol0 = numpy.zeros([data.shape[0]//4,length_channels],dtype='uint8', order = 'c')
		pol1 = numpy.zeros([data.shape[0]//4,length_channels],dtype='uint8', order = 'c')
	elif bit_mode == 1:
		pol0 = numpy.zeros([data.shape[0]//8,length_channels],dtype='uint8', order = 'c')
		pol1 = numpy.zeros([data.shape[0]//8,length_channels],dtype='uint8', order = 'c')
			
	t1 = time.time()
	sortpols_c(data1.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,missing_loc.ctypes.data,\
		missing_num.ctypes.data, missing_num.shape[0], nrows, ncols, bit_mode)
	t2 = time.time()
	print("Took " + str(t2 - t1) + " to unpack")
	
	return pol0, pol1