import numpy
import ctypes
import time
import os
import sys

mylib=ctypes.cdll.LoadLibrary(os.path.realpath(__file__+r"/..")+"/lib_unpacking.so")
unpack_4bit_float_c = mylib.unpack_4bit_float
unpack_4bit_float_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
unpack_1bit_float_c = mylib.unpack_1bit_float
unpack_1bit_float_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
sortpols_c = mylib.sortpols
sortpols_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short, ctypes.c_int, ctypes.c_int]
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


def sortpols(data, length_channels, bit_mode, spec_num, chanstart, chanend):
	# For packed data we don't need to unpack bytes. But re-arrange the raw data in npsec x () form and separate the two pols.
	# number of rows should be nspec because we want to iterate over spectra while corr averaging in python
	data1 = data.copy()
	if(chanend is None):
		chanstart=0
		chanend=length_channels
	
	sp_num = spec_num.copy() # don't trust slices anymore
	if bit_mode == 4:
		spectra_per_packet = data1.shape[1]//length_channels//2
		nspec = data1.shape[0]*spectra_per_packet
		nrows =  int(spec_num[-1] + spectra_per_packet) #nrows is nspec + missing spectra that'll be added as zeros
		# print(type(spectra_per_packet), type(data1.shape[0]))
		print(f"nrows: {nrows}, nspec: {nspec}")
		ncols = chanend-chanstart # gotta be careful with this for 1 bit and 2 bit. for 4 bits, ncols = nchans
		# print(type(nspec), type(nrows),type(ncols))
		pol0 = numpy.empty([nrows,ncols],dtype='uint8', order = 'c')
		pol1 = numpy.empty([nrows,ncols],dtype='uint8', order = 'c')
	elif bit_mode == 1:
		if(chanstart%2>0):
			raise ValueError("ERROR: Start channel index must be even.")
		spectra_per_packet = data.shape[1]*2//length_channels
		print("calculated spec per packet", spectra_per_packet)
		ncols = numpy.ceil((chanend-chanstart)/4).astype(int) # if num channels is not 4x, there will be a fractional byte at the end
		nspec = data1.shape[0]*spectra_per_packet
		nrows = nspec # fundamentally we cannot insert 0s in 1 bit because there's no 0 level
		pol0 = numpy.empty([nrows,ncols],dtype='uint8', order = 'c')
		pol1 = numpy.empty([nrows,ncols],dtype='uint8', order = 'c')
			
	t1 = time.time()
	sortpols_c(data1.ctypes.data, pol0.ctypes.data, pol1.ctypes.data, sp_num.ctypes.data, data1.shape[0], nrows, ncols, spectra_per_packet, length_channels, bit_mode, chanstart, chanend)
	t2 = time.time()
	print(f"Took {(t2 - t1):5.3f} to unpack")
	
	return pol0, pol1