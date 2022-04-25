from baseband_data_classes import baseband_data_packed
import ctypes
import numpy
import os

mylib=ctypes.cdll.LoadLibrary(os.path.realpath(__file__+r"/..")+"/lib_correlations_cpu.so")
mylib.average_cross_correlations.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short]
mylib.average_cross_correlations.restype = None
mylib.average_auto.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short]
mylib.average_auto.restype = None

def average_cross_correlations (filename, chunkSize=1000):
	bb_data = baseband_data_packed(filename)
	
	averages_pointer = (ctypes.c_float * (2 * bb_data.length_channels))()
	if bb_data.bit_mode == 4 or bb_data.bit_mode == 2:
		mylib.average_cross_correlations(bb_data.pol0.ctypes.data, bb_data.pol1.ctypes.data, averages_pointer, bb_data.pol0.shape[0], bb_data.length_channels, chunkSize, bb_data.bit_mode)
	else: #for 1 bit the length is bb_data.pol0.shape[0]/2 + 1
		mylib.average_cross_correlations(bb_data.pol0.ctypes.data, bb_data.pol1.ctypes.data, averages_pointer, bb_data.pol0.shape[0]//2 + 1, bb_data.length_channels, chunkSize, bb_data.bit_mode) 
	
	averages = numpy.zeros(bb_data.length_channels, dtype = "complex64")
	for i in range(bb_data.length_channels):
		averages[i] = averages_pointer[2 * i] + averages_pointer[2 * i + 1] * 1j
	
	print(averages)
	
def average_auto_correlations (filename, chunkSize=1000):
	bb_data = baseband_data_packed(filename)
	
	averages_pointer = (ctypes.c_float * (bb_data.length_channels))()
	
	print("")
	
	mylib.average_auto(bb_data.pol0.ctypes.data, averages_pointer, bb_data.pol0.shape[0], bb_data.length_channels, chunkSize, bb_data.bit_mode)
	print("Average autocorrelations for pol0:")
	print(list(averages_pointer))
	
	mylib.average_auto(bb_data.pol1.ctypes.data, averages_pointer, bb_data.pol0.shape[0], bb_data.length_channels, chunkSize, bb_data.bit_mode)	
	print("Average autocorrelations for pol1:")
	print(list(averages_pointer))
