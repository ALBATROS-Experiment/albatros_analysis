import numpy
import struct
import time
import ctypes
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
myfunc_c = mylib.myfunc
# sortpols_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_short]
sortpols_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,\
	 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_short]
mylib.dropped_packets.restype = ctypes.c_uint
mylib.dropped_packets.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_short]
# myfunc_c()

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


def get_packet_indecies(file_data, low_specnum, high_specnum, bytes_per_packet, header_bytes):
	#reading the smallest specnum and the highest specnum
	file_data.seek(header_bytes); smallest_specnum = struct.unpack(">I", file_data.read(4))[0]
	file_data.seek(-bytes_per_packet,2); highest_specnum = struct.unpack(">I", file_data.read(4))[0]
	
	#finding the number of packets (which should be the maximum number of entries in the spec_num array)
	file_data.seek(0,2); num_packets = (file_data.tell() - header_bytes)//bytes_per_packet # this is supposed to divide evenly. If it doesn't, something's wrong.
	
	if (low_specnum >= high_specnum):
		raise ValueError("The lower bound for the spectrum number choice is higher or the same as the upper bound.")
		return 0, num_packets - 1
	elif (low_specnum >= highest_specnum):
		raise ValueError("The lower bound for the spectrum number choice is too high for the range of the raw file.")
		return 0, num_packets - 1
	elif (high_specnum <= smallest_specnum):
		raise ValueError("The upper bound for the spectrum number choice is too low for the range of the raw file.")
		return 0, num_packets - 1
	else:
		#now we locate the low index and high index bounds in spec_num using a binary search
		snLowIdx = 0
		file_data.seek(header_bytes + bytes_per_packet)
		if (low_specnum >= struct.unpack(">I", file_data.read(4))[0]):
			tempUpperBoundIdx = num_packets - 1
			while True:
				file_data.seek(header_bytes + bytes_per_packet * ((tempUpperBoundIdx + snLowIdx)//2))
				halfwaySpecnum = struct.unpack(">I", file_data.read(4))[0]
				if (low_specnum < halfwaySpecnum):
					tempUpperBoundIdx = (tempUpperBoundIdx + snLowIdx)//2
				else:
					snLowIdx = (tempUpperBoundIdx + snLowIdx)//2
				
				file_data.seek(header_bytes + bytes_per_packet * snLowIdx); currentIdxSpecnum = struct.unpack(">I", file_data.read(4))[0]
				file_data.seek(header_bytes + bytes_per_packet * (snLowIdx + 1)); comparisonSpecnum = struct.unpack(">I", file_data.read(4))[0]
				if ((low_specnum >= currentIdxSpecnum) and (low_specnum < comparisonSpecnum)):
					break
		#and repeat for the spectrum number high index
		snHighIdx = num_packets - 1
		if (high_specnum < highest_specnum):
			tempLowerBoundIdx = 0
			while True:
				file_data.seek(header_bytes + bytes_per_packet * ((tempLowerBoundIdx + snHighIdx)//2))
				halfwaySpecnum = struct.unpack(">I", file_data.read(4))[0]
				if (high_specnum >= halfwaySpecnum):
					tempLowerBoundIdx = (tempLowerBoundIdx + snHighIdx)//2
				else:
					snHighIdx = (tempLowerBoundIdx + snHighIdx)//2
				
				file_data.seek(header_bytes + bytes_per_packet * snHighIdx); currentIdxSpecnum = struct.unpack(">I", file_data.read(4))[0]
				file_data.seek(header_bytes + bytes_per_packet * (snHighIdx - 1)); comparisonSpecnum = struct.unpack(">I", file_data.read(4))[0]
				if ((high_specnum > comparisonSpecnum) and (high_specnum <= currentIdxSpecnum)):
					break
		#now that we know both index bounds, we return them
		return snLowIdx, snHighIdx

class baseband_data_float:
	def __init__(self, file_name, spec_selection = False, low_specnum = 0, high_specnum = 0):
		file_data=open(file_name, "rb") #,encoding='ascii')
		header_bytes = struct.unpack(">Q", file_data.read(8))[0]
		#setting all the header values
		self.header_bytes = 8 + header_bytes # first 8 bytes were the no. of bytes in header data
		self.bytes_per_packet = struct.unpack(">Q", file_data.read(8))[0]
		self.length_channels = struct.unpack(">Q", file_data.read(8))[0]
		self.spectra_per_packet = struct.unpack(">Q", file_data.read(8))[0]
		self.bit_mode = struct.unpack(">Q", file_data.read(8))[0]
		self.have_trimble = struct.unpack(">Q", file_data.read(8))[0]
		self.channels = numpy.frombuffer(file_data.read(self.header_bytes - 88), ">%dQ"%(int((header_bytes-8*10)/8)))[0] #this line is sketchy but it should work as long as the header structure stays the same. I know there's 88 bytes of the header which is not the channel array, so the rest is the length of the channel array.
		self.gps_week = struct.unpack(">Q", file_data.read(8))[0]
		self.gps_timestamp = struct.unpack(">Q", file_data.read(8))[0]
		self.gps_latitude = struct.unpack(">d", file_data.read(8))[0]
		self.gps_longitude = struct.unpack(">d", file_data.read(8))[0]
		self.gps_elevation = struct.unpack(">d", file_data.read(8))[0]
	    	
		if self.bit_mode == 1:
			self.channels = numpy.ravel(numpy.column_stack((self.channels, self.channels+1)))
			self.length_channels = int(self.length_channels * 2)
		if self.bit_mode == 4:
			self.channels = self.channels[::2]
			self.length_channels = int(self.length_channels / 2)
	    	
		if (spec_selection == True):
			lowerPacketIndex, upperPacketIndex = get_packet_indecies(file_data, low_specnum, high_specnum, int(self.bytes_per_packet), self.header_bytes) #we need the int for bytes_per packet since it doesn't like inputting longs
			file_data.seek(int(self.header_bytes + lowerPacketIndex * self.bytes_per_packet))
			t1 = time.time()
			data = numpy.fromfile(file_data, count= upperPacketIndex - lowerPacketIndex + 1, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(self.bytes_per_packet-4))])
			t2 = time.time()
			print('took ',t2-t1,' seconds to read raw data on ', file_name)
			file_data.close()
		else:
			file_data.seek(self.header_bytes)
			t1 = time.time()
			data = numpy.fromfile(file_data, count= -1, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(self.bytes_per_packet-4))])
			t2 = time.time()

			print('took ',t2-t1,' seconds to read raw data on ', file_name)

			file_data.close()
		
		self.spec_num = numpy.array(data["spec_num"], dtype = numpy.dtype(numpy.uint64))
		
		# Dropped packets stuff will be fixed
		# self.dropped_packets = mylib.dropped_packets(data["spectra"].ctypes.data, self.spec_num.ctypes.data, len(self.spec_num), self.spectra_per_packet, self.length_channels, self.bit_mode)
		# print("Number of dropped packets: " + str(self.dropped_packets))
		
		if self.bit_mode == 4:
			self.pol0, self.pol1 = unpack_4bit(data["spectra"], self.length_channels, True)
		elif self.bit_mode == 2:
			raw_spectra = data["spectra"].reshape(-1, self.length_channels)
			self.pol0, self.pol1 = unpack_2bit(raw_spectra, self.length_channels, True)
		elif self.bit_mode == 1:
			raw_spectra = data["spectra"].reshape(-1, self.length_channels//2)
			self.pol0, self.pol1 = unpack_1bit(raw_spectra, self.length_channels, True)
		else:
			print("Unknown bit depth")
    
	def print_header(self):
		print("Header Bytes = " + str(self.header_bytes) + ". Bytes per packet = " + str(self.bytes_per_packet) + ". Channel length = " + str(self.length_channels) + ". Spectra per packet: " + str(self.spectra_per_packet) + ". Bit mode: " + str(self.bit_mode) + ". Have trimble = " + str(self.have_trimble) + ". Channels: " + str(self.channels) + " GPS week = " + str(self.gps_week)+ ". GPS timestamp = " + str(self.gps_timestamp) + ". GPS latitude = " + str(self.gps_latitude) + ". GPS longitude = " + str(self.gps_longitude) + ". GPS elevation = " + str(self.gps_elevation) + ".")
		
		
def sortpols(data, length_channels, bit_mode, missing_loc, missing_num):
	# For packed data we don't need to unpack bytes. But re-arrange the raw data in npsec x () form and separate the two pols.
	# number of rows should be nspec because we want to iterate over spectra while corr averaging in python
	assert(missing_loc.shape[0]==missing_num.shape[0])
	data1 = data.copy()
	if bit_mode == 4:
		nspec = data1.shape[0]*data1.shape[1]//length_channels//2
		nrows =  int(nspec + missing_num.sum()) #nrows is nspec + missing spectra that'll be added as zeros
		ncols = length_channels # gotta be careful with this for 1 bit and 2 bit. for 4 bits, ncols = nchans
		print(type(nspec), type(nrows),type(ncols))
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
		missing_num.ctypes.data, missing_num.shape[0], nspec, ncols, bit_mode)
	t2 = time.time()
	print("Took " + str(t2 - t1) + " to unpack")
	
	return pol0, pol1


class baseband_data_packed:
	#turn spec_selection to true and enter the range of spectra you want to save only part of the file
	def __init__(self, file_name, spec_selection = False, low_specnum = 0, high_specnum = 0):
		file_data=open(file_name, "rb") #,encoding='ascii')
		header_bytes = struct.unpack(">Q", file_data.read(8))[0]
	    	#setting all the header values
		self.header_bytes = 8 + header_bytes
		self.bytes_per_packet = struct.unpack(">Q", file_data.read(8))[0]
		self.length_channels = struct.unpack(">Q", file_data.read(8))[0]
		self.spectra_per_packet = struct.unpack(">Q", file_data.read(8))[0]
		self.bit_mode = struct.unpack(">Q", file_data.read(8))[0]
		self.have_trimble = struct.unpack(">Q", file_data.read(8))[0]
		self.channels = numpy.frombuffer(file_data.read(self.header_bytes - 88), ">%dQ"%(int((header_bytes-8*10)/8)))[0] #this line is sketchy but it should work as long as the header structure stays the same. I know there's 88 bytes of the header which is not the channel array, so the rest is the length of the channel array.
		self.gps_week = struct.unpack(">Q", file_data.read(8))[0]
		self.gps_timestamp = struct.unpack(">Q", file_data.read(8))[0]
		self.gps_latitude = struct.unpack(">d", file_data.read(8))[0]
		self.gps_longitude = struct.unpack(">d", file_data.read(8))[0]
		self.gps_elevation = struct.unpack(">d", file_data.read(8))[0]
	    	
		if self.bit_mode == 1:
			self.channels = numpy.ravel(numpy.column_stack((self.channels, self.channels+1)))
			self.length_channels = int(self.length_channels * 2)
		if self.bit_mode == 4:
			self.channels = self.channels[::2]
			self.length_channels = int(self.length_channels / 2)
	   	
		if (spec_selection == True):
			lowerPacketIndex, upperPacketIndex = get_packet_indecies(file_data, low_specnum, high_specnum, int(self.bytes_per_packet), self.header_bytes) #we need the int for bytes_per packet since it doesn't like inputting longs
			file_data.seek(int(self.header_bytes + lowerPacketIndex * self.bytes_per_packet))
			t1 = time.time()
			data = numpy.fromfile(file_data, count= upperPacketIndex - lowerPacketIndex + 1, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(self.bytes_per_packet-4))])
			t2 = time.time()
			print('took ',t2-t1,' seconds to read raw data on ', file_name)
			file_data.close()
		else:
			file_data.seek(self.header_bytes)
			t1 = time.time()
			data = numpy.fromfile(file_data, count= -1, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(self.bytes_per_packet-4))])
			t2 = time.time()
			print('took ',t2-t1,' seconds to read raw data on ', file_name)
			file_data.close()
		
		self.spec_num = numpy.array(data["spec_num"], dtype = numpy.dtype(numpy.uint64))

		specdiff=numpy.diff(self.spec_num)
		idx=numpy.where(specdiff!=self.spectra_per_packet)[0]
		self.missing_loc = (self.spec_num[idx]+self.spectra_per_packet-self.spec_num[0]-1).astype('uint32')
		self.missing_num = (specdiff[idx]-self.spectra_per_packet).astype('uint32') # number of missing spectra for each location
		
		# To be fixed later
		# self.dropped_packets = mylib.dropped_packets(data["spectra"].ctypes.data, self.spec_num.ctypes.data, len(self.spec_num), self.spectra_per_packet, self.length_channels, self.bit_mode)
		# print("Number of dropped packets: " + str(self.dropped_packets))

		self.pol0, self.pol1 = sortpols(data['spectra'], self.length_channels, self.bit_mode, self.missing_loc, self.missing_num)
    	
	def print_header(self):
		print("Header Bytes = " + str(self.header_bytes) + ". Bytes per packet = " + str(self.bytes_per_packet) + ". Channel length = " + str(self.length_channels) + ". Spectra per packet: " + str(self.spectra_per_packet) + ". Bit mode: " + str(self.bit_mode) + ". Have trimble = " + str(self.have_trimble) + ". Channels: " + str(self.channels) + " GPS week = " + str(self.gps_week)+ ". GPS timestamp = " + str(self.gps_timestamp) + ". GPS latitude = " + str(self.gps_latitude) + ". GPS longitude = " + str(self.gps_longitude) + ". GPS elevation = " + str(self.gps_elevation) + ".")
