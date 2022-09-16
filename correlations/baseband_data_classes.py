import numpy
import struct
import time
import numba as nb
# import unpacking as unpk
from . import unpacking as unpk

@nb.njit(parallel=True)
def fill_arr(myarr,specnum, spec_per_packet):
    n=len(specnum)
    for i in nb.prange(n):
        for j in nb.prange(spec_per_packet):
            myarr[i*spec_per_packet+j] = specnum[i]+j

class Baseband:
	def __init__(self, file_name):
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
		
		file_data.seek(self.header_bytes)
		t1 = time.time()
		data = numpy.fromfile(file_data, count= -1, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(self.bytes_per_packet-4))])
		t2 = time.time()
		print(f'took {t2-t1:5.3f} seconds to read raw data on ', file_name)
		file_data.close()
		
		self.spec_num = numpy.array(data["spec_num"], dtype = "int64")
		self.raw_data = numpy.array(data["spectra"], dtype = "uint8")
		self.spec_idx = numpy.zeros(self.spec_num.shape[0]*self.spectra_per_packet, dtype = "int64") # keep dtype int64 otherwise numpy binary search becomes slow
		fill_arr(self.spec_idx, self.spec_num, self.spectra_per_packet)
		# self.spec_idx = self.spec_idx - self.spec_idx[0]
	
	def print_header(self):
		print("Header Bytes = " + str(self.header_bytes) + ". Bytes per packet = " + str(self.bytes_per_packet) + ". Channel length = " + str(self.length_channels) + ". Spectra per packet: " +\
			str(self.spectra_per_packet) + ". Bit mode: " + str(self.bit_mode) + ". Have trimble = " + str(self.have_trimble) + ". Channels: " + str(self.channels) + \
				" GPS week = " + str(self.gps_week)+ ". GPS timestamp = " + str(self.gps_timestamp) + ". GPS latitude = " + str(self.gps_latitude) + ". GPS longitude = " +\
					str(self.gps_longitude) + ". GPS elevation = " + str(self.gps_elevation) + ".")
	
	def get_hist(self, mode=-1):
		# mode = 0 for pol0, 1 for pol1, -1 for both
		return unpk.hist(self.raw_data, self.length_channels, self.bit_mode, mode)


class BasebandFloat(Baseband):
	def __init__(self, file_name):
		super().__init__(file_name)

		if self.bit_mode == 4:
			self.pol0, self.pol1 = unpk.unpack_4bit(self.raw_data, self.length_channels, True)
		elif self.bit_mode == 2:
			raw_spectra = self.raw_data.reshape(-1, self.length_channels)
			self.pol0, self.pol1 = unpk.unpack_2bit(raw_spectra, self.length_channels, True)
		elif self.bit_mode == 1:
			self.pol0, self.pol1 = unpk.unpack_1bit(self.raw_data, self.length_channels, True)
		else:
			print("Unknown bit depth")


class BasebandPacked(Baseband):
	#turn spec_selection to true and enter the range of spectra you want to save only part of the file
	def __init__(self, file_name, chanstart=0, chanend=None):
		super().__init__(file_name)
		specdiff=numpy.diff(self.spec_num)
		idx=numpy.where(specdiff!=self.spectra_per_packet)[0]
		self.missing_loc = (self.spec_num[idx]+self.spectra_per_packet-self.spec_num[0]).astype('uint32')
		self.missing_num = (specdiff[idx]-self.spectra_per_packet).astype('uint32')

		self.spec_idx2 = self.spec_num - self.spec_num[0]
		# print(self.spectra_per_packet)
		# unpk.sortpols(self.raw_data, self.length_channels, self.bit_mode, spec_idx)
		self.pol0, self.pol1 = unpk.sortpols(self.raw_data, self.length_channels, self.bit_mode, self.spec_idx, chanstart, chanend)
		# print(self.pol0.strides)

def get_rows_from_specnum(stidx,endidx,spec_arr):
    #follows numpy convention
    #endidx is assumed not included
    # print("utils get_rows received:",stidx,endidx,spec_arr)
    l=numpy.searchsorted(spec_arr,stidx,side='left')
    r=numpy.searchsorted(spec_arr,endidx,side='left')
    return l, r

class BasebandFileIterator():
	def __init__(self, file_paths, fileidx, idxstart, acclen, nchunks=None):
		#you need to pass nchunks if you are passing the iterator to zip(). without nchunks, iteration won't stop
		self.acclen=acclen
		self.file_paths = file_paths
		self.fileidx=fileidx
		self.nchunks=nchunks
		self.chunksread=0
		self.obj = BasebandPacked(file_paths[fileidx],unpack=False)
		self.spec_num_start=idxstart+self.obj.spec_idx[0] # REPLACE SPEC_IDX to be SPEC_NUM, not 0 indexed
	
	def __iter__(self):
		return self

	def __next__(self):
		if(self.nchunks and self.chunksread==self.nchunks):
			raise StopIteration
		data=numpy.zeros(self.acclen,self.obj.length_channels) #for now take all channels. will modify to accept chanstart, chanend
		specnums=numpy.array([]) #len of this array will control everything in corr, neeed the len.

		rem=self.acclen
		while(rem):
			if(self.spec_num_start < self.obj.spec_num[0]):
				# we are in a gap between the files
				step = min(self.obj.spec_num[0]-self.spec_num_start,rem)
				rem-=step
				self.spec_num_start+=step
			else:
				l = self.obj.spec_idx[-1]-self.spec_num_start+1 #length to end from the point in file we're starting from
				if(rem>=l):
					#spillover to next file. 
					rowstart, rowend = get_rows_from_specnum(self.spec_num_start,self.spec_num_start+l,self.obj.spec_idx)
					specnums=numpy.append(specnums,self.obj.spec_idx[rowstart:rowend])
					rem-=l
					self.spec_num_start+=l
				else:
					rowstart, rowend = get_rows_from_specnum(self.spec_num_start,self.spec_num_start+rem,self.obj.spec_idx)
					specnums=numpy.append(specnums,self.obj.spec_idx[rowstart:rowend])
					rem=0
					self.spec_num_start+=rem
		self.chunksread+=1
		return [data,specnums]

			

