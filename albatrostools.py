import numpy
import struct
import datetime
#import scio
import ctypes
import time

mylib=ctypes.cdll.LoadLibrary("libalbatrostools.so")
unpack_4bit_c=mylib.unpack_4bit
unpack_4bit_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
unpack_4bit_float_c=mylib.unpack_4bit_float
unpack_4bit_float_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
unpack_1bit_float_c=mylib.unpack_1bit_float
unpack_1bit_float_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
bin_crosses_float_c=mylib.bin_crosses_float
bin_crosses_float_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
bin_crosses_double_c=mylib.bin_crosses_double
bin_crosses_double_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
bin_autos_float_c=mylib.bin_autos_float
bin_autos_float_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
bin_autos_double_c=mylib.bin_autos_double
bin_autos_double_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]



def unpack_1_bit(data, num_channels):
    real_pol0_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x80), 7), dtype="int8")
    imag_pol0_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x40), 6), dtype="int8")
    real_pol1_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x20), 5), dtype="int8")
    imag_pol1_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x10), 4), dtype="int8")
    real_pol0_chan1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x08), 3), dtype="int8")
    imag_pol0_chan1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x04), 2), dtype="int8")
    real_pol1_chan1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x02), 1), dtype="int8")
    imag_pol1_chan1=numpy.asarray(numpy.bitwise_and(data, 0x01), dtype="int8")
    real_pol0=numpy.ravel(numpy.column_stack((real_pol0_chan0, real_pol0_chan1)))
    imag_pol0=numpy.ravel(numpy.column_stack((imag_pol0_chan0, imag_pol0_chan1)))
    real_pol1=numpy.ravel(numpy.column_stack((real_pol1_chan0, real_pol1_chan1)))
    imag_pol1=numpy.ravel(numpy.column_stack((imag_pol1_chan0, imag_pol1_chan1)))
    if True:
        real_pol0[real_pol0==0]=-1
        imag_pol0[imag_pol0==0]=-1
        real_pol1[real_pol1==0]=-1
        imag_pol1[imag_pol1==0]=-1
        pol0=real_pol0+1J*imag_pol0
        pol1=real_pol1+1J*imag_pol1
    else:
        pol0=2*real_pol0+2J*imag_pol0-(1+1J)
        pol1=2*real_pol1+2J*imag_pol1-(1+1J)
    del real_pol0
    del imag_pol0
    del real_pol1
    del imag_pol1
    pol0=numpy.reshape(pol0, (-1, num_channels))
    pol1=numpy.reshape(pol1, (-1, num_channels))
    return pol0, pol1
    
def unpack_2_bit(data, num_channels):
    real_pol0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0xC0), 6), dtype="int8")
    imag_pol0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x30), 4), dtype="int8")
    real_pol1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x0C), 2), dtype="int8")
    imag_pol1=numpy.asarray(numpy.bitwise_and(data, 0x03), dtype="int8")
    real_pol0[real_pol0<=1]=real_pol0[real_pol0<=1]-2
    real_pol0[real_pol0>=2]=real_pol0[real_pol0>=2]-1
    imag_pol0[imag_pol0<=1]=imag_pol0[imag_pol0<=1]-2
    imag_pol0[imag_pol0>=2]=imag_pol0[imag_pol0>=2]-1
    real_pol1[real_pol1<=1]=real_pol1[real_pol1<=1]-2
    real_pol1[real_pol1>=2]=real_pol1[real_pol1>=2]-1
    imag_pol1[imag_pol1<=1]=imag_pol1[imag_pol1<=1]-2
    imag_pol1[imag_pol1>=2]=imag_pol1[imag_pol1>=2]-1
    pol0=real_pol0+1J*imag_pol0
    pol1=real_pol1+1J*imag_pol1
    del real_pol0
    del imag_pol0
    del real_pol1
    del imag_pol1
    pol0=pol0.reshape(-1, num_channels)
    pol1=pol1.reshape(-1, num_channels)
    return pol0, pol1
    

def unpack_1bit_fast(data,num_channels,float=False):
    if float:
        pol0=numpy.empty([data.shape[0]*2,num_channels],dtype='complex64')
        pol1=numpy.empty([data.shape[0]*2,num_channels],dtype='complex64')
        unpack_1bit_float_c(data.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,data.shape[0],data.shape[1]);
        
    else:
        return None
    return pol0,pol1;

    
def unpack_4bit_fast(data,num_channels,float=False):
    if float:
        pol0=numpy.zeros([data.shape[0]//2,num_channels],dtype='complex64')
        pol1=numpy.zeros([data.shape[0]//2,num_channels],dtype='complex64')
        unpack_4bit_float_c(data.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,data.shape[0],data.shape[1])
    else:
        pol0=numpy.zeros([data.shape[0]//2,num_channels],dtype='complex')
        pol1=numpy.zeros([data.shape[0]//2,num_channels],dtype='complex')
        unpack_4bit_c(data.ctypes.data,pol0.ctypes.data,pol1.ctypes.data,data.shape[0],data.shape[1])

    return pol0,pol1
def unpack_4_bit(data, num_channels):
    #print('data shape is ',data.shape,data.dtype)
    #unpack_4bit_fast(data,num_channels)
    #assert(1==0)
    pol0_bytes=data[:, 0::2]
    pol1_bytes=data[:, 1::2]
    real_pol0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(pol0_bytes, 0xf0), 4), dtype="int8")
    imag_pol0=numpy.asarray(numpy.bitwise_and(pol0_bytes, 0x0f), dtype="int8")
    real_pol1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(pol1_bytes, 0xf0), 4), dtype="int8")
    imag_pol1=numpy.asarray(numpy.bitwise_and(pol1_bytes, 0x0f), dtype="int8")
    real_pol0[real_pol0>8]=real_pol0[real_pol0>8]-16
    imag_pol0[imag_pol0>8]=imag_pol0[imag_pol0>8]-16
    real_pol1[real_pol1>8]=real_pol1[real_pol1>8]-16
    imag_pol1[imag_pol1>8]=imag_pol1[imag_pol1>8]-16
    pol0=real_pol0+1J*imag_pol0
    pol1=real_pol1+1J*imag_pol1
    del real_pol0
    del imag_pol0
    del real_pol1
    del imag_pol1
    pol0=pol0.reshape(-1, num_channels)
    pol1=pol1.reshape(-1, num_channels)
    return pol0, pol1

def bin_crosses(pol0,pol1,chunk=100):
    ndat=pol0.shape[0]
    nchan=pol0.shape[1]
    nchunk=ndat//chunk
    print('nchunk is ',nchunk)
    if pol0.itemsize==8:
        spec=numpy.zeros([nchunk,nchan],dtype='complex64')
        bin_crosses_float_c(pol0.ctypes.data,pol1.ctypes.data,spec.ctypes.data,ndat,nchan,chunk)
    else:
        assert(pol0.itemsize==16) #better be double precision complex
        spec=numpy.zeros([nchunk,nchan],dtype='complex128')
        bin_crosses_double_c(pol0.ctypes.data,pol1.ctypes.data,spec.ctypes.data,ndat,nchan,chunk)
        
    return spec
def bin_autos(dat,chunk=100):
    ndat=dat.shape[0]
    nchan=dat.shape[1]
    nchunk=ndat//chunk
    if dat.itemsize==8: #this means we're in single precision
        spec=numpy.zeros([nchunk,nchan],dtype='float32')
        bin_autos_float_c(dat.ctypes.data,spec.ctypes.data,ndat,nchan,chunk)
    else:
        spec=numpy.zeros([nhcunk,nchan],dtype='float64')
        bin_autos_double_c(dat.ctypes.data,spec.ctypes.data,ndat,nchan,chunk)
    return spec

def correlate(pol0, pol1):
    pols={}
    data=[pol0, pol1]
    for i in range(2):
        for j in range(i, 2):
            pols["pol%d%d"%(i, j)]=data[i]*numpy.conj(data[j])
    return pols

def get_header(file_name, gps_format='timestamp'):
    file_data=open(file_name, "rb") #,encoding='ascii')
    #file_data=open(file_name, "r")
    #header_bytes=struct.unpack(">Q", numpy.fromfile(file_data,'int',1))[0]
    header_bytes=struct.unpack(">Q", file_data.read(8))[0]
    #print(header_bytes,type(header_bytes))
    header_raw=file_data.read(header_bytes)
    #header_raw=numpy.fromfile(file_data,'int8',header_bytes)
    file_data.close()

    header_data=numpy.frombuffer(header_raw, dtype=[("bytes_per_packet", ">Q"), ("length_channels", ">Q"), ("spectra_per_packet", ">Q"), ("bit_mode", ">Q"), ("have_trimble", ">Q"), ("channels", ">%dQ"%(int((header_bytes-8*10)/8))), ("gps_week", ">Q"), ("gps_seconds", ">Q"), ("gps_lat", ">d"), ("gps_lon", ">d"), ("gps_elev", ">d")])
    if header_data["gps_week"][0] == 0:
	# New header format with ctime GPS timestamp
	header={"header_bytes":8+header_bytes,
                "bytes_per_packet":header_data["bytes_per_packet"][0],
                "length_channels":header_data["length_channels"][0],
                "spectra_per_packet":header_data["spectra_per_packet"][0],
                "bit_mode":header_data["bit_mode"][0],
                "have_trimble":header_data["have_trimble"][0],
                "channels":header_data["channels"][0],
                "gps_timestamp":header_data["gps_seconds"][0],
                "gps_latitude":header_data["gps_lat"][0],
                "gps_longitude":header_data["gps_lon"][0],
                "gps_elevation":header_data["gps_elev"][0]}
    else:
	# Old header format with GPS week and seconds
	header={"header_bytes":8+header_bytes,
                "bytes_per_packet":header_data["bytes_per_packet"][0],
                "length_channels":header_data["length_channels"][0],
                "spectra_per_packet":header_data["spectra_per_packet"][0],
                "bit_mode":header_data["bit_mode"][0],
                "have_trimble":header_data["have_trimble"][0],
                "channels":header_data["channels"][0],
                "gps_week":header_data["gps_week"][0],
                "gps_seconds":header_data["gps_seconds"][0],
                "gps_latitude":header_data["gps_lat"][0],
                "gps_longitude":header_data["gps_lon"][0],
                "gps_elevation":header_data["gps_elev"][0]}

    if header["bit_mode"]==1:
        header["channels"]=numpy.ravel(numpy.column_stack((header["channels"], header["channels"]+1)))
        header["length_channels"]=int(header["length_channels"]*2)
    if header["bit_mode"]==4:
        header["channels"]=header["channels"][::2]
        header["length_channels"]=int(header["length_channels"]/2)
    return header

def get_data(file_name, items=-1,unpack_fast=False,float=False,byte_delta=0):
    header=get_header(file_name)
    file_data=open(file_name, "r")
    file_data.seek(8+header["header_bytes"]+byte_delta)
    t1=time.time()
    data=numpy.fromfile(file_data, count=items, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(header["bytes_per_packet"]-4))])
    t2=time.time()
    print('took ',t2-t1,' seconds to read raw data on ',file_name)
    file_data.close()
    if header["bit_mode"]==1:
        raw_spectra=data["spectra"].reshape(-1, header["length_channels"]//2)
        if unpack_fast:
            if (float==False):
                print("requested fast unpacking but not float.  Preferring the fast so return will be fast.")
            print('nchannels is ',header["length_channels"])
            print('data shape is ',raw_spectra.shape)
            print(raw_spectra[:5,:5])
            pol0, pol1=unpack_1bit_fast(raw_spectra, header["length_channels"],float=True)
        else:
            pol0, pol1=unpack_1_bit(raw_spectra, header["length_channels"])
    if header["bit_mode"]==2:
        raw_spectra=data["spectra"].reshape(-1, header["length_channels"])
        pol0, pol1=unpack_2_bit(raw_spectra, header["length_channels"])
    if header["bit_mode"]==4:
        raw_spectra=data["spectra"].reshape(-1, header["length_channels"])
        if unpack_fast:
            pol0, pol1=unpack_4bit_fast(raw_spectra, header["length_channels"],float)
        else:
            pol0, pol1=unpack_4_bit(raw_spectra, header["length_channels"])
    # all_spec_num=[]
    # for i in range(header["spectra_per_packet"]):
    #     all_spec_num.append(data["spec_num"]+i)
    # spec_num=numpy.ravel(numpy.column_stack(tuple(all_spec_num)))
    spec_num=data["spec_num"]
    return header, {"spectrum_number":spec_num, "pol0":pol0, "pol1":pol1}
