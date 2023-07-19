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
mylib.avg_autocorr_4bit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
mylib.xcorr_4bit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
mylib.avg_xcorr_4bit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
mylib.avg_xcorr_4bit_2ant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,\
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int,ctypes.c_int,ctypes.c_int]
autocorr_4bit_c = mylib.autocorr_4bit
avg_autocorr_4bit_c = mylib.avg_autocorr_4bit
xcorr_4bit_c = mylib.xcorr_4bit
avg_xcorr_4bit_c = mylib.avg_xcorr_4bit
avg_xcorr_4bit_2ant_c = mylib.avg_xcorr_4bit_2ant

mylib.avg_xcorr_1bit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32]
avg_xcorr_1bit_c = mylib.avg_xcorr_1bit

mylib.avg_xcorr_1bit_vanvleck.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, \
    ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32]
avg_xcorr_1bit_vanvleck_c = mylib.avg_xcorr_1bit_vanvleck

mylib.avg_xcorr_1bit_vanvleck_2ant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,ctypes.c_void_p,\
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
avg_xcorr_1bit_vanvleck_2ant_c = mylib.avg_xcorr_1bit_vanvleck_2ant


def autocorr_4bit(pol):

    data = pol.copy()
    print(data.shape)
    corr = np.zeros(data.shape,dtype='uint8',order='c') # ncols = nchan for 4 bit
    t1=time.time()
    autocorr_4bit_c(data.ctypes.data, corr.ctypes.data, data.shape[0], data.shape[1])
    t2=time.time()
    print(f"time taken for corr {t2-t1:5.3f}s")
    return corr

def avg_autocorr_4bit(data, specnums):

    # print("data being passed from python is", data)
    nrows = len(specnums)
    print("NROWS", nrows)
    # x=np.sum(data,axis=1)
    # nn=np.where(x==0)[0][0]
    # print(nn)
    # print("DATA FROM PYTHON")
    # print(data)
    corr = np.empty(data.shape[1],dtype='int64',order='c') #will be put in float64 in frontend script 
    if(nrows==0):
        print("empty block")
        corr = np.nan
        return corr
    t1=time.time()
    avg_autocorr_4bit_c(data.ctypes.data, corr.ctypes.data, nrows, data.shape[1])
    t2=time.time()
    # print(corr)
    # print("last element from python", data[-1][-1])

    print(f"time taken for avg_corr {t2-t1:5.3f}s")
    return corr/nrows

def xcorr_4bit(data0, data1):
    assert(data0.shape[1]==data1.shape[1])
    assert(data0.shape[0]==data1.shape[0])
    xcorr = np.empty(data0.shape,dtype='complex64',order='c')
    t1=time.time()
    xcorr_4bit_c(data0.ctypes.data, data1.ctypes.data, xcorr.ctypes.data, data0.shape[0], data0.shape[1])
    t2=time.time()
    print(f"time taken for avg_xcorr {t2-t1:5.3f}s")
    return xcorr

def avg_xcorr_4bit(data0, data1, specnums):

    assert(data0.shape[1]==data1.shape[1])
    assert(data0.shape[0]==data1.shape[0])
    nrows = len(specnums)
    # xcorr = np.zeros(data0.shape[1],dtype='complex64',order='c')
    xcorr = np.empty(data0.shape[1],dtype='complex64',order='c')
    if(nrows==0):
        print("empty block")
        xcorr = np.nan
        return xcorr
    t1=time.time()
    avg_xcorr_4bit_c(data0.ctypes.data,data1.ctypes.data, xcorr.ctypes.data, nrows, data0.shape[1])
    t2=time.time()
    print(f"time taken for avg_xcorr {t2-t1:5.3f}s")
    return xcorr/nrows

def avg_xcorr_4bit_2ant(data0, data1, specnum0, specnum1, start_idx0, start_idx1):
    
    assert(data0.shape[1]==data1.shape[1])
    xcorr = np.empty(data0.shape[1],dtype='complex64',order='c')
    if(len(specnum0)==0 or len(specnum1)==0):
        xcorr=np.nan
        return xcorr
    # print("Start idx recieved in python", start_idx0, start_idx1)
    # print(specnum0.shape, specnum1.shape)
    # print("First specnums", specnum0[0],specnum1[0])
    # print(specnum0-start_idx0, specnum1-start_idx1)
    t1=time.time()
    row_count = avg_xcorr_4bit_2ant_c(data0.ctypes.data,data1.ctypes.data, xcorr.ctypes.data, specnum0.ctypes.data, specnum1.ctypes.data,\
        start_idx0, start_idx1, len(specnum0), len(specnum1), data0.shape[1])
    t2=time.time()
    print(f"time taken for avg_xcorr {t2-t1:5.3f}s")
    print("ROW COUNT IS ", row_count)
    if(row_count==0):
        xcorr=np.nan
        return xcorr
    return xcorr/row_count

def avg_xcorr_1bit(data0, data1, specnums, nchannels):

    #nchannels = num of channels contained in packed pol0/pol1 data
    assert(data0.shape[0]==data1.shape[0])
    assert(data0.shape[1]==data1.shape[1])
    nrows= len(specnums)
    print("Input shape is", nrows)
    # xcorr = np.zeros(data0.shape[1],dtype='complex64',order='c')
    xcorr = np.empty(nchannels,dtype='complex64',order='c')
    if(nrows==0):
        xcorr=np.nan
        return xcorr
    t1=time.time()
    avg_xcorr_1bit_c(data0.ctypes.data,data1.ctypes.data, xcorr.ctypes.data, nchannels, nrows, data0.shape[1])
    t2=time.time()
    print(f"time taken for avg_xcorr {t2-t1:5.3f}s")
    return xcorr/nrows

def avg_xcorr_1bit_vanvleck(data0, data1, specnums, nchannels):

    #nchannels = num of channels contained in packed pol0/pol1 data
    assert(data0.shape[0]==data1.shape[0])
    assert(data0.shape[1]==data1.shape[1])
    nrows= len(specnums)
    print("Input shape is", nrows)
    # xcorr = np.zeros(data0.shape[1],dtype='complex64',order='c')
    R0 = np.empty(nchannels,dtype='float32',order='c')
    R1 = np.empty(nchannels,dtype='float32',order='c')
    IM0 = np.empty(nchannels,dtype='float32',order='c')
    IM1 = np.empty(nchannels,dtype='float32',order='c')
    if(nrows==0):
        R0[:]=np.nan
        R1[:]=np.nan
        IM0[:]=np.nan
        IM1[:]=np.nan
        return [R0,R1,IM0,IM1]
    t1=time.time()
    avg_xcorr_1bit_vanvleck_c(data0.ctypes.data,data1.ctypes.data, R0.ctypes.data, R1.ctypes.data, IM0.ctypes.data, IM1.ctypes.data,\
        nchannels, nrows, data0.shape[1])
    t2=time.time()
    print(f"time taken for avg_xcorr {t2-t1:5.3f}s")
    return [R0/nrows, R1/nrows, IM0/nrows, IM1/nrows]

def avg_xcorr_1bit_vanvleck_2ant(data0, data1, nchannels, specnum0, specnum1, idxstart0, idxstart1):

    #nchannels = num of channels contained in packed pol0/pol1 data
    assert(data0.shape[0]==data1.shape[0])
    assert(data0.shape[1]==data1.shape[1])
    print("Input shapes are", len(specnum0), len(specnum1))
    # xcorr = np.zeros(data0.shape[1],dtype='complex64',order='c')
    R0 = np.empty(nchannels,dtype='float32',order='c')
    R1 = np.empty(nchannels,dtype='float32',order='c')
    IM0 = np.empty(nchannels,dtype='float32',order='c')
    IM1 = np.empty(nchannels,dtype='float32',order='c')
    t1=time.time()
    if(len(specnum0)==0 or len(specnum1)==0):
        R0[:]=np.nan
        R1[:]=np.nan
        IM0[:]=np.nan
        IM1[:]=np.nan
        return [R0,R1,IM0,IM1]
    rowcount= avg_xcorr_1bit_vanvleck_2ant_c(data0.ctypes.data,data1.ctypes.data, R0.ctypes.data, R1.ctypes.data, IM0.ctypes.data, IM1.ctypes.data,\
        specnum0.ctypes.data, specnum1.ctypes.data, idxstart0, idxstart1, len(specnum0), len(specnum1), data0.shape[1], nchannels)
    if(rowcount==0):
        R0[:]=np.nan
        R1[:]=np.nan
        IM0[:]=np.nan
        IM1[:]=np.nan
        return [R0,R1,IM0,IM1]
    t2=time.time()
    print(f"time taken for avg_xcorr {t2-t1:5.3f}s")
    return [R0/rowcount, R1/rowcount, IM0/rowcount, IM1/rowcount]





