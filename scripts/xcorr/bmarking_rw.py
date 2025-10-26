import numpy as np
import time
import os
from os import path
from pyuvdata import UVData
import cupy as cp
import helper
import json
from albatros_analysis.src.utils import pfb_utils as pu
from albatros_analysis.src.utils import baseband_data_classes as bdc


nant = 

vis_file = #pyuv stuff

for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
    # start_event.record()
    ts1=time.time()
    for ant_idx in range(nant):
        chunk=chunks[ant_idx]
        start_specnum = start_specnums[ant_idx]
        pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
        pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums']-start_specnum,cp.arange(0,nchan),read_size, nchan)
        # print("continuous pol0 shape", pol0.shape)
        # start_event.record()
        spec0=ipfb.ipfb(ant_idx,0,pol0,thresh=filt_thresh)
        spec1=ipfb.ipfb(ant_idx,1,pol1,thresh=filt_thresh)
        # end_event.record()
        # end_event.synchronize()
        # print("tot ipfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        # start_event.record()
        pol0_new = fpfb.pfb(ant_idx,0, spec0)
        pol1_new = fpfb.pfb(ant_idx,1, spec1)
        # end_event.record()
        # end_event.synchronize()
        # print("tot pfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        # print("PFB returned shape", pol0_new.shape, pol1_new.shape)
        # start_event.record()
        xcorr.load(ant_idx, 0, pol0_new)
        xcorr.load(ant_idx, 1, pol1_new)
        # end_event.record()
        # end_event.synchronize()
        # print("load time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        
    rows = xcorr.xcorr()
    n = len(rows)
    if n > 0:
        for row in rows:
            if rowidx == vis_chunk_size:
                t1=time.time()
                vis_file[:, : , : , vis_file_ptr : vis_file_ptr + vis_chunk_size] = vis
                t2=time.time()
                print("host to disk time", t2-t1)
                rowidx = 0
                vis_file_ptr += vis_chunk_size
            t1=time.time()
            vis[:,:,:,rowidx] = cp.asnumpy(row) #dev to host
            t2=time.time()
            # print("dev to host time", t2-t1)
            rowidx+=1
    ts2=time.time()
    print(f"chunk {chunk_idx}/{nchunks}, chunk time {ts2-ts1:5.3f}")
if rowidx>0:
    vis_file[:, : , : , vis_file_ptr : vis_file_ptr + rowidx] = vis[:,:,:,:rowidx]
vis_file.flush()


