import sys
sys.path.insert(0,'/home/s/sievers/mohanagr/albatros_analysis')
from src.correlations import baseband_data_classes as bdc
import time
import numpy as np
import os
import psutil
import cupy as cp

print("PROCESS ID", os.getpid())

def print_memory_info(process):
    mem_info = process.memory_full_info()
    rss = mem_info.rss / 1024**2  # Physical memory usage (in MB)
    heap = mem_info.uss / 1024**2  # Unique memory that belongs to this process (in MB)
    print(f"RSS used: {rss:.4f} MB")
    print(f"Heap used: {heap:.4f} MB")
    gpu_memory = cp.get_default_memory_pool().used_bytes() / 1024**2  # GPU memory in MB
    print(f"GPU memory used: {gpu_memory:.2f} MB")

# time.sleep(5)
# files=['/gpfs/fs1/home/s/sievers/mohanagr/albatros_analysis/src/correlations/tests/data/1627202039.raw']
# filenum=0
# idxstart=0
# acclen=1000000
# ant1=bdc.BasebandFileIterator(files,filenum,idxstart,acclen,type='float')

# process = psutil.Process(os.getpid())

# #test type=float and packed
# arrays=[]

# t1=time.time()
# for i,chunk in enumerate(ant1):
#     pass
#     # print(chunk['specnums'].shape[0], "chans=",chunk['pol0'].shape[1], type(chunk['pol0']), chunk['pol0'].dtype)
#     # x=np.mean(chunk['pol0'],axis=0)
#     # arrays.append(chunk['pol0'])
#     # print_memory_info(process)
#     # time.sleep(1)
# t2=time.time()
# sizes=(10**np.linspace(1,9,101)).astype(int)
sizes=(2**np.arange(5,31))
gpu_avg=np.zeros(len(sizes),dtype='float64')
cpu_avg=np.zeros(len(sizes),dtype='float64')
niter=100
for sn,s in enumerate(sizes):
    t1=time.time()
    for i in range(niter):
        x=cp.zeros(s,dtype='complex64')
    t2=time.time()
    gpu_avg[sn]=(t2-t1)/(i+1)
    t1=time.time()
    for i in range(niter):
        y=np.zeros(s,dtype='complex64')
    t2=time.time()
    cpu_avg[sn]=(t2-t1)/(i+1)
    print("done size num", sn, s)

print("sizes", sizes)
print("CPU",cpu_avg)
print("GPU", gpu_avg)

#mem usage stays constant if I don't reference arrays outside