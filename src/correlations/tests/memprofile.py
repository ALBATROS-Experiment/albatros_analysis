import os
import psutil
import time
import sys
import gc
import numpy as np
import psutil
import gc
import time
import cupy as cp

print("GPU mem limit is ", cp.get_default_memory_pool().get_limit(), "GiB")  # 1073741824

# Function to print memory info
def print_memory_info(process):
    mem_info = process.memory_full_info()
    rss = mem_info.rss / 1024**2  # Physical memory usage (in MB)
    heap = mem_info.uss / 1024**2  # Unique memory that belongs to this process (in MB)
    print(f"RSS used: {rss:.2f} MB")
    print(f"Heap used: {heap:.2f} MB")
    mempool = cp.get_default_memory_pool()
    gpu_used = mempool.used_bytes() / 1024**2  # Used pool memory in MB
    gpu_total = mempool.total_bytes() / 1024**2  # Total pool memory in MB
    print(f"Mempool memory used: {gpu_used:.4f} MB, Mempool memory avlbl: {gpu_total:.4f} MB")

process = psutil.Process(os.getpid())
# Generator that creates and yields NumPy/CuPy arrays
def array_generator(num_arrays, array_size):
    for _ in range(num_arrays):
        # arr = np.random.randn(array_size, array_size)
        arr = cp.random.randn(array_size, array_size)
        yield {"data":arr}

# Monitor memory before, during, and after iteration
print("Before iterations:")
print_memory_info(process)

# Number of arrays to generate and their size
num_arrays = 10
array_size = 5000  # This creates large arrays (~190 MB each)
arrays=[]
for i, data in enumerate(array_generator(num_arrays, array_size)):
    # arrays.append(data["data"])
    print(f"Iteration {i+1}:")
    print_memory_info(process)
    time.sleep(1)  # Adding a delay to observe memory changes clearly

# Run garbage collection after the iterations to ensure memory cleanup
gc.collect()
data=None #returned arrays are out of scope. causes Cupy to garbage collect.
print("\nAfter iterations and GC:")
print_memory_info(process) #used 0, avlbl ~ size of array or twice it

# if len(sys.argv) < 2:
#     print("pass pid please")
#     sys.exit(1)
# process = psutil.Process(int(sys.argv[1]))
# for i in range(20):
#     print_memory_info(process)
#     time.sleep(2)

