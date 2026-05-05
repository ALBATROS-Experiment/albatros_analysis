import cupy as cp
import numpy as np
import time
import sys
import os

# Ensure the local packages are discoverable
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src.correlations import correlations_gpu

def run_single_test(nant, npol, acclen, nchan, nchunks):
    M = nant * npol
    K = acclen
    
    # Input matrix in F-order
    x_in = cp.random.randn(M, K * nchan).astype(cp.complex64)
    x_in = cp.asfortranarray(x_in)
    
    # Output buffer
    out = cp.zeros((M, M, nchan), dtype='complex64', order='F')
    
    # Warm-up
    for _ in range(2):
        correlations_gpu.avg_xcorr_all_ant_gpu(x_in, nant, npol, acclen, nchan, out=out)
    cp.cuda.Device().synchronize()

    ev_start = cp.cuda.Event()
    ev_end = cp.cuda.Event()
    
    ev_start.record()
    for _ in range(nchunks):
        correlations_gpu.avg_xcorr_all_ant_gpu(x_in, nant, npol, acclen, nchan, out=out)
    ev_end.record()
    ev_end.synchronize()
    
    t_ms = cp.cuda.get_elapsed_time(ev_start, ev_end)
    total_sec = t_ms / 1000
    
    # Ops for Complex GEMM: 8 * M * N * K per batch
    ops = nchunks * (8 * M * M * K * nchan)
    tflops = (ops / 1e12) / total_sec
    return tflops

def sweep_benchmark():
    npol = 2
    nchan = 1024 
    nchunks = 50

    nants = [4, 8, 16, 32, 48, 64]
    acclens = [1024, 2048, 4096, 8192]

    results = np.zeros((len(nants), len(acclens)))

    print(f"--- XCorr (CGEMM) Sweep Benchmark (TFLOPS) ---")
    print(f"nchan: {nchan}, npol: {npol}, nchunks: {nchunks}\n")

    # Header row
    header = f"{'Ants - Acc':<10}"
    for acclen in acclens:
        header += f" | {acclen:<8}"
    print(header)
    print("-" * len(header))

    for i, nant in enumerate(nants):
        row_str = f"{nant:<10}"
        for j, acclen in enumerate(acclens):
            mem_needed = (nant * npol) * acclen * nchan * 8
            if mem_needed > 12 * 1024**3:
                row_str += f" | {'SKIPPED':<8}"
                results[i, j] = np.nan
                continue

            try:
                tflops = run_single_test(nant, npol, acclen, nchan, nchunks)
                results[i, j] = tflops
                row_str += f" | {tflops:<8.4f}"
            except Exception as e:
                row_str += f" | {'ERR':<8}"
                results[i, j] = -1

            cp.get_default_memory_pool().free_all_blocks()
        print(row_str)

if __name__ == "__main__":
    sweep_benchmark()

