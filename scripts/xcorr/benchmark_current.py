import cupy as cp
import numpy as np
import time
import sys
import os

# Ensure the local packages are discoverable
sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu

def benchmark_pipeline(nant=8, npol=2, pfb_size=32768, nchan=200, osamp=64, nchunks=50, bufsize_frac=1.0):
    # Pipeline parameters
    lblock = 4096
    cutsize = 16
    read_size = pfb_size - 2*cutsize
    timestream_size = read_size * lblock
    new_acclen = 1024
    new_nchan = nchan * osamp

    # Initialize Streaming classes
    # ipfb = pu.StreamingIPFB_IQ(nant, npol, np.arange(nchan), nblock=pfb_size, lblock=lblock, cut=cutsize)
    ipfb = pu.StreamingIPFB(nant, npol, np.arange(nchan), nblock=pfb_size, lblock=lblock, cut=cutsize)
    fpfb = pu.StreamingPFB(nant, npol, timestream_size=timestream_size, lblock=ipfb.lblock*osamp, dtype='complex64')
    xcorr = pu.StreamingCorrelator(nant, npol, new_acclen, np.arange(new_nchan), bufsize_frac=bufsize_frac)

    print(f"--- Benchmark Configuration ---")
    print(f"Antennas: {nant}, Polarizations: {npol}")
    print(f"IPFB Size: {pfb_size}, Channels: {nchan}")
    print(f"PFB Oversampling: {osamp}, New Accumulation: {new_acclen}")

    dummy_input = cp.random.rand(read_size, nchan).astype("complex64")
    dummy_input_host = np.random.randn(read_size*100, nchan) #simulate data transferred from each file for each chunk
    # Warm-up Phase
    print("Warming up (FFT planning, library init)...")
    for _ in range(2): # 2 chunks of warm-up
        for a in range(nant):
            _ = cp.asarray(dummy_input_host)
            for p in range(npol):
                spec = ipfb.ipfb(a, p, dummy_input)
                pol_new = fpfb.pfb(a, p, spec)
                if pol_new is not None:
                    xcorr.load(a, p, pol_new)
        _warmup_rows = xcorr.xcorr()
    cp.cuda.Device().synchronize()
    
    print(f"Processing {nchunks} chunks of data...")
    
    # Timing accumulation (in ms)
    t_ipfb_ms = 0
    t_pfb_ms = 0
    t_xcorr_ms = 0
    t_d2h_ms = 0
    t_h2d_ms = 0
    
    ev_start = cp.cuda.Event()
    ev_end = cp.cuda.Event()
    num_xcorr = 0
    
    start_wall = time.perf_counter()
    
    for c in range(nchunks):
        for a in range(nant):
            ev_start.record()
            _ = cp.asarray(dummy_input_host)   
            ev_end.record()
            ev_end.synchronize()
            t_h2d_ms += cp.cuda.get_elapsed_time(ev_start, ev_end)

            for p in range(npol):
                # 1. IPFB Execution
                ev_start.record()
                spec = ipfb.ipfb(a, p, dummy_input)
                ev_end.record()
                ev_end.synchronize()
                t_ipfb_ms += cp.cuda.get_elapsed_time(ev_start, ev_end)

                # 2. PFB Execution
                ev_start.record()
                pol_new = fpfb.pfb(a, p, spec)
                ev_end.record()
                ev_end.synchronize()

                if pol_new is not None:
                    t_pfb_ms += cp.cuda.get_elapsed_time(ev_start, ev_end)
                    xcorr.load(a, p, pol_new)

        # 3. X-Correlation Execution
        ev_start.record()
        rows = xcorr.xcorr()
        ev_end.record()
        ev_end.synchronize()
        
        if len(rows) > 0:
            t_xcorr_ms += cp.cuda.get_elapsed_time(ev_start, ev_end)
            num_xcorr += len(rows)
            
            # 4. Device to Host Transfer (D2H)
            t1 = time.perf_counter()
            for row in rows:
                res = cp.asnumpy(row)
            t2 = time.perf_counter()
            t_d2h_ms += (t2 - t1) * 1000

    end_wall = time.perf_counter()
    total_wall_ms = (end_wall - start_wall) * 1000

    # Metric Calculations
    # --------------------
    # Mem bandwidth
    t_h2d_sec = t_h2d_ms/1000
    t_d2h_sec = t_d2h_ms/1000
    print("host nbytes", dummy_input_host.nbytes)
    h2d_bw = dummy_input_host.nbytes/(t_h2d_sec / (nant * nchunks)) / 1e9
    d2h_bw = row.nbytes/(t_d2h_sec / max(1, num_xcorr)) / 1e9

    # Samples reconstructed per IPFB/PFB call
    samples_per_call = pfb_size * lblock
    total_samples = nchunks * nant * npol * samples_per_call
    
    # Execution times in seconds
    ipfb_sec = t_ipfb_ms / 1000
    pfb_sec = t_pfb_ms / 1000
    xcorr_sec = t_xcorr_ms / 1000
    total_gpu_active_sec = ipfb_sec + pfb_sec + xcorr_sec
    
    # GSps (Giga-samples per second)
    gsps_ipfb = (total_samples / 1e9) / ipfb_sec
    gsps_pfb = (total_samples / 1e9) / pfb_sec
    gsps_total_rechan = (total_samples / 1e9) / (ipfb_sec + pfb_sec)
    
    # TFLOPS (Floating Point Ops)
    # IPFB Ops: nblock * FFT(lblock) + 2 * lblock * FFT(nblock)
    ops_fft_lblock = 5 * lblock * np.log2(lblock)
    ops_fft_nblock = 5 * pfb_size * np.log2(pfb_size)
    ops_ipfb_total = nchunks * nant * npol * ((pfb_size * ops_fft_lblock) + (2 * lblock * ops_fft_nblock))
    tflops_ipfb = (ops_ipfb_total / 1e12) / ipfb_sec
    
    # PFB Ops: (read_size / osamp) * FFT(lblock * osamp)
    lblock_new = lblock * osamp
    ops_fft_pfb = 5 * lblock_new * np.log2(lblock_new)
    ops_pfb_total = nchunks * nant * npol * ((read_size / osamp) * ops_fft_pfb)
    tflops_pfb = (ops_pfb_total / 1e12) / pfb_sec
    
    # Correlator Ops: 8 * M * N * K * nchan_new
    M = nant * npol
    ops_xcorr_total = num_xcorr * (8 * M * M * new_acclen * new_nchan)
    tflops_xcorr = (ops_xcorr_total / 1e12) / xcorr_sec

    gpu_usage_percent = (total_gpu_active_sec * 1000 / total_wall_ms) * 100

    print("\n--- Detailed Results ---")
    print(f"Stage          | Total Time (s) | Latency (ms/call)")
    print(f"---------------|----------------|------------------")
    print(f"IPFB           | {ipfb_sec:14.3f} | {t_ipfb_ms / (nchunks * nant * npol):16.2f}")
    print(f"PFB            | {pfb_sec:14.3f} | {t_pfb_ms / (nchunks * nant * npol):16.2f}")
    print(f"XCorr          | {xcorr_sec:14.3f} | {t_xcorr_ms / max(1, num_xcorr):16.2f}")
    print(f"H2D            | {t_h2d_ms/1000:14.3f} | {t_h2d_ms / (nchunks * nant):16.2f}")
    print(f"D2H            | {t_d2h_ms/1000:14.3f} | {t_d2h_ms / max(1, num_xcorr):16.2f}")
    print(f"Wall Time      | {total_wall_ms/1000:14.3f} |")
    
    print(f"\nGPU Usage: {gpu_usage_percent:6.1f}%")

    print("\n--- Scientific Metrics ---")
    print(f"IPFB Sample Rate:           {gsps_ipfb:8.2f} GSps")
    print(f"PFB Sample Rate:            {gsps_pfb:8.2f} GSps")
    print(f"Total Re-channelization:    {gsps_total_rechan:8.2f} GSps")
    print(f"H2D BW:                     {h2d_bw:8.2f} GB/s")
    print(f"D2H BW:                     {d2h_bw:8.2f} GB/s")
    print("")
    print(f"IPFB Compute:               {tflops_ipfb:8.4f} TFLOPS")
    print(f"PFB Compute:                {tflops_pfb:8.4f} TFLOPS")
    print(f"XCorr Compute:              {tflops_xcorr:8.4f} TFLOPS")
    print(f"Num of. XCorr:              {num_xcorr:8d} ")

if __name__ == "__main__":
    benchmark_pipeline()
