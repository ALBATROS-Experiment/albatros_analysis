import cupy as cp
import os
import sys
sys.path.insert(0,os.path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
from albatros_analysis.src.utils import pycufft
import numpy as np
cp.random.seed(42)

def cupy_pfb(timestream, win, out=None, lblock=4096, ntap=4):
    nblock = timestream.size // lblock - (ntap - 1)
    timestream=timestream.reshape(-1,lblock)
    if out is not None:
        assert out.shape == (nblock, nchan)
    mywin=win.reshape(ntap,lblock)
    y = timestream * mywin[:,cp.newaxis,:]
    y=y[0,:nblock,:]+y[1,1:nblock+1,:]+y[2,2:nblock+2,:]+y[3,3:nblock+3,:]
    # out=y
    out=pycufft.rfft(y,axis=1)
    # out=cp.fft.rfft(y,axis=1)
    # print(pycufft.pycufft_cache)
    return out

def cupy_pfb_dumb(timestream, win, out=None, lblock=4096, ntap=4):
    nblock = timestream.size // lblock - (ntap - 1)
    timestream=timestream.reshape(-1,lblock)
    if out is not None:
        assert out.shape == (nblock, nchan)
    out = cp.empty((nblock, lblock), dtype=timestream.dtype)
    mywin = win.reshape(ntap,lblock)
    for i in range(nblock):
        out[i] = cp.sum(timestream[i:i+ntap, :] * mywin,axis=0)
    out=pycufft.rfft(y,axis=1)
    # out=cp.fft.rfft(y,axis=1)
    # print(pycufft.pycufft_cache)
    return out

def kernel_pfb(timestream, win, lblock=4096, ntap=4, threads_per_block=256):
    nblock = timestream.size // lblock
    timestream=timestream.reshape(-1,lblock)
    y = cp.empty((nblock, lblock), dtype=timestream.dtype)
    blocks_per_grid = (lblock + threads_per_block - 1) // threads_per_block
    pfb_kernel((blocks_per_grid,), (threads_per_block,), (timestream, win, y, lblock, nblock))
    # out=y
    # out=cp.fft.rfft(y,axis=1)
    out=pycufft.rfft(y[3:,],axis=1)
    return out


pfb_kernel = cp.RawKernel(r"""
extern "C" __global__
void streaming_pfb_kernel(const float* __restrict__ timestream, 
                          const float* __restrict__ window_coeffs, 
                          float* __restrict__ output, 
                          long long nchan, long long num_spectra) {

    long long n = blockIdx.x * blockDim.x + threadIdx.x; 
    if (n >= nchan) return;

    // 1. Fetch Window Coefficients
    float h0 = window_coeffs[0 * nchan + n];
    float h1 = window_coeffs[1 * nchan + n];
    float h2 = window_coeffs[2 * nchan + n];
    float h3 = window_coeffs[3 * nchan + n];

    float x0 = 0;
    float x1 = 0;
    float x2 = 0;

    // 3. The Time Loop (Now starts at t=0!)
    for (long long t = 0; t < num_spectra; ++t) {
        
        // Fetch the brand new time sample
        float x3 = timestream[t * nchan + n]; 

        // Compute the 4-tap sum
        float y = (x0 * h0) + (x1 * h1) + (x2 * h2) + (x3 * h3);

        // Write output (No offset needed, 1-to-1 input to output)
        output[t * nchan + n] = y;

        // Shift the registers
        
        x0 = x1;
        x1 = x2;
        x2 = x3;

    }

}""", 'streaming_pfb_kernel')

def run_test(ts, win, lblock, ntap):
    x1 = cupy_pfb(ts, win, lblock=lblock, ntap=ntap)
    x2 = kernel_pfb(ts, win, lblock=lblock, ntap=ntap)
    
    # The kernel implementation in this file has a 3-block offset 
    # compared to the cupy_pfb implementation based on the original code's x2[3:,:]
    # actual = x2[3:, :]
    actual = x2[:, :]
    expected = x1
    
    print(f"\n--- Testing Timestream (size {ts.size}) ---")
    print(f"Allclose (1e-5, 1e-8): {cp.allclose(actual, expected, atol=1e-5, rtol=1e-8)}")
    
    nblock, nchan = expected.shape
    
    for part in ['real', 'imag']:
        a_p = getattr(actual, part)
        e_p = getattr(expected, part)
        abs_err = cp.abs(a_p - e_p)
        
        # Max Absolute Error
        idx_abs = cp.argmax(abs_err)
        b_abs, c_abs = idx_abs // nchan, idx_abs % nchan
        
        # Max Relative Error
        mask = cp.abs(e_p) > 1e-10
        if cp.any(mask):
            rel_err_vals = abs_err[mask] / cp.abs(e_p[mask])
            idx_rel_masked = cp.argmax(rel_err_vals)
            # Map back to original indices to get block/channel
            flat_indices = cp.where(mask.ravel())[0]
            idx_rel = flat_indices[idx_rel_masked]
            b_rel, c_rel = idx_rel // nchan, idx_rel % nchan
            max_rel = rel_err_vals[idx_rel_masked]
        else:
            b_rel, c_rel, max_rel = -1, -1, 0
            
        print(f"  [{part:4}] Max Abs Error: {cp.max(abs_err):.2e} at block {b_abs:2}, chan {c_abs}", a_p[b_abs, c_abs], e_p[b_abs, c_abs], expected[b_abs, c_abs])
        if b_rel != -1:
            print(f"  [{part:4}] Max Rel Error: {float(max_rel):.2e} at block {b_rel:2}, chan {c_rel}", a_p[b_rel, c_rel], e_p[b_rel, c_rel], expected[b_rel, c_rel])
        else:
            print(f"  [{part:4}] Max Rel Error: N/A (all zeros)")

ntap=4
lblock=4096*64
win = cp.asarray(pu.sinc_hamming(ntap,lblock),dtype='float32')

# Define test cases
ts_ones = cp.ones(20*lblock).astype('float32')
ts_delta = cp.zeros(20*lblock).astype('float32')
ts_delta[0] = 1.
ts_random = cp.random.randn(20*lblock).astype('float32')

# Run modular tests
run_test(ts_ones, win, lblock, ntap)
run_test(ts_delta, win, lblock, ntap)
run_test(ts_random, win, lblock, ntap)

# Speed test
ts_large = cp.random.randn(32768*4096).astype('float32')
niter = 100
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

print(f"\nBenchmarking with 4-tap PFB with block length {lblock}...")

# cupy_pfb timing
times = []
for i in range(niter):
    start_event.record()
    _ = cupy_pfb(ts_large, win, lblock=lblock, ntap=ntap)
    end_event.record()
    end_event.synchronize()
    times.append(cp.cuda.get_elapsed_time(start_event, end_event)/1000)
median_cupy = np.median(times)
print(f"cupy_pfb   | Latency: {median_cupy:.4f} s | Throughput: {ts_large.size/median_cupy/1e9:.2f} GSPS")

# kernel_pfb timing
times = []
for i in range(niter):
    start_event.record()
    _ = kernel_pfb(ts_large, win, lblock=lblock, ntap=ntap, threads_per_block=64)
    end_event.record()
    end_event.synchronize()
    times.append(cp.cuda.get_elapsed_time(start_event, end_event)/1000)
median_kernel = np.median(times)
print(f"kernel_pfb | Latency: {median_kernel:.4f} s | Throughput: {ts_large.size/median_kernel/1e9:.2f} GSPS")