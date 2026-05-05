extern "C" __global__
void streaming_pfb_kernel(const float* __restrict__ timestream, 
                          const float* __restrict__ window_coeffs, 
                        //   float* __restrict__ history, // The new state buffer
                          float* __restrict__ output, 
                          int nchan, int num_spectra) {
    
    int n = blockIdx.x * blockDim.x + threadIdx.x; 
    if (n >= nchan) return;

    // 1. Fetch Window Coefficients
    float h0 = window_coeffs[0 * nchan + n];
    float h1 = window_coeffs[1 * nchan + n];
    float h2 = window_coeffs[2 * nchan + n];
    float h3 = window_coeffs[3 * nchan + n];

    // 2. Fetch the History (from the previous chunk)
    // x1 is t-1, x2 is t-2, x3 is t-3
    // float x1 = history[0 * nchan + n]; 
    // float x2 = history[1 * nchan + n]; 
    // float x3 = history[2 * nchan + n]; 
    float x0 = 0;
    float x1 = 0;
    float x2 = 0;


    // 3. The Time Loop (Now starts at t=0!)
    for (int t = 0; t < num_spectra; ++t) {
        
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

    // 4. Save the new History for the next kernel launch
    // At the end of the loop, x1 holds the very last sample of this chunk,
    // x2 holds the second-to-last, and x3 holds the third-to-last.
    // history[0 * N + n] = x1;
    // history[1 * N + n] = x2;
    // history[2 * N + n] = x3;
}