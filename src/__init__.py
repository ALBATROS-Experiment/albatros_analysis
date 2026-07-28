import os, sys
sys.path.insert(0, os.path.expanduser("~"))

USE_GPU = os.getenv('USE_GPU', '0') == '1'

if USE_GPU:
    os.environ["CUPY_CACHE_DIR"] = "/scratch/thomasb/.cupy/kernel_cache"
    try:
        import cupy as xp
    except ImportError:
        print("Cupy not found. Falling back to numpy.")
        import numpy as xp

    try:
        from albatros_analysis.src.utils.pycufft import fft, ifft, rfft, irfft
    except ImportError:
        print("Pycufft not found. Falling back to scipy.")
        from scipy.fft import fft, ifft, rfft, irfft
else:
    import numpy as xp
    from scipy.fft import fft, ifft, rfft, irfft


