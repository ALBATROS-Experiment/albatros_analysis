import os
USE_GPU = os.getenv('USE_GPU', '0') == '1'
if USE_GPU:
    os.environ["CUPY_CACHE_DIR"] = "/scratch/thomasb/.cupy/kernel_cache"
    os.environ["CUPY_CACHE_DIR"] = "/scratch/thomasb/.cupy/kernel_cache/"
    try:
        import cupy as xp
        print("Using Cupy for GPU computations.")
    except ImportError:
        print("Cupy not found. Falling back to numpy.")
        import numpy as xp
else:
    import numpy as xp

