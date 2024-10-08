
import os
use_gpu = os.getenv('USE_GPU', '0') == '1'
if use_gpu:
    os.environ["CUPY_CACHE_DIR"] = "/project/s/sievers/mohan/.cupy/kernel_cache/"
    try:
        import cupy as xp
    except ImportError:
        print("Cupy not found. Falling back to numpy.")
        import numpy as xp
else:
    import numpy as xp

