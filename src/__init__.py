
import os
USE_GPU = os.getenv('USE_GPU', '0') == '1'
if USE_GPU:
    os.environ["CUPY_CACHE_DIR"] = "/project/s/sievers/thomasb/.cupy/kernel_cache/"
    try:
        import cupy as xp
    except ImportError:
        print("Cupy not found. Falling back to numpy.")
        import numpy as xp
else:
    import numpy as xp

