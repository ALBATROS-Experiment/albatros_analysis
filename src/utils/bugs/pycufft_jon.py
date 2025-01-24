import cupy as cp, os
import ctypes
import numpy as np
print(os.path.dirname(__file__))
mylib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libpycufft_jon.so"))

# All function arguments are pointers or ints
_i = ctypes.c_int
_p = ctypes.c_void_p

class C:
    cufft_r2c_gpu = mylib.cufft_r2c_gpu
    cufft_r2c_gpu.argtypes = (_p,_p,_i,_i,_i)
    #void cufft_r2c_gpu(cufftComplex *out, float *data, int n, int m, int axis)

    cufft_r2c_gpu_wplan = mylib.cufft_r2c_gpu_wplan
    cufft_r2c_gpu_wplan.argtypes = (_p,_p,_i,_i,_i,_p)
    #void cufft_r2c_gpu_wplan(cufftComplex *out, float *data, int n, int m, int axis,cufftHandle *plan)

    cufft_c2r_gpu = mylib.cufft_c2r_gpu
    cufft_c2r_gpu.argtypes = (_p,_p,_i,_i,_i)
    #void cufft_c2r_gpu(float *out, cufftComplex *data, int n, int m, int axis)

    cufft_c2r_gpu_wplan = mylib.cufft_c2r_gpu_wplan
    cufft_c2r_gpu_wplan.argtypes = (_p,_p,_p)
    #void cufft_c2r_gpu_wplan(float  *out, cufftComplex *data, cufftHandle *plan)

    get_plan_r2c = mylib.get_plan_r2c
    get_plan_r2c.argtypes = (_i,_i,_i,_p,_i)
    #void get_plan_r2c(int n, int m, int axis, cufftHandle *plan, int alloc)

    get_plan_c2r = mylib.get_plan_c2r
    get_plan_c2r.argtypes = (_i,_i,_i,_p,_i)
    #get_plan_c2r(int n, int m, int axis, cufftHandle *plan, int alloc)

    get_plan_size = mylib.get_plan_size
    get_plan_size.argtypes = (_p,_p)
    #get_plan_size(cufftHandle *plan, size_t *sz)

    destroy_plan = mylib.destroy_plan
    destroy_plan.argtypes = (_p,)
    #destroy_plan(cufftHandle *plan)

    set_plan_scratch = mylib.set_plan_scratch
    set_plan_scratch.argtypes = (_i,_p)
    #set_plan_scratch(cufftHandle plan,void *buf)

def get_plan_size(plan):
    sz = np.zeros(1,dtype=np.uint64)
    C.get_plan_size(plan.ctypes.data, sz.ctypes.data)
    return sz[0]

def set_plan_scratch(plan, buf):
    C.set_plan_scratch(plan[0], buf.data.ptr)

def get_plan_r2c(n, m, axis=1, alloc=True):
    plan = np.empty(1, dtype=np.int32) #I checked, and sizeof(plan) is 4 bytes
    C.get_plan_r2c(n, m, axis, plan.ctypes.data, alloc)
    return plan

def get_plan_c2r(n, m, axis=1, alloc=True):
    plan = np.empty(1, dtype=np.int32)
    C.get_plan_c2r(n, m, axis, plan.ctypes.data, alloc)
    return plan

def destroy_plan(plan):
    C.destroy_plan(plan.ctypes.data)

def rfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
    # We apparently only support 2d input arrays
    axis = axis % dat.ndim
    if dat.ndim != 2:
        raise ValueError("pycufft.rfft only supports 2d arrays")
    if dat.dtype != np.float32:
        raise ValueError("pycufft.rfft only supports single precision")
    if out is None:
        oshape = dat.shape[:axis] + (dat.shape[axis]//2+1,)+dat.shape[axis+1:]
        out    = cp.empty(oshape, dtype=np.complex64)
    elif out.dtype != np.complex64:
        raise ValueError("pycufft.rfft only supports single precision")
    if plan is None and plan_cache is not None:
        plan = plan_cache.get("r2c", dat.shape, axis)
    if plan is None:
        C.cufft_r2c_gpu(out.data.ptr, dat.data.ptr, dat.shape[0], dat.shape[1], axis)
    else:
        C.cufft_r2c_gpu_wplan(out.data.ptr, dat.data.ptr, dat.shape[0], dat.shape[1], axis, plan.ctypes.data)
    return out

def irfft(dat, out=None, n=None, axis=-1, plan=None, plan_cache=None):
    # We apparently only support 2d input arrays
    axis = axis % dat.ndim
    if dat.ndim != 2:
        raise ValueError("pycufft.rfft only supports 2d arrays")
    if dat.dtype != np.complex64:
        raise ValueError("pycufft.rfft only supports single precision")
    if out is None:
        if n is None: n = 2*(dat.shape[axis]-1)
        oshape = dat.shape[:axis] + (n,)+dat.shape[axis+1:]
        out    = cp.empty(oshape, dtype=np.float32)
    elif out.dtype != np.float32:
        raise ValueError("pycufft.rfft only supports single precision")
    if plan is None and plan_cache is not None:
        plan = plan_cache.get("c2r", out.shape, axis)
    if plan is None:
        C.cufft_c2r_gpu(out.data.ptr, dat.data.ptr, out.shape[0], out.shape[1], axis)
    else:
        C.cufft_c2r_gpu_wplan(out.data.ptr, dat.data.ptr, plan.ctypes.data)
    out/=n
    return out

class PlanCache:
    def __init__(self):
        self.maxsize = -1
        self.scratch = None
        self.plans   = {}
        #self.maxsize = 2600000000
        #self.scratch = cp.empty(self.maxsize, dtype=np.uint8)
    def get(self, kind, shape, axis=1):
        # Get a plan from the cache, or set it up if not present.
        # Reallocates the scratch space if necessary. We assume that
        # get_plan_size and set_plan_scratch are very fast compared to an fft
        tag = "%s_%s_ax%d" % (str(kind), str(shape), axis)
        if tag in self.plans:
            plan = self.plans[tag]
        else:
            if kind == "r2c": fun = get_plan_r2c
            else:             fun = get_plan_c2r
            plan = fun(shape[0], shape[1], alloc=False)
            ######### TAB THE FOLLOWING LEFT #########
            size = get_plan_size(plan)
            if size > self.maxsize:
                self.maxsize = size
                self.scratch = cp.empty(self.maxsize, dtype=np.uint8)
                print("fft scratch increased to %d" % self.maxsize, "for tag", tag)
                set_plan_scratch(plan, self.scratch)
            # Update cache with this plan
            self.plans[tag] = plan
        # And return
        return plan
