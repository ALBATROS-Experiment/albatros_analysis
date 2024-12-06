import ctypes
import sys
import numpy as np
import threading
import cupy as cp
cupy_cache=cp.fft.config.get_plan_cache()
cupy_cache.set_size(0) #disable all cupy caching
mylib=ctypes.cdll.LoadLibrary("/home/s/sievers/mohanagr/cuda_ipfb/libpycufft.so")
r2c = mylib.cufft_r2c_mohan
r2c.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p)
c2r = mylib.cufft_c2r_mohan
c2r.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p)

get_plan_r2c = mylib.get_plan_r2c
get_plan_r2c.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p)
get_plan_c2r = mylib.get_plan_c2r
get_plan_c2r.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p)
get_plan_size = mylib.get_plan_size
get_plan_size.argtypes = (ctypes.c_void_p,ctypes.c_void_p)
set_plan_scratch = mylib.set_plan_scratch
set_plan_scratch.argtypes = (ctypes.c_void_p,ctypes.c_void_p)

class PlanCache():
    def __init__(self,maxsize=4*1024**3):
        self._cache={}
        self.hits=0
        self.misses=0
        self.maxsize=maxsize
        self.cursize=0
        self.plan_func = {'C2R': get_plan_c2r, 'R2C': get_plan_r2c}
    def __getitem__(self,key):
        '''
        Return pointer to a plan. None equivalent to passing a NULL pointer in ctypes.
        '''
        if self.maxsize <= 0:
            return None #no cache will be used when calling FFTs.
        if len(key) < 3:
            raise ValueError("Not enough indices")
        tag = f"{key[0]}_{key[1]}_{key[2]}" #shape[0], shape[1], type (C2R, R2C, C2C), axis (0 or 1)
        # print(tag)
        if tag in self._cache: 
            self.hits+=1
            # print("Found", key, "in cache!!!")
            return ctypes.byref(self._cache[tag]) #if plan already in cache, return that
        else:
            size=ctypes.c_uint64(0)
            plan=ctypes.c_uint32(0)
            self.plan_func[key[2]](key[0], key[1], ctypes.byref(plan), ctypes.byref(size)) #every call creates a new unique plan.
            # size_check = np.empty(1,dtype=np.uint64)
            # get_plan_size(ctypes.byref(plan),size_check.ctypes.data)
            # assert size_check[0]==size.value # this can be removed at some point
            if size.value > self.maxsize:
                raise RuntimeError("Plan cache requires too much memory. Disable caching or use FFT calls with use_cache=False while using this plan.")
                # Raising an error as per CUPY protocol.
                # Don't wanna keep wasting time calling get_plan_*() to get a plan every time we call an FFT whose cache that won't fit.
                # We'll eventually have to fallback to using un-cached implementation anyway, which does the planning.
            if size.value > self.cursize:
                self.cursize = size.value
                self.scratch = cp.empty(self.cursize, dtype="uint8") #work area bound to cache object's lifetime
            set_plan_scratch(ctypes.byref(plan), self.scratch.data.ptr)
            self._cache[tag]=plan
            self.misses+=1
            return ctypes.byref(plan)
    def __repr__(self):
        return "plans cached currently are: " + ", ".join(self._cache.keys()) + "\n" +\
        f"current work area size: {self.cursize} B" + "\n" +\
        f"hits {self.hits}, misses {self.misses}."
    #may be implement cleanup?

# Python import is thread-safe. All threads will use sys.modules after pycufft has been imported once.
# but the cache itself is NOT thread-safe.
# If multiple threads try to modify the cache at the same time,
# it can lead to indeterminate situations.
# However, so long as there's only one interpreter running, 
# it doesn't matter how many modules/sub-modules use pycufft. 
# they will all use the same cache.
pycufft_cache=PlanCache()

def rfft(inp,axis=1,use_cache=True,copy=False):
    if axis!=inp.ndim-1: 
        inp=inp.swapaxes(axis,-1)
    if inp.base is not None or not inp.flags.c_contiguous:
        # print("making a copy of inp")
        inp = inp.copy()
    oshape = inp.shape[:-1] + (inp.shape[-1]//2+1,)
    out = cp.empty(oshape,dtype='complex64')
    n = inp.shape[-1]
    batch = inp.size // n
    plan_ptr = pycufft_cache[batch,n,'R2C'] if use_cache else None
    r2c(out.data.ptr,inp.data.ptr,batch,n,plan_ptr)
    if axis!=inp.ndim-1: 
        out=out.swapaxes(axis,-1)
        if copy and out.base is not None:
            out=out.copy()
    return out

def irfft(inp,axis=1,use_cache=True,copy=False):
    if axis!=inp.ndim-1: 
        inp=inp.swapaxes(axis,-1)
    if inp.base is not None or not inp.flags.c_contiguous:
        # print("making a copy of inp")
        inp = inp.copy()
    # print(inp.shape)
    oshape = inp.shape[:-1] + (2*(inp.shape[-1]-1),)
    out = cp.empty(oshape,dtype='float32')
    n = out.shape[-1]
    batch = out.size // n
    plan_ptr = pycufft_cache[batch,n,'C2R'] if use_cache else None
    c2r(out.data.ptr,inp.data.ptr,batch, n, plan_ptr)
    out/=n
    if axis!=inp.ndim-1: 
        out=out.swapaxes(axis,-1)
        if copy and out.base is not None:
            # print("making output copy")
            out=out.copy()
    return out

def test(cache=False):
    inp = cp.random.randn(100,80)
    inp=inp.astype("float32")
    axis=1
    print("testing rfft axis", axis)
    tru=cp.fft.rfft(inp,axis=axis)
    out=rfft(inp,use_cache=cache)
    axis=0
    assert cp.allclose(out,tru,rtol=1e-15,atol=1e-15)
    print("testing rfft axis", axis)
    tru=cp.fft.rfft(inp,axis=axis)
    out=rfft(inp,axis=axis,use_cache=cache)
    assert cp.allclose(out,tru,rtol=1e-15,atol=1e-15)

    nc=17 #no roundoff errors if 2*(nc-1) is a power of 2
    nr=100
    inp = cp.zeros((nr,nc),dtype='complex64')
    xx=cp.random.randn(nr,nc)+1j*cp.random.randn(nr,nc)
    inp[:,:]=xx 
    # print(inp)
    axis=1
    print("testing irfft axis", axis)
    tru=cp.fft.irfft(inp,axis=axis)
    out=irfft(inp,axis=axis,use_cache=cache)
    # print(tru,out)
    assert cp.allclose(tru,out,atol=1.5e-7,rtol=1e-6)

    nr=41 #no roundoff errors if 2*(nc-1) is a power of 2
    nc=100
    inp = cp.zeros((nr,nc),dtype='complex64')
    xx=cp.random.randn(nr,nc)+1j*cp.random.randn(nr,nc)
    inp[:,:]=xx 
    # print(inp)
    axis=0
    print("testing irfft axis", axis)
    tru=cp.fft.irfft(inp,axis=axis)
    out=irfft(inp,axis=axis,use_cache=cache)
    print(out.shape)
    assert cp.allclose(tru,out,atol=1.5e-7,rtol=1e-6)

if __name__=='__main__':

    # Digging into the errors a bit to see what tolerances to use
    #------------------------------------------------------------
    # aa=cp.ravel(tru)
    # bb=cp.ravel(out)
    # err=aa-bb
    # mm=cp.argmax(cp.abs(err))
    # print(cp.argmax(cp.abs(err)), cp.max(cp.abs(err))) #got 1.1920929e-07, the smallest unit of roundoff in floats
    # print(aa[mm], bb[mm])
    # rtol=1e-6
    # idx=cp.where(cp.abs(err)>(1e-8+rtol*cp.abs(bb)))
    # print(aa[idx])
    # print(bb[idx])
    # print(np.abs(aa[idx]-bb[idx]))
    # print("limit is", 1e-8+rtol*cp.abs(bb)[idx])
    #-------------------------------------------------------------
    test()
    assert pycufft_cache.hits==0
    assert pycufft_cache.misses==0
    test(cache=True)
    assert pycufft_cache.hits==0
    assert pycufft_cache.misses==4
    test(cache=True)
    assert pycufft_cache.hits==4
    assert pycufft_cache.misses==4
    print(pycufft_cache)