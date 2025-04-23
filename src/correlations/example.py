import cupy as cp
import ctypes

# 1) Load the shared library
lib = ctypes.CDLL('./libcgemm_batch.so')

# 2) Declare the C function signature
lib.cgemm_strided_batched.argtypes = [
    ctypes.c_void_p,  # A.ptr
    ctypes.c_void_p,  # B.ptr
    ctypes.c_void_p,  # C.ptr
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
    ctypes.c_int      # batchCount
]
lib.cgemm_strided_batched.restype = None

# 3) Dimensions
M, K, N = 64, 100, 64
batchCount = 10

# 4) Allocate three 3-D arrays in column-major order, with shape (M,K,batch)
#    so that the **slowest**‑moving index is the batch dimension.
A = cp.ones((M, K, batchCount), dtype=cp.complex64, order='F')
A[:] = cp.random.randn(M*K*batchCount).reshape(A.shape)
B = cp.ones((N, K, batchCount), dtype=cp.complex64, order='F')
B[:] = cp.random.randn(N*K*batchCount).reshape(B.shape)
C = cp.empty        ((M, N, batchCount), dtype=cp.complex64, order='F')

# 5) Call the CUBLAS wrapper
lib.cgemm_strided_batched(
    ctypes.c_void_p(A.data.ptr),
    ctypes.c_void_p(B.data.ptr),
    ctypes.c_void_p(C.data.ptr),
    M, N, K, batchCount
)
# 6) Verify against a pure‑Python (CuPy) result for the first batch
#    (cupy.einsum does A·Bᴴ for each batch)
C_ref0 = A[:, :, 0] @ B[:, :, 0].conj().T

print("max abs diff on batch 0:", 
      float(cp.max(cp.abs(C_ref0 - C[:, :, 0]))))