#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <stdio.h>
//nvcc -Xcompiler -fPIC -shared -o libcgemm_batch.so cgemm_batch.cu -lcublas

extern "C"
void cgemm_strided_batched(
    const cuComplex* A,    // ptr to batch of A, shape (M,K,batch), column‑major
    const cuComplex* B,    // ptr to batch of B, shape (N,K,batch), column‑major
    cuComplex*       C,    // ptr to batch of C, shape (M,N,batch), column‑major
    int M, int N, int K,
    int batchCount)
{
    // leading dims in column‑major
    int lda = M, ldb = N, ldc = M;
    // strides between consecutive batches
    long long strideA = (long long)M * K;
    long long strideB = (long long)N * K;
    long long strideC = (long long)M * N;

    // Scalars
    const cuComplex α = make_cuComplex(1.0f, 0.0f);
    const cuComplex β = make_cuComplex(0.0f, 0.0f);

    // cuBLAS handle
    // int vnum;
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return;
    }
    // cublasGetVersion(h, &vnum);
    // printf("CuBLAS version number %d\n", vnum);
    // Perform: C = α·A·Bᴴ + β·C  (batched)
    cublasStatus_t stat = cublasCgemmStridedBatched(
        h,
        CUBLAS_OP_N,      // A not transposed
        CUBLAS_OP_C,      // B conjugate‑transposed
        M,                // #rows of A and C
        N,                // #cols of Bᴴ and C
        K,                // #cols of A == #rows of B
        &α,
        A, lda, strideA,
        B, ldb, strideB,
        &β,
        C, ldc, strideC,
        batchCount);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCgemmStridedBatched failed: %d\n", stat);
    }
    cublasDestroy(h);
}