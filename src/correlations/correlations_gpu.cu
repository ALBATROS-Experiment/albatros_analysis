#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuComplex.h>

//nvcc -Xcompiler -fPIC -o lib_sgemm.so cuda_sgemm.cu -shared -lcublas
extern "C"
{
void Sxc(float * dev_c, float * dev_a, float * dev_b, int m, int n, int k, int nbatch)
{
    // each sub A is m x k
    // each sub B is k x n
    // cublasOperation_t op;
    // printf("optypes are %d %d %d");
    cublasStatus_t stat;
    float alpha = 1, beta = 0;
    // float alpha = make_float(1, 0);
    // float beta = make_float(0, 0);
    // long long int stride_A = m*k, stride_B = n*k, stride_C =  m*n;
    long long int stride_A = m*k, stride_B = n*k, stride_C =  m*n;
    // long long int stride_A = 1, stride_B = 1, stride_C =  1;
    cublasHandle_t h;
	if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS)
        fprintf(stderr,"Error creating cublas handle\n");
    stat = cublasSgemmStridedBatched(
		h,
		CUBLAS_OP_N, // not transposed
		CUBLAS_OP_T,
		m, n, k, // 4x3 * 3x2 matrix multiply
		&alpha,
		dev_a, // i.e. dev_a is the beginning of column 1 of a[0],
		m,     //      dev_a+4 is the beginning of column 2 of a[0]
		stride_A, // i.e. a[0] takes up 4 x 3 = 12 floats of memory
		dev_b, // i.e. dev_b is the beginning of column 1 of b[0],
		n,     //      dev_b+3 is the beginning of column 2 of b[0]
		stride_B, // i.e. b[0] takes up 3 x 2 = 6 floats of memory
		&beta,
		dev_c, // i.e. dev_c is the beginning of column 1 of c[0],
		m,     //      dev_c+4 is the beginning of column 2 of c[0]
		stride_C, // i.e. c[0] takes up 4 x 2 = 8 floats
		nbatch      // multiply 2 matrices
	);
    if(stat!=CUBLAS_STATUS_SUCCESS)
        fprintf(stderr,"Error executing GEMM\n");
    cublasDestroy(h);
    printf("finished....\n");
}

void Cxc(cuComplex * dev_c, cuComplex * dev_a, cuComplex * dev_b, int m, int n, int k, int nbatch)
{
    // each sub A is m x k
    // each sub B is k x n
    // cublasOperation_t op;
    // printf("optypes are %d %d %d");
    cublasStatus_t stat;
    cudaError_t cudaError;

    cudaError = cudaGetLastError();

    if( cudaError != cudaSuccess )
    {
        fprintf(stderr, "CUDA Runtime API Error reported Pre GEMM: %s\n", cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
    cuComplex alpha = make_cuComplex(1,0);
    cuComplex beta = make_cuComplex(0,0);
    cuComplex test = make_cuComplex(0,0);
    // float alpha = make_float(1, 0);
    // float beta = make_float(0, 0);
    // long long int stride_A = m*k, stride_B = n*k, stride_C =  m*n;
    long long int stride_A = m*k, stride_B = n*k, stride_C =  m*n;
    printf("stride A %d stride B %d, stride C %d\n", stride_A, stride_B, stride_C );
    printf("m %d, n %d, k %d\n", m, n, k);
    // long long int stride_A = 1, stride_B = 1, stride_C =  1;
    // for (int ii = 0; ii < nbatch; ii++)
    // {
    //     for (int jj = 0; jj < k; jj++) //k cols
    //     {
    //         for (int kk = 0; kk < m; kk++) //m rows
    //         {
    //             // printf("A = %f", cuCrealf(dev_a[kk + jj*m + stride_A * ii]));
    //             test = dev_a[kk + jj*m + stride_A * ii] ;
    //         }
            
    //     }
        
    // }
    cublasHandle_t h;
    int vnum;
	if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS)
        fprintf(stderr,"Error creating cublas handle\n");
    cublasGetVersion(h, &vnum);
    printf("CuBLAS version number %d", vnum);
    stat = cublasCgemmStridedBatched(
		h,
		CUBLAS_OP_N, // not transposed
		CUBLAS_OP_C, //hermitian transpose
		m, n, k, // 4x3 * 3x2 matrix multiply
		&alpha,
		dev_a, // i.e. dev_a is the beginning of column 1 of a[0],
		m,     //      dev_a+4 is the beginning of column 2 of a[0]
		stride_A, // i.e. a[0] takes up 4 x 3 = 12 floats of memory
		dev_b, // i.e. dev_b is the beginning of column 1 of b[0],
		n,     //      dev_b+3 is the beginning of column 2 of b[0]
		stride_B, // i.e. b[0] takes up 3 x 2 = 6 floats of memory
		&beta,
		dev_c, // i.e. dev_c is the beginning of column 1 of c[0],
		m,     //      dev_c+4 is the beginning of column 2 of c[0]
		stride_C, // i.e. c[0] takes up 4 x 2 = 8 floats
		nbatch      // multiply 2 matrices
	);
    if( cudaError != cudaSuccess )
    {
        fprintf(stderr, "CUDA Runtime API Error reported post GEMM: %s\n", cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
    if(stat!=CUBLAS_STATUS_SUCCESS)
        fprintf(stderr,"Error executing GEMM\n");
    cublasDestroy(h);
    if( cudaError != cudaSuccess )
    {
        fprintf(stderr, "CUDA Runtime API Error reported post destroy: %s\n", cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
    printf("finished....\n");
}
}