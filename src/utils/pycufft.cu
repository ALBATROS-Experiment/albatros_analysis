//nvcc -o libpycufft.so pycufft.cu -shared -lcufft -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
/*--------------------------------------------------------------------------------*/
const char* _cufftGetErrorEnum( cufftResult_t error )
{
    switch ( error )
    {
        case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
        return "cuFFT was passed an invalid plan handle\n";

        case CUFFT_ALLOC_FAILED:
        return "cuFFT failed to allocate GPU or CPU memory\n";

        // No longer used
        case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE\n";

        case CUFFT_INVALID_VALUE:
        return "User specified an invalid pointer or parameter\n";

        case CUFFT_INTERNAL_ERROR:
        return "Driver or internal cuFFT library error\n";

        case CUFFT_EXEC_FAILED:
        return "Failed to execute an FFT on the GPU\n";

        case CUFFT_SETUP_FAILED:
        return "The cuFFT library failed to initialize\n";

        case CUFFT_INVALID_SIZE:
        return "User specified an invalid transform size\n";

        // No longer used
        case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA\n";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "Missing parameters in call\n";

        case CUFFT_INVALID_DEVICE:
        return "Execution of a plan was on different GPU than plan creation\n";

        case CUFFT_PARSE_ERROR:
        return "Internal plan database error\n";

        case CUFFT_NO_WORKSPACE:
        return "No workspace has been provided prior to plan execution\n";

        case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED\n";

        case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR\n";
    }

    return "<unknown>";
}

extern "C"{
void cufft_r2c_mohan(cufftComplex *out, float * data, int nrows, int ncols, cufftHandle * plan_ptr)
{
    // nrows = rows in input data; ncols = columns in input data.
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaDeviceSynchronize();
    // cudaEventRecord(start, 0);
    cudaError_t cudaError;

        cudaError = cudaGetLastError();

        if( cudaError != cudaSuccess )

        {

        fprintf(stderr, "CUDA Runtime API Error reported : %s\n", cudaGetErrorString(cudaError));

        exit(EXIT_FAILURE);

        }
    if (plan_ptr)
    {
        //if a plan is available execute and move on
        // printf("Using user's plan r2c\n");
        // cudaEventRecord(start, 0);
        cufftResult_t error;
        error=cufftExecR2C(*plan_ptr,data,out);
        if (error!=CUFFT_SUCCESS)
        {
            fprintf(stderr,"R2C (%d, %d) FAILED\n", nrows, ncols);
            fprintf(stderr,_cufftGetErrorEnum(error));
            exit(EXIT_FAILURE);
        }
        // cudaDeviceSynchronize();
    }
    else
    {
        cufftHandle plan;
        if (cufftPlan1d(&plan,ncols,CUFFT_R2C,nrows)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error planning dft\n");
        if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft manual r2c\n");
        // cudaDeviceSynchronize();
        if (cufftDestroy(plan)!= CUFFT_SUCCESS)
            fprintf(stderr,"Error destroying plan.\n");
    }
    // cudaDeviceSynchronize();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("r2c took %12.4g\n",milliseconds);
}
void cufft_c2c(cufftComplex *out, cufftComplex * data, int nrows, int ncols, int direction, cufftHandle * plan_ptr)
{
    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if( cudaError != cudaSuccess )
    {
        fprintf(stderr, "CUDA Runtime API Error reported : %s\n", cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
    direction = (direction == -1) ? CUFFT_FORWARD : CUFFT_INVERSE;
    if (plan_ptr)
    {
        //if a plan is available execute and move on
        // printf("Using user's plan r2c\n");
        // cudaEventRecord(start, 0);
        cufftResult_t error;
        error=cufftExecC2C(*plan_ptr,data,out,direction);
        if (error!=CUFFT_SUCCESS)
        {
            fprintf(stderr,"C2C (%d, %d) FAILED\n", nrows, ncols);
            fprintf(stderr,_cufftGetErrorEnum(error));
            exit(EXIT_FAILURE);
        }
        // cudaDeviceSynchronize();
    }
    else
    {
        cufftHandle plan;
        if (cufftPlan1d(&plan,ncols,CUFFT_C2C,nrows)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error planning dft\n");
        if (cufftExecC2C(plan,data,out,direction)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft manual r2c\n");
        // cudaDeviceSynchronize();
        if (cufftDestroy(plan)!= CUFFT_SUCCESS)
            fprintf(stderr,"Error destroying plan.\n");
    }
}
void cufft_c2r_mohan(float *out, cufftComplex * data, int nrows, int ncols, cufftHandle * plan_ptr)
{
    // nrows = rows in output data; ncols = columns in output data.
    // if (plan==NULL)
    // {
    //     printf("plan is null\n");
    //     printf("nrows = %d ncols = %d\n", nrows, ncols);
    // }
    cudaError_t cudaError;

        cudaError = cudaGetLastError();

        if( cudaError != cudaSuccess )

        {

        fprintf(stderr, "CUDA Runtime API Error reported : %s\n", cudaGetErrorString(cudaError));

        exit(EXIT_FAILURE);

        }
    if (plan_ptr)
    {
        // printf("Using user's plan c2r\n");
        //if a plan is available execute and move on
        if (cufftExecC2R(*plan_ptr,data,out)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft c2r\n");
        // cudaDeviceSynchronize();
    }
    else
    {  
        cufftHandle plan;
        if (cufftPlan1d(&plan,ncols,CUFFT_C2R,nrows)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error planning dft\n");
        if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft manual c2r\n");
        // cudaDeviceSynchronize();
        if (cufftDestroy(plan)!= CUFFT_SUCCESS)
            fprintf(stderr,"Error destroying plan.\n");
    }
}
/*--------------------------------------------------------------------------------*/
void get_plan_r2c(int nrows, int ncols, cufftHandle *plan, size_t * work_size)
{
    // nrows, ncols of input
    // Initialize an empty plan.
    cudaError_t cudaError;

    cudaError = cudaGetLastError();

    if( cudaError != cudaSuccess )

    {

    fprintf(stderr, "CUDA Runtime API Error reported : %s\n", cudaGetErrorString(cudaError));

    exit(EXIT_FAILURE);

    }
    cufftCreate(plan);
    // Turn off auto-allocation by default. This must be done before the
    // plan is actually created, so we can't use the shortcut cufftPlan1d
    cufftResult_t error;
    error=cufftSetAutoAllocation(*plan, 0);
    if(error!=CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT setAutoAllocation returned %d", error);
    }
    // We can finally actually set up the plan
    printf("axis 1, get plan r2c\n");
    if (cufftMakePlan1d(*plan, ncols, CUFFT_R2C, nrows, work_size)!=CUFFT_SUCCESS)
        fprintf(stderr,"Error planning dft.\n");
    printf("the plan in C has value %d\n", *plan);
    printf("plan worksize in C has value %d\n", *work_size);
    printf("plan worksize in C has size %d\n", sizeof(work_size));
}
/*--------------------------------------------------------------------------------*/
void get_plan_c2r(int nrows, int ncols, cufftHandle *plan, size_t * work_size)
{
    // nrows, ncols of output
    // Initialize an empty plan.
    cudaError_t cudaError;

        cudaError = cudaGetLastError();

        if( cudaError != cudaSuccess )

        {

        fprintf(stderr, "CUDA Runtime API Error reported : %s\n", cudaGetErrorString(cudaError));

        exit(EXIT_FAILURE);

        }
    cufftCreate(plan);
    // Turn off auto-allocation by default. This must be done before the
    // plan is actually created, so we can't use the shortcut cufftPlan1d
    cufftResult_t error;
    error=cufftSetAutoAllocation(*plan, 0);
    if(error!=CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT setAutoAllocation returned %d", error);
    }
    printf("axis 1, get plan c2r\n");
    // We can finally actually set up the plan
    if (cufftMakePlan1d(*plan, ncols, CUFFT_C2R, nrows, work_size)!=CUFFT_SUCCESS)
        fprintf(stderr,"Error planning dft.\n");
    printf("the plan in C has value %d\n", *plan);
    printf("plan worksize in C has value %d\n", *work_size);
    printf("plan worksize in C has size %d\n", sizeof(work_size));
}
void get_plan_c2c(int nrows, int ncols, cufftHandle *plan, size_t * work_size)
{
    // nrows -> batch, ncols -> size
    // Initialize an empty plan.
    cudaError_t cudaError;

        cudaError = cudaGetLastError();

        if( cudaError != cudaSuccess )

        {

        fprintf(stderr, "CUDA Runtime API Error reported : %s\n", cudaGetErrorString(cudaError));

        exit(EXIT_FAILURE);

        }
    cufftCreate(plan);
    // Turn off auto-allocation by default. This must be done before the
    // plan is actually created, so we can't use the shortcut cufftPlan1d
    cufftResult_t error;
    error=cufftSetAutoAllocation(*plan, 0);
    if(error!=CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT setAutoAllocation returned %d", error);
    }
    printf("axis 1, get plan c2c\n");
    // We can finally actually set up the plan
    if (cufftMakePlan1d(*plan, ncols, CUFFT_C2C, nrows, work_size)!=CUFFT_SUCCESS)
        fprintf(stderr,"Error planning dft.\n");
    printf("the plan in C has value %d\n", *plan);
    printf("plan worksize in C has value %d\n", *work_size);
    printf("plan worksize in C has size %d\n", sizeof(work_size));
}
        
/*--------------------------------------------------------------------------------*/
void destroy_plan(cufftHandle *plan)
{
  if (cufftDestroy(*plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
}
void get_plan_size(cufftHandle *plan, size_t *sz)
{
  if (cufftGetSize(*plan,sz)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error querying plan size.\n");
}
/*--------------------------------------------------------------------------------*/
void set_plan_scratch(cufftHandle *plan, void *buf)
{
  if (cufftSetWorkArea(*plan,buf)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error assigning buffer in set_plan_scratch.\n");
  //else
  //printf("successfully assigned buffer.\n");
          
}

/*================================================================================*/
}