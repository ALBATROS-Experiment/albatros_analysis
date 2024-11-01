//nvcc -o libpycufft.so pycufft.cu -shared -lcufft -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
/*--------------------------------------------------------------------------------*/
extern "C"{
void cufft_r2c_mohan(cufftComplex *out, float * data, int nrows, int ncols, cufftHandle * plan_ptr)
{
    // nrows = rows in input data; ncols = columns in input data.
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaDeviceSynchronize();
    // cudaEventRecord(start, 0);
    if (plan_ptr)
    {
        //if a plan is available execute and move on
        // printf("Using user's plan r2c\n");
        // cudaEventRecord(start, 0);
        if (cufftExecR2C(*plan_ptr,data,out)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft\n");
    }
    else
    {
        cufftHandle plan;
        if (cufftPlan1d(&plan,ncols,CUFFT_R2C,nrows)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error planning dft\n");
        if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft\n");
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
void cufft_c2r_mohan(float *out, cufftComplex * data, int nrows, int ncols, cufftHandle * plan_ptr)
{
    // nrows = rows in output data; ncols = columns in output data.
    // if (plan==NULL)
    // {
    //     printf("plan is null\n");
    //     printf("nrows = %d ncols = %d\n", nrows, ncols);
    // }
    if (plan_ptr)
    {
        // printf("Using user's plan c2r\n");
        //if a plan is available execute and move on
        if (cufftExecC2R(*plan_ptr,data,out)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft\n");
        // cudaDeviceSynchronize();
    }
    else
    {  
        cufftHandle plan;
        if (cufftPlan1d(&plan,ncols,CUFFT_C2R,nrows)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error planning dft\n");
        if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
            fprintf(stderr,"Error executing dft\n");
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
    cufftCreate(plan);
    // Turn off auto-allocation by default. This must be done before the
    // plan is actually created, so we can't use the shortcut cufftPlan1d
    cufftSetAutoAllocation(*plan, 0);
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
    cufftCreate(plan);
    // Turn off auto-allocation by default. This must be done before the
    // plan is actually created, so we can't use the shortcut cufftPlan1d
    cufftSetAutoAllocation(*plan, 0);
    printf("axis 0, get plan c2r\n");
    // We can finally actually set up the plan
    if (cufftMakePlan1d(*plan, ncols, CUFFT_C2R, nrows, work_size)!=CUFFT_SUCCESS)
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