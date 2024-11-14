//nvcc -o libpycufft.so pycufft.cu -shared -lcufft -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <omp.h>

extern "C" void get_work_sizes_r2c(long int *sz, int nsize, long int *nbytes)
{
  size_t nb_max=0;
  for (int i=0;i<nsize;i++)
    {
      cufftHandle plan;
      if (i==0)
        printf("plan size is %ld\n",sizeof(cufftHandle));
      int n=sz[2*i];
      int ntrans=sz[2*i+1];
      int rank=1; //we're doing 1D transforms
      int nn=(n/2)+1;
      int istride=1;
      int idist=nn;
      int oembed=nn;
      if (cufftPlanMany(&plan, rank, &n, &nn, istride, idist,&oembed, istride,oembed, CUFFT_R2C,ntrans)!=CUFFT_SUCCESS) {
        fprintf(stderr,"Error in planning r2c with dimensions %d %d\n",n,ntrans);
        *nbytes=-1;
        return;
          
      }
      cufftSetAutoAllocation(plan,1);
      size_t nb;
      if (cufftGetSize(plan,&nb)!=CUFFT_SUCCESS) {
        fprintf(stderr,"Error in querying size wth dimensions %d %d\n",n,ntrans);
        *nbytes=-1;
        return;
      }
      if (nb>nb_max)
        nb_max=nb;
      if (cufftDestroy(plan)!= CUFFT_SUCCESS) {
        fprintf(stderr,"Error destroying plan.\n");
        *nbytes=-1;
        return;
      }
    }
}
/*--------------------------------------------------------------------------------*/

void cufft_c2r(float *out, cufftComplex *data, int len, int ntrans)
{
  //float *out;
  cufftHandle plan;
  
  if (cufftPlan1d(&plan,len,CUFFT_C2R, ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning dft\n");
  //cudaDeviceSynchronize();
  //double t1=omp_get_wtime();
  if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing dft\n");
  //cudaDeviceSynchronize();
  //double t2=omp_get_wtime();
  //printf("took %12.4g seconds to do fft.\n",t2-t1);

  if (cufftDestroy(plan)!= CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
}
/*--------------------------------------------------------------------------------*/
void cufft_c2r_wplan(float *out, cufftComplex *data, cufftHandle plan)
{
  if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing idft\n");
  cudaDeviceSynchronize();
}
/*--------------------------------------------------------------------------------*/
void cufft_c2r_columns(float *out, cufftComplex *data,int len, int ntrans)
{
  cufftHandle plan;
  int rank=1;
  int inembed[rank] = {ntrans};
  int onembed[rank]={ntrans};
  int istride=ntrans;
  int idist=1;
  int ostride=ntrans;
  int odist=1;
  if (cufftPlanMany(&plan,rank,&len,inembed,istride,idist,onembed,ostride,odist,CUFFT_C2R,ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning DFT in c2r_columns.\n");
  if (cufftExecC2R(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing DFT in c2r_columns.\n");
  if (cufftDestroy(plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan in c2r_columns.\n");

}

/*--------------------------------------------------------------------------------*/
extern "C" void cufft_c2r_host(float *out, cufftComplex *data, int n, int m, int axis)
{
  float *dout;
  cufftComplex *din;
  if (cudaMalloc((void **)&din,sizeof(cufftComplex)*n*m)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(din,data,n*m*sizeof(cufftComplex),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  if (axis==0) {
    if (cudaMalloc((void **)&dout,sizeof(float)*n*m)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_c2r_columns(dout,din,n,m);
    if (cudaMemcpy(out,dout,sizeof(float)*n*m,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in c2r\n");
  }
  else {
    if (cudaMalloc((void **)&dout,sizeof(float)*n*m)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_c2r(dout,din,m,n);
    if (cudaMemcpy(out,dout,sizeof(float)*m*n,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in c2r\n");

  }
}

/*--------------------------------------------------------------------------------*/
void cufft_r2c(cufftComplex *out, float *data, int len, int ntrans)
{
  cufftHandle plan;
  cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
  if (cufftPlan1d(&plan,len,CUFFT_R2C, ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning dft\n");
  
//   double t1=omp_get_wtime();
//     // 
  if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing dft\n");
    
  if (cufftDestroy(plan)!= CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
    // cudaDeviceSynchronize();
//   double t2=omp_get_wtime();
//   printf("r2c took %12.4g\n",t2-t1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("r2c took %12.4g\n",milliseconds);
}
/*--------------------------------------------------------------------------------*/
void cufft_r2c_wplan(cufftComplex *out, float *data, int len, int ntrans,cufftHandle plan)
{
  if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing dft\n");
}

/*--------------------------------------------------------------------------------*/
void cufft_r2c_columns(cufftComplex *out, float *data, int len, int ntrans)
{
  cufftHandle plan;
  int rank=1;
  int inembed[rank] = {len};
  int onembed[rank]={ntrans};
  int istride=ntrans;
  int idist=1;
  int ostride=ntrans;
  int odist=1;
  if (cufftPlanMany(&plan,rank,&len,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,ntrans)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error planning DFT in r2c_columns.\n");
  if (cufftExecR2C(plan,data,out)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error executing DFT in r2c_columns.\n");
  if (cufftDestroy(plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan in r2c_columns.\n");
  
}


/*--------------------------------------------------------------------------------*/
extern "C" {

void cufft_r2c_gpu(cufftComplex *out, float *data, int n, int m, int axis)
{
  if (axis==1)
    cufft_r2c(out,data,m,n);
  else
    cufft_r2c_columns(out,data,n,m);
}
/*--------------------------------------------------------------------------------*/

void cufft_r2c_gpu_wplan(cufftComplex *out, float *data, int n, int m, int axis,cufftHandle *plan)
{
  if (axis==1)
    cufft_r2c_wplan(out,data,m,n,*plan);
  else
    cufft_r2c_columns(out,data,n,m);
}
/*--------------------------------------------------------------------------------*/

void cufft_c2r_gpu_wplan(float  *out, cufftComplex *data, cufftHandle *plan)
{
  cufft_c2r_wplan(out,data,*plan);
}
/*--------------------------------------------------------------------------------*/

void cufft_c2r_gpu(float *out, cufftComplex *data, int n, int m, int axis)
{
  if (axis==1)
    cufft_c2r(out,data,m,n);
  else
    cufft_c2r_columns(out,data,n,m);
}
/*--------------------------------------------------------------------------------*/
void get_plan_size(cufftHandle *plan, size_t *sz)
{
  if (cufftGetSize(*plan,sz)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error querying plan size.\n");
}
/*--------------------------------------------------------------------------------*/
void get_plan_r2c(int n, int m, int axis, cufftHandle *plan, int alloc)
{
  if (axis==1) {
    // Initialize an empty plan.
    cufftCreate(plan);
    // Turn off auto-allocation if required. This must be done before the
    // plan is actually created, so we can't use the shortcut cufftPlan1d
    if (!alloc) {
        cufftSetAutoAllocation(*plan, 0);
        //cufftSetWorkArea(*plan, NULL);
    }
    // We can finally actually set up the plan
    size_t work_size;
    if (cufftMakePlan1d(*plan, m, CUFFT_R2C, n, &work_size)!=CUFFT_SUCCESS)
      fprintf(stderr,"Error planning dft.\n");
  } else {
    // SKN: I assume this is TODO?
  }
}
        
/*--------------------------------------------------------------------------------*/
void get_plan_c2r(int n, int m, int axis,cufftHandle *plan, int alloc)
//make sure n and m correspond to the size of the *output* transform
{
  if (axis==1) {
    // Initialize an empty plan.
    cufftCreate(plan);
    // Turn off auto-allocation if required. This must be done before the
    // plan is actually created, so we can't use the shortcut cufftPlan1d
    if (!alloc) {
        cufftSetAutoAllocation(*plan, 0);
        //cufftSetWorkArea(*plan, NULL);
    }
    // We can finally actually set up the plan
    size_t work_size;
    if (cufftMakePlan1d(*plan, m, CUFFT_C2R, n, &work_size)!=CUFFT_SUCCESS)
      fprintf(stderr,"Error planning dft.\n");
  } else {
    // SKN: I assume this is TODO?
  }
}
        
/*--------------------------------------------------------------------------------*/
void destroy_plan(cufftHandle *plan)
{
  if (cufftDestroy(*plan)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error destroying plan.\n");
}
        
/*--------------------------------------------------------------------------------*/
void set_plan_scratch(cufftHandle plan,void *buf)
{
  if (cufftSetWorkArea(plan,buf)!=CUFFT_SUCCESS)
    fprintf(stderr,"Error assigning buffer in set_plan_scratch.\n");
  //else
  //printf("successfully assigned buffer.\n");
          
}

/*--------------------------------------------------------------------------------*/
void cufft_r2c_host(cufftComplex *out, float *data, int n, int m, int axis)
{
  cufftComplex *dout;
  float *din;
  int nn;
  if (axis==0)
    nn=n/2+1;
  else
    nn=m/2+1;
  if (cudaMalloc((void **)&din,sizeof(float)*n*m)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  if (cudaMemcpy(din,data,n*m*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Error copying data to device.\n");
  if (axis==0) {
    if (cudaMalloc((void **)&dout,sizeof(cufftComplex)*nn*m)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_r2c_columns(dout,din,n,m);
    //printf("copying %d %d\n",nn,m);
    if (cudaMemcpy(out,dout,sizeof(cufftComplex)*nn*m,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in r2c\n");
  }
  else {
    if (cudaMalloc((void **)&dout,sizeof(cufftComplex)*n*nn)!=cudaSuccess)
      fprintf(stderr,"error in cudaMalloc\n");
    cufft_r2c(dout,din,m,n);
    //printf("copying %d %d\n",n,nn);
    if (cudaMemcpy(out,dout,sizeof(cufftComplex)*nn*n,cudaMemcpyDeviceToHost)!=cudaSuccess)
      fprintf(stderr,"Error copying result to host in r2c\n");
  
  }
}

}



/*================================================================================*/


#if 0

int main(int argc, char *argv[])
{
  printf("Hello world!\n");
  int ndet=1000;
  int nsamp=1<<18;
  printf("nsamp is %d\n",nsamp);

  float *fdat=(float *)malloc(sizeof(float)*ndet*nsamp);
  if (fdat!=NULL)
    printf("successfully malloced array on host.\n");

  float *ddat;
  if (cudaMalloc((void **)&ddat,sizeof(float)*nsamp*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");
  cuComplex *dtrans;
  if (cudaMalloc((void **)&dtrans,sizeof(cuComplex)*nsamp*ndet)!=cudaSuccess)
    fprintf(stderr,"error in cudaMalloc\n");

  
  
}
#endif