#include <stdio.h>
#include <stdint.h>
// gcc-9 -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp
// also need to add ur directory to LD_LIBRARY_PATH

void unpack_4bit(uint8_t *data,double *pol0, double *pol1, int ndat, int nchan)
{
  uint8_t rmask=15;
  uint8_t imask=255-15;

  //  for (int i=0;i<6;i++) {
  // uint8_t r=data[i]&rmask;
  // uint8_t im=(data[i]&imask)>>4;
  // printf("value %d is %d %d\n",i,r,im);
  //}
  long nn=ndat*nchan/2;
#pragma omp parallel for
  for (int i=0;i<nn;i++) {
    uint8_t r=data[2*i]&rmask;
    uint8_t im=(data[2*i]&imask)>>4;
    pol0[2*i]=im;
    pol0[2*i+1]=r;
    if (pol0[2*i]>8)
      pol0[2*i]-=16;
    if (pol0[2*i+1]>8)
      pol0[2*i+1]-=16;

    r=data[2*i+1]&rmask;
    im=(data[2*i+1]&imask)>>4;
    pol1[2*i]=im;
    pol1[2*i+1]=r;
    if (pol1[2*i]>8)
      pol1[2*i]-=16;
    if (pol1[2*i+1]>8)
      pol1[2*i+1]-=16;

  }
}

/*--------------------------------------------------------------------------------*/

void unpack_4bit_float(uint8_t *data,float *pol0, float *pol1, int ndat, int nchan)
{
  uint8_t rmask=15;
  uint8_t imask=255-15;

  //  for (int i=0;i<6;i++) {
  // uint8_t r=data[i]&rmask;
  // uint8_t im=(data[i]&imask)>>4;
  // printf("value %d is %d %d\n",i,r,im);
  //}
  long nn=ndat*nchan/2;
#pragma omp parallel for
  for (int i=0;i<nn;i++) {
    uint8_t r=data[2*i]&rmask;
    uint8_t im=(data[2*i]&imask)>>4;
    pol0[2*i]=im;
    pol0[2*i+1]=r;
    if (pol0[2*i]>8)
      pol0[2*i]-=16;
    if (pol0[2*i+1]>8)
      pol0[2*i+1]-=16;

    r=data[2*i+1]&rmask;
    im=(data[2*i+1]&imask)>>4;
    pol1[2*i]=im;
    pol1[2*i+1]=r;
    if (pol1[2*i]>8)
      pol1[2*i]-=16;
    if (pol1[2*i+1]>8)
      pol1[2*i+1]-=16;

  }
}

/*--------------------------------------------------------------------------------*/
void bin_crosses_float(float *pol0, float *pol1, float *sum, int ndata, int nchan, int chunk)
{
  //printf("ndata/nchan/chunk are %d,%d,%d\n",ndata,nchan,chunk);
  //take assumed complex data in *data, that's ndata by nchan, and sum conj(data)*data in units of the chunk size
  int nchunk=ndata/chunk;
#pragma omp parallel for
  for (int i=0;i<nchunk;i++) {
    int ik2=(2*i*nchan);
    //for (int j=0;j<2*nchan;j++) 
    //  sum[i*nchan+j]=2;
    for (int j=0;j<nchan;j++) {
      sum[ik2+2*j]=0;
      sum[ik2+2*j+1]=0;
    }
    for (int k=0;k<chunk;k++) {
      int ik=2*(i*chunk+k)*nchan;
      for (int j=0;j<nchan;j++) {
	sum[ik2+2*j]+=pol0[ik+2*j]*pol1[ik+2*j]+pol0[ik+2*j+1]*pol1[ik+2*j+1];
	sum[ik2+2*j+1]+=pol0[ik+2*j]*pol1[ik+2*j+1]-pol0[ik+2*j+1]*pol1[ik+2*j];
	//sum[ik2+2*j]=1;
	//sum[ik2+2*j+1]=1;
      }	
    }
  }  
}
/*--------------------------------------------------------------------------------*/
void bin_crosses_double(double *pol0, double *pol1, double *sum, int ndata, int nchan, int chunk)
{
  //printf("ndata/nchan/chunk are %d,%d,%d\n",ndata,nchan,chunk);
  //take assumed complex data in *data, that's ndata by nchan, and sum conj(data)*data in units of the chunk size
  int nchunk=ndata/chunk;
#pragma omp parallel for
  for (int i=0;i<nchunk;i++) {
    int ik2=(2*i*nchan);
    //for (int j=0;j<2*nchan;j++) 
    //  sum[i*nchan+j]=2;
    for (int j=0;j<nchan;j++) {
      sum[ik2+2*j]=0;
      sum[ik2+2*j+1]=0;
    }
    for (int k=0;k<chunk;k++) {
      int ik=2*(i*chunk+k)*nchan;
      for (int j=0;j<nchan;j++) {
	sum[ik2+2*j]+=pol0[ik+2*j]*pol1[ik+2*j]+pol0[ik+2*j+1]*pol1[ik+2*j+1];
	sum[ik2+2*j+1]+=pol0[ik+2*j]*pol1[ik+2*j+1]-pol0[ik+2*j+1]*pol1[ik+2*j];
	//sum[ik2+2*j]=1;
	//sum[ik2+2*j+1]=1;
      }	
    }
  }  
}

/*--------------------------------------------------------------------------------*/
void bin_autos_float(float *data, float *sum, int ndata, int nchan, int chunk)
{
  //take assumed complex data in *data, that's ndata by nchan, and sum conj(data)*data in units of the chunk size
  int nchunk=ndata/chunk;
#pragma omp parallel for
  for (int i=0;i<nchunk;i++) {
    for (int j=0;j<nchan;j++) 
      sum[i*nchan+j]=0;
    for (int k=0;k<chunk;k++) {
      int ik=2*(i*chunk+k)*nchan;
      int ik2=(i*nchan);
      for (int j=0;j<nchan;j++)
	sum[ik2+j]+=data[ik+2*j]*data[ik+2*j]+data[ik+2*j+1]*data[ik+2*j+1];      
    }
  }  
}

/*--------------------------------------------------------------------------------*/
void bin_autos_double(double *data, double *sum, int ndata, int nchan, int chunk)
{
  //take assumed complex data in *data, that's ndata by nchan, and sum conj(data)*data in units of the chunk size
  int nchunk=ndata/chunk;
#pragma omp parallel for
  for (int i=0;i<nchunk;i++) {
    for (int j=0;j<nchan;j++) 
      sum[i*nchan+j]=0;
    for (int k=0;k<chunk;k++) {
      int ik=2*(i*chunk+k)*nchan;
      int ik2=(i*nchan);
      for (int j=0;j<nchan;j++)
	sum[ik2+j]+=data[ik+2*j]*data[ik+2*j]+data[ik+2*j+1]*data[ik+2*j+1];      
    }
  }  
}
