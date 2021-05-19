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
void unpack_2bit_float(uint8_t *data,float *pol0, float *pol1, int ndat, int nchan)
{
  long nn=ndat*nchan;
  uint8_t mask=3;
#pragma omp parallel for
  for (int i=0;i<nn;i++) {
    uint8_t r0=(data[i]>>6)&mask;
    uint8_t i0=(data[i]>>4)&mask;
    uint8_t r1=(data[i]>>2)&mask;
    uint8_t i1=(data[i])&mask;
    pol0[2*i]=r0-1.0;
    pol0[2*i+1]=i0-1.0;
    pol1[2*i]=r1-1.0;
    pol1[2*i+1]=i1-1.0;
    if (pol0[2*i]<=0)
      pol0[2*i]--;
    if (pol0[2*i+1]<=0)
      pol0[2*i+1]--;
    if (pol1[2*i]<=0)
      pol1[2*i]--;

    if (pol1[2*i+1]<=0)
      pol1[2*i+1]--;

    //printf("Values are %d %d %d %d\n",r0,i0,r1,i1);
    //if (i>10)
    //  return;
  }
}
/*--------------------------------------------------------------------------------*/
void unpack_1bit_float_busted(uint8_t *data, float *pol0, float *pol1, int ndat, int nchan)
//this mapping is almost certainly wrong since it doesn't agree with python, but leaving in since as
//of now, the python has not been confirmed to be correct
{

  //[0,0] -> [0,0]  0->0
  //[0,1] -> [1,0]  1->nchan
  //[0,2] -> [0,1]  2->1
  //[0,3] -> [1,1]  3->nchan+1
  //[1,0] -> [0,nchan/2]  nchan->nchan/2
  //[1,1] -> [1,nchan/2]  nchan+1 ->nchan+nchan/2
  //[1,2] -> [0,nchan/2+1]


  //  int nn=ndat*nchan/2*4;
  int nn=ndat*nchan;
  //#pragma omp parallel for
  for (int i=0;i<nn;i++) {
    float r0c0=(data[i]>>7)&1;
    float i0c0=(data[i]>>6)&1;
    float r1c0=(data[i]>>5)&1;
    float i1c0=(data[i]>>4)&1;

    float r0c1=(data[i]>>3)&1;
    float i0c1=(data[i]>>2)&1;
    float r1c1=(data[i]>>1)&1;
    float i1c1=(data[i]>>0)&1;

    //if ( (i&1)==0) {

    pol0[4*i+0]=2*r0c0-1;
    pol0[4*i+1]=2*i0c0-1;
    pol0[4*i+2]=2*r0c1-1;
    pol0[4*i+3]=2*i0c1-1;


    //pol0[2*i+0]=2*r0c0-1;
    //pol0[2*i+1]=2*i0c0-1;
    //pol0[2*i+2]=2*r0c1-1;
    //pol0[2*i+3]=2*i0c1-1;



      //pol0[4*i+0]=2*r0c0-1;
      //pol0[4*i+1]=2*i0c0-1;
      //pol0[4*i+2*nchan+0]=2*r0c1-1;
      //pol0[4*i+2*nchan+1]=2*i0c1-1;
      
      pol1[4*i+0]=2*r1c0-1;
      pol1[4*i+1]=2*i1c0-1;
      pol1[4*i+2]=2*r1c1-1;
      pol1[4*i+3]=2*i1c1-1;
      //}
      //else
      //{
      //}
  }
}

/*--------------------------------------------------------------------------------*/
void unpack_1bit_float(uint8_t *data, float *pol0, float *pol1, int ndat, int nchan)
{
  //  int nn=ndat*nchan/2*4;
  int nn=ndat*nchan;
#pragma omp parallel for
  for (int ii=0;ii<ndat;ii++) {
    for (int jj=0;jj<nchan;jj++) {
      int i=ii*nchan+jj;
      
      float r0c0=(data[i]>>7)&1;
      float i0c0=(data[i]>>6)&1;
      float r1c0=(data[i]>>5)&1;
      float i1c0=(data[i]>>4)&1;

      float r0c1=(data[i]>>3)&1;
      float i0c1=(data[i]>>2)&1;
      float r1c1=(data[i]>>1)&1;
      float i1c1=(data[i]>>0)&1;
      
      pol0[4*ii*nchan+2*jj]=2*r0c0-1;
      pol0[4*ii*nchan+2*jj+1]=2*i0c0-1;
      pol0[(4*ii+2)*nchan+2*jj]=2*r0c1-1;
      pol0[(4*ii+2)*nchan+2*jj+1]=2*i0c1-1;


      pol1[4*ii*nchan+2*jj]=2*r1c0-1;
      pol1[4*ii*nchan+2*jj+1]=2*i1c0-1;
      pol1[(4*ii+2)*nchan+2*jj]=2*r1c1-1;
      pol1[(4*ii+2)*nchan+2*jj+1]=2*i1c1-1;
      
      //}
    }
  }
}
/*--------------------------------------------------------------------------------*/
void bin_crosses_float(float *pol0, float *pol1, float *sum, int ndata, int nchan, int chunk)
{
  //printf("ndata/nchan/chunk are %d,%d,%d\n",ndata,nchan,chunk);
  //take assumed complex data in *data, that's ndata by nchan, and sum conj(data)*data in units of the chunk size
  int nchunk=ndata/chunk;
  //#pragma omp parallel for
  for (int i=0;i<nchunk;i++) {
    printf("I is %d\n",i);
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
