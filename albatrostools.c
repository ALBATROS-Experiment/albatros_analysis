#include <stdio.h>
#include <stdint.h>
// gcc-9 -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp


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
