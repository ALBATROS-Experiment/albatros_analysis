#include <stdio.h>
#include <math.h>
#include <complex.h>

//gcc-9 -fopenmp -O3 -shared -fPIC -o libpfb_helper.so pfb_helper.c  -lm -lgomp


void pfb_from_mat(double complex *pfb, double *data, double complex *mat, int nchan, int block_len, int nblock)
{

#pragma omp parallel for
  for (int block=0;block<nblock;block++) {
    for (int ichan=0;ichan<nchan;ichan++) {
      int ind=block*nchan+ichan;
      pfb[ind]=1;
      for (int i=0;i<block_len;i++)  {
	pfb[ind]+=data[block*(2*(nchan-1))+i]*mat[block_len*ichan+i];
      }
    }
  }
}


/*--------------------------------------------------------------------------------*/

void accumulate_pfb_transpose_complex(double complex *out, double complex *prod, int ntap, int nblock, int nchan)
{
  int jj=2*(nchan-1);
  int kk=jj*ntap;
  //#pragma omp parallel for
  for (int i=0;i<nblock;i++)
    for (int k=0;k<kk;k++)
      out[i*jj+k]+=prod[i*kk+k];
      
  //for (int i=0;i<nblock;i++) {
  // for (int j=0;j<jj;j++) {
  //  out[i*jj+j]=0;
  //  for (int k=0;jntap;k++) {
  //out[i*jj+j]+=prod[(i+k)*
  //  }
  //}
  //}
}
