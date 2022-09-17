#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void hist_4bit(uint8_t * data, uint64_t data_len, uint64_t * hist, uint8_t nbins, int8_t mode)
{/*
	Default implementation assumes bins are 0 indexed, of width = 1, and nbins = len(hist)-1.
	Bin convention is [l,r). First bin left edge is included. Last bin right edge is excluded.

	Mode: 
	0 	= pol0 only
	1 	= pol1
	-1 	= both pols
	
*/
	for(int k=0;k<=nbins;k++) hist[k]=0;

    #pragma omp parallel default(none) firstprivate(data_len, nbins, mode) shared(data, hist)
    {
        uint64_t hist_pvt[nbins+1];
		uint8_t r, im, imask=15, rmask=240;

        for(int k=0;k<=nbins;k++) hist_pvt[k]=0; //initialize
        
        #pragma omp for nowait
        for(int i=0;i<data_len;i++)
        {
			if(mode==0)
			{
				im=data[2*i]&imask;
    			r=(data[2*i]&rmask)>>4;
				++hist_pvt[r];
				++hist_pvt[im];
			}
			else if(mode==1)
			{
				im=data[2*i+1]&imask;
    			r=(data[2*i+1]&rmask)>>4;
				++hist_pvt[r];
				++hist_pvt[im];
			}
			else if(mode==-1)
			{
				im=data[2*i]&imask;
    			r=(data[2*i]&rmask)>>4;
				++hist_pvt[r];
				++hist_pvt[im];

				im=data[2*i+1]&imask;
    			r=(data[2*i+1]&rmask)>>4;
				++hist_pvt[r];
				++hist_pvt[im];
			}
			// add else to exit

        }
        #pragma omp critical
        {
            for(int k=0;k<=nbins;k++) hist[k]+=hist_pvt[k];
        }
    }
}

void myfunc()
{
	printf("Finally yea");
}

void unpack_4bit_float(uint8_t *data,float *pol0, float *pol1, int nspec, int nchan)
{
  /*
  nspec: number of spectra = no. of rows
  nchan: number of channels = no. of columns

  Byte structure
  chan1-pol0 (rrrriiii) chan1-pol1 (rrrriiii)
  */
  
  uint8_t imask=15;
  uint8_t rmask=255-15;

  long nn=nspec*nchan;

  for (int i=0;i<nn;i++) {
    uint8_t im=data[2*i]&imask;
    uint8_t r=(data[2*i]&rmask)>>4;

    if (r > 8){pol0[2*i] = r - 16;}
    else {pol0[2*i] = r;}

    if (im > 8){pol0[2*i+1] = im - 16;}
    else {pol0[2*i+1] = im;}
    
    im = data[2*i+1]&imask;
    r = (data[2*i+1]&rmask)>>4;

    if (r > 8){pol1[2*i] = r - 16;}
    else {pol1[2*i] = r;}

    if (im > 8){pol1[2*i+1] = im - 16;}
    else {pol1[2*i+1] = im;}
    
  }
}

void unpack_1bit_float(uint8_t *data, float *pol0, float *pol1, int nspec, int nchan)
{
  uint64_t nn=nspec*nchan/2; //length of the read raw data in bytes. nn is total bytes
  #pragma omp parallel for
  for (uint64_t i = 0; i<nn; i++) {
    float r0c0=(data[i]>>7)&1;
    float i0c0=(data[i]>>6)&1;
    float r1c0=(data[i]>>5)&1;
    float i1c0=(data[i]>>4)&1;
    float r0c1=(data[i]>>3)&1;
    float i0c1=(data[i]>>2)&1;
    float r1c1=(data[i]>>1)&1;
    float i1c1=(data[i]>>0)&1;

    pol0[4*i]   = 2*r0c0-1;
    pol0[4*i+1] = 2*i0c0-1;
    pol1[4*i]   = 2*r1c0-1;
    pol1[4*i+1] = 2*i1c0-1;
    pol0[4*i+2] = 2*r0c1-1;
    pol0[4*i+3] = 2*i0c1-1;
    pol1[4*i+2] = 2*r1c1-1;
    pol1[4*i+3] = 2*i1c1-1;
  }
}

void sortpols (uint8_t *data, uint8_t *pol0, uint8_t *pol1, int rowstart, int rowend, int ncols, int nchan, short bit_depth, int chanstart, int chanend)
{
  //nchan is the actual number of channels in raw data. user may decide to unpack only a subset of them.
	int nrows = rowend-rowstart;
  int nn = nrows*ncols;
	// printf("Oi!\n");
	// printf("nrows %d ncols %d\n", nrows, ncols);
	#pragma omp parallel for
	for(int i=0;i<nn;i++)
	{
		pol0[i]=0;
		pol1[i]=0;
	}

  //printf("first spec_num %d and last specnum %d and nspec %d\n", spec_num[0], spec_num[nspec-1], nspec);
  if (bit_depth == 4)
	{
		int c1 = 2*nchan; //2 is because we have pol0 byte pol1 byte
		#pragma omp parallel for
		for(int i = 0; i<nrows; i++)  //this nspec is < nrows (as defined on python side), but corresponds to rowstart->rowend. fix the convention.
		{
				for(int k=chanstart; k<chanend; k++)
				{	
					pol0[i*ncols+k-chanstart] = data[(i+rowstart)*c1 + 2*k];
					pol1[i*ncols+k-chanstart] = data[(i+rowstart)*c1 + 2*k+1];
				}
		}
	}
  else if(bit_depth == 1)
  {
    // I want to read two bytes at a time and pack 4 channels of each pol into a new byte.
    // *MISSING SPECTRA IS IGNORED. KEPT TRACK SEPARATELY DURING XCORR AVERAGING*

    int cndn = (chanend-chanstart)%4;
 
    #pragma omp parallel for
    for(int i=0;i<nrows;i++)
    { //nrows=nspec for 1 bit since no missing zeros inserted
      int m=0,idx=0;
      uint8_t p0c0,p0c1,p0c2,p0c3,p1c0,p1c1,p1c2,p1c3;
      for(int j=0; j<ncols-1; j++)
      {
        idx = ceil(nchan/2)*i + 2*j+ chanstart/2;
        //byte 1
        p0c0 = (data[idx])&192;
        p1c0 = (data[idx])&48;
        p0c1 = (data[idx])&12;
        p1c1 = (data[idx])&3;
        //byte 2
        p0c2 = (data[idx+1])&192;
        p1c2 = (data[idx+1])&48;
        p0c3 = (data[idx+1])&12 ;
        p1c3 = (data[idx+1])&3;

        m = ncols*i+j;
        pol0[m] = p0c0+(p0c1<<2)+(p0c2>>4)+(p0c3>>2);
        pol1[m] = (p1c0<<2)+(p1c1<<4)+(p1c2>>2)+p1c3;
      }
      int j = ncols-1;
      idx = ceil(nchan/2)*i + 2*j+chanstart/2;
      m = ncols*i+j;

      switch(cndn)
      {
        //use up last byte of raw data to fill 25% to 100% of the last pol0 byte
        case 0:
          // need to read two more raw bytes
          //byte 1
          p0c0 = (data[idx])&192;
          p1c0 = (data[idx])&48;
          p0c1 = (data[idx])&12;
          p1c1 = (data[idx])&3;

          //byte 2
          p0c2 = (data[idx+1])&192;
          p1c2 = (data[idx+1])&48;
          p0c3 = (data[idx+1])&12 ;
          p1c3 = (data[idx+1])&3;

          pol0[m] = p0c0+(p0c1<<2)+(p0c2>>4)+(p0c3>>2);
          pol1[m] = (p1c0<<2)+(p1c1<<4)+(p1c2>>2)+p1c3;
          break;
        case 3:
          // need to read two more raw bytes
          //byte 1
          // printf("case 3\n");
          p0c0 = (data[idx])&192;
          p1c0 = (data[idx])&48;
          p0c1 = (data[idx])&12;
          p1c1 = (data[idx])&3;
          //byte 2
          p0c2 = (data[idx+1])&192;
          p1c2 = (data[idx+1])&48;

          pol0[m] = p0c0+(p0c1<<2)+(p0c2>>4);
          pol1[m] = (p1c0<<2)+(p1c1<<4)+(p1c2>>2);
          break;
        case 2:
          //only one more raw byte
          // printf("case 2\n");
          p0c0 = (data[idx])&192;
          p1c0 = (data[idx])&48;
          p0c1 = (data[idx])&12;
          p1c1 = (data[idx])&3;

          pol0[m] = p0c0+(p0c1<<2);
          pol1[m] = (p1c0<<2)+(p1c1<<4);
          break;
        case 1:
          // printf("case 1\n");
          //only one more raw byte
          p0c0 = (data[idx])&192;
          p1c0 = (data[idx])&48;

          pol0[m] = p0c0;
          pol1[m] = (p1c0<<2);
          break;
      }
    }
  }
}








































