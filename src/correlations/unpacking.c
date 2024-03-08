#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void hist_4bit(uint8_t * data, uint64_t * hist, int rowstart, int rowend, int nchan, int nbins, int mode)
{/*
	Default implementation assumes bins are 0 indexed, of width = 1, and nbins = len(hist)-1.
	Bin convention is [l,r). First bin left edge is included. Last bin right edge is excluded.

	Mode: 
	0 	= pol0 only
	1 	= pol1
	-1 	= both pols
	
*/
  int nrows=rowend-rowstart;
  int c1 = 2*nchan; //c1 is how many bytes to skip to get to the next spectra
  int hist_len = nchan*(nbins+1);
	for(int k=0;k<hist_len;k++)
  {
    hist[k]=0;
  }

  #pragma omp parallel
  {
    uint64_t hist_pvt[hist_len];
    uint8_t r, im, imask=15, rmask=240;

    for(int k=0;k<hist_len;k++)
    {
      hist_pvt[k]=0;
    } 
    
    #pragma omp for nowait
    for(int i=0;i<nrows;i++)
    {
      for(int j=0; j<nchan; j++)
      {
        if(mode==0)
        {
          im=data[(i+rowstart)*c1+2*j]&imask;
          r=(data[(i+rowstart)*c1+2*j]&rmask)>>4;
          ++hist_pvt[r*nchan+j];
          ++hist_pvt[im*nchan+j];
        }
        else if(mode==1)
        {
          im=data[(i+rowstart)*c1+2*j+1]&imask;
          r=(data[(i+rowstart)*c1+2*j+1]&rmask)>>4;
          ++hist_pvt[r*nchan+j];
          ++hist_pvt[im*nchan+j];
        }
        else if(mode==-1)
        {
          im=data[(i+rowstart)*c1+2*j]&imask;
          r=(data[(i+rowstart)*c1+2*j]&rmask)>>4;
          ++hist_pvt[r*nchan+j];
          ++hist_pvt[im*nchan+j];

          im=data[(i+rowstart)*c1+2*j+1]&imask;
          r=(data[(i+rowstart)*c1+2*j+1]&rmask)>>4;
          ++hist_pvt[r*nchan+j];
          ++hist_pvt[im*nchan+j];
        }
      }
    }
    #pragma omp critical
    {
        for(int k=0;k<=hist_len;k++) hist[k]+=hist_pvt[k];
    }
  }
}

void hist_1bit(uint8_t * data, uint64_t * hist, int rowstart, int rowend, int nchan, int nbins, int mode)
{
  int nrows=rowend-rowstart;
  int ncols= nchan/2;
  int hist_len = nchan*(nbins+1);
	for(int k=0;k<hist_len;k++)
  {
    hist[k]=0;
  }

  #pragma omp parallel
  {
    uint64_t hist_pvt[hist_len];
    uint8_t p0c0r,p0c0im,p0c1r,p0c1im,p1c0r,p1c0im,p1c1r,p1c1im;

    for(int k=0;k<hist_len;k++)
    {
      hist_pvt[k]=0;
    } 
    
    #pragma omp for nowait
    for(int i=0;i<nrows;i++)
    {
      for(int j=0; j<ncols; j++)
      {
        int idx = (i+rowstart)*ncols+j;
        if(mode==0)
        {
          //byte 1
          p0c0r = (data[idx]>>7)&1;
          p0c0im = (data[idx]>>6)&1;
          p0c1r = (data[idx]>>3)&1;
          p0c1im = (data[idx]>>2)&1;
          ++hist_pvt[p0c0r*nchan+2*j];
          ++hist_pvt[p0c0im*nchan+2*j];
          ++hist_pvt[p0c1r*nchan+2*j+1];
          ++hist_pvt[p0c1im*nchan+2*j+1];
        }
        else if(mode==1)
        {
          p1c0r = (data[idx]>>5)&1;
          p1c0im = (data[idx]>>4)&1;
          p1c1r = (data[idx]>>1)&1;
          p1c1im = (data[idx])&1;
          ++hist_pvt[p1c0r*nchan+2*j];
          ++hist_pvt[p1c0im*nchan+2*j];
          ++hist_pvt[p1c1r*nchan+2*j+1];
          ++hist_pvt[p1c1im*nchan+2*j+1];
        }
        else if(mode==-1)
        {
          p0c0r = (data[idx]>>7)&1;
          p0c0im = (data[idx]>>6)&1;
          p0c1r = (data[idx]>>3)&1;
          p0c1im = (data[idx]>>2)&1;
          p1c0r = (data[idx]>>5)&1;
          p1c0im = (data[idx]>>4)&1;
          p1c1r = (data[idx]>>1)&1;
          p1c1im = (data[idx])&1;
          ++hist_pvt[p0c0r*nchan+2*j];
          ++hist_pvt[p0c0im*nchan+2*j];
          ++hist_pvt[p0c1r*nchan+2*j+1];
          ++hist_pvt[p0c1im*nchan+2*j+1];
          ++hist_pvt[p1c0r*nchan+2*j];
          ++hist_pvt[p1c0im*nchan+2*j];
          ++hist_pvt[p1c1r*nchan+2*j+1];
          ++hist_pvt[p1c1im*nchan+2*j+1];
        }
      }
    }
    for(int k=0;k<=hist_len;k++)
    {
      #pragma omp atomic //let's try atomic instead of a whole critical block
      hist[k]+=hist_pvt[k];
    }
  
  }
}

void unpack_4bit_float(uint8_t *data, float *pol0, float *pol1, int rowstart, int rowend, int64_t * channels, int ncols, int nchan)
{
  /*
  ncols: number of channels you want to unpack
  nchan: total number of channels in the original baseband data
  channels: list of channel indicies you're unpacking

  Byte structure
  chan1-pol0 (rrrriiii) chan1-pol1 (rrrriiii)
  */
  int nrows = rowend-rowstart;
  int nn=nrows*ncols;
  uint8_t imask=15;
  uint8_t rmask=255-15;

  #pragma omp parallel for
	for(int i=0;i<2*nn;i++) //complex array
	{
		pol0[i]=0;
		pol1[i]=0;
	}

  int c1 = 2*nchan;
  int c2 = 2*ncols;
  #pragma omp parallel for
  for(int i = 0; i<nrows; i++)
  {
    for(int k=0; k<ncols; k++)
    {
      int polidx = i*c2+2*k;
      int dataidx = (i+rowstart)*c1 + 2*(channels[k]);

      uint8_t im=data[dataidx]&imask;
      uint8_t r=(data[dataidx]&rmask)>>4;
      if (r > 8){pol0[polidx] = r - 16;}
      else {pol0[polidx] = r;}

      if (im > 8){pol0[polidx+1] = im - 16;}
      else {pol0[polidx+1] = im;}

      im=data[dataidx+1]&imask;
      r=(data[dataidx+1]&rmask)>>4;

      if (r > 8){pol1[polidx] = r - 16;}
      else {pol1[polidx] = r;}

      if (im > 8){pol1[polidx+1] = im - 16;}
      else {pol1[polidx+1] = im;}
    }
  }
}

void unpack_1bit_float(uint8_t *data, float *pol0, float *pol1, int rowstart, int rowend, int chanstart, int chanend, int nchan)
{
  int nrows = rowend-rowstart;
  int bytes_per_spec=nchan/2;
  #pragma omp parallel for
	for(int i=0;i<2*nrows*(chanend-chanstart);i++) //complex array
	{
		pol0[i]=0;
		pol1[i]=0;
	}
  // #pragma omp parallel for
  for (int i = 0; i<nrows; i++) 
  {
    for (int j =0; j<(chanend-chanstart)/2; j++) // number of bytes to read for each spectra.
    {
      int dataidx = (i+rowstart)*bytes_per_spec+chanstart/2+j;
      float r0c0=(data[dataidx]>>7)&1;
      float i0c0=(data[dataidx]>>6)&1;
      float r1c0=(data[dataidx]>>5)&1;
      float i1c0=(data[dataidx]>>4)&1;
      float r0c1=(data[dataidx]>>3)&1;
      float i0c1=(data[dataidx]>>2)&1;
      float r1c1=(data[dataidx]>>1)&1;
      float i1c1=(data[dataidx]>>0)&1;
      // printf("Reading current byte %d %f %f %f %f %f %f %f %f %f\n",data[dataidx], r0c0,i0c0,r1c0,i1c0,r0c1,i0c1,r1c1,i1c1);

      int polidx = i*(chanend-chanstart)*2 + 4*j; // for each spectra-byte read, skip by 4 since each byte has two channels = 4 x int32 numbers.
      // printf("Filling current polidx %d \n", polidx);
      pol0[polidx]   = 2*r0c0-1;
      pol0[polidx+1] = 2*i0c0-1;
      pol0[polidx+2] = 2*r0c1-1;
      pol0[polidx+3] = 2*i0c1-1;
      pol1[polidx]   = 2*r1c0-1;
      pol1[polidx+1] = 2*i1c0-1;
      pol1[polidx+2] = 2*r1c1-1;
      pol1[polidx+3] = 2*i1c1-1;
    }
    
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
        idx = ceil(nchan/2)*(i+rowstart) + 2*j+ chanstart/2; //skipping by 2*j = 4 channels since we fill each byte of output with 4 chans
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
      idx = ceil(nchan/2)*(i+rowstart) + 2*j+chanstart/2;
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








































