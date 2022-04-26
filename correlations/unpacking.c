#include <stdio.h>
#include <stdint.h>

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
  }
}

void unpack_1bit_float(uint8_t *data, float *pol0, float *pol1, int ndat, int nchan)
{
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
    }
  }
}

void sortpols (uint8_t *data, uint8_t *pol0, uint8_t *pol1, int ndat, int nchan, short bit_depth)
{
	if (bit_depth == 4){
		long nn=ndat*nchan/2;
		for (int i = 0; i < nn; i++)
		{
			pol0[i] = data[2 * i];
			pol1[i] = data[2 * i + 1];
		}
	}
	else if (bit_depth == 2){
		long nn=ndat*nchan/2;
		uint8_t mask1 = 15;
		uint8_t mask0 = 240;
		for (int i = 0; i < nn; i++)
		{
			switch (i % 2)
			{
				case 0:
				  pol0[i/2] = data[i] & mask0;
				  pol1[i/2] = (data[i] & mask1) << 4;
				  break;
				case 1:
				  pol0[i/2] += (data[i] & mask0) >> 4;
				  pol1[i/2] += data[i] & mask1;
				  break;
			}
		}
	}
	else if (bit_depth == 1){
		long nn=ndat*nchan/2;
		uint8_t mask = 3;
		for (int i = 0; i < nn; i++)
		{
			switch (i % 4)
			{
				case 0:
				  pol0[i/4] = ((data[i/2] >> 6) & mask) << 6;
				  pol1[i/4] = ((data[i/2] >> 4) & mask) << 6;
				  break;
				case 1:
				  pol0[i/4] = ((data[i/2] >> 2) & mask) << 4;
				  pol1[i/4] = (data[i/2] & mask) << 4;
				  break;
				case 2:
				  pol0[i/4] = ((data[i/2] >> 6) & mask) << 2;
				  pol1[i/4] = ((data[i/2] >> 4) & mask) << 2;
				  break;
				case 3:
				  pol0[i/4] = (data[i/2] >> 2) & mask;
				  pol1[i/4] = data[i/2] & mask;
				  break; 
			}
		}
	}
	else printf("sortpols unknown bit depth");
}

unsigned int dropped_packets (uint8_t *data, unsigned long *spec_num, unsigned int num_packets, const int spectra_per_packet, int nchan, short bit_depth)
{
	if (num_packets >= 2)
	{
		unsigned int num_dropped = 0;
		unsigned int entries_per_packet;
		if (bit_depth == 4)
		{
			entries_per_packet = 2 * nchan * spectra_per_packet;
		}
		else if (bit_depth == 2) 
		{
			entries_per_packet = nchan * spectra_per_packet;
		}
		else if (bit_depth == 1)
		{
			entries_per_packet = nchan * spectra_per_packet/2; //This is supposed to divide evenly. If it doesn't, that means the packet ends in the middle of a byte, which would be bad
		}
		else return 0;

		for (unsigned int j = 0; j < num_packets - 1; j++)
		{
			if (spec_num[j + 1] - spec_num[j] != spectra_per_packet)
			{
				num_dropped++;
				const unsigned int initial = entries_per_packet * j;
				const unsigned int bound = initial + entries_per_packet;
				for (unsigned int i = initial; i < bound; i++)
				{
					data[i] = 0;
				}
			}
		}
		return num_dropped;
	}
	else 
	{
		printf("There are fewer than 2 packets in the dropped packets function\n");
		return 0;
	}
}









































