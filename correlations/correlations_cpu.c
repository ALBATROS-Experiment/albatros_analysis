#include <stdio.h>
#include <stdint.h>
#include <omp.h>

void autocorr_4bit(uint8_t * data, uint8_t * corr, uint32_t nspec, uint32_t ncol)
{
	/*
		This is for fine control over what rows to correlate for some given data

		data: nspec * ncol array. ncol = nchan only for 4 bit

		corr: nspec * nchan array. Stores autocorr for each spec and each channel. 
		
		In 4 bit case input and output dimensions are same.
		However, for 1 bit, you'll be filling 4 channels for each input data point.
	*/
	
	uint64_t nn = nspec * ncol;
	uint8_t imask=15;
  	uint8_t rmask=255-15;

	#pragma omp parallel for default(none) firstprivate(imask,rmask,nn) shared(data,corr)
	for(int i = 0; i<nn; i++)
	{
		int8_t im=data[i]&imask;
    	int8_t r=(data[i]&rmask)>>4;
		if (r > 8){r = r - 16;}
		if (im > 8){im = im - 16;}
		//printf("data %d, r %d, im %d\n", data[i], r, im);
		//printf("r %d, im %d", r, im);
		corr[i] = r*r + im*im;
		//printf("corr[i]=%d\n", corr[i]);
	}

}

void avg_autocorr_4bit(uint8_t * data, uint64_t * corr, uint32_t nspec, uint32_t ncol)
{
	uint8_t imask=15;
  	uint8_t rmask=255-15;
	uint64_t sum = 0;

	for(int i = 0; i<1; i++)
	{	
		sum = 0;

		#pragma omp parallel for default(none) firstprivate(imask,rmask,nspec,ncol,i) shared(data) reduction(+: sum)
		for(int j = 0; j<64; j++)
		{
			int8_t im=data[j*ncol+i]&imask;
			int8_t r=(data[j*ncol+i]&rmask)>>4;
			if (r > 8){r = r - 16;}
			if (im > 8){im = im - 16;}
			printf("data %d, r %d, im %d\n", data[j*ncol+i], r, im);
			sum = sum + r*r + im*im;
		}
		// implicit barrier here
		corr[i] = sum;
	}	
}

void avg_autocorr_4bit_raw()
{
	// something that can average whole files, both pols from raw data without any unpacking/sortpol
}
