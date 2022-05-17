#include <stdio.h>
#include <stdint.h>
#include <omp.h>

void autocorr_4bit(uint8_t * data, uint8_t * corr, uint32_t nspec, uint32_t ncol)
{
	/*
		This is for fine control over what rows to correlate for some given data.
		Data is assumed to consist of only one pol (already sortpol'd).

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

void avg_autocorr_4bit(uint8_t * data, int64_t * corr, uint32_t nspec, uint32_t ncol)
{
	/*
		Returns an array of nchan elements. Sum over all spectra for each channel. 
		Division by appropriate spectra count will be taken care by python frontend.
	*/

	uint8_t imask=15;
  	uint8_t rmask=255-15;
	int64_t sum = 0; //should be more than enough. also eventually if data stored as float64 in autocorravg.py, should be compatible

	for(int i = 0; i<ncol; i++)
	{	
		sum = 0;

		#pragma omp parallel for default(none) firstprivate(imask,rmask,nspec,ncol,i) shared(data) reduction(+: sum)
		for(int j = 0; j<nspec; j++)
		{
			int8_t im=data[j*ncol+i]&imask;
			int8_t r=(data[j*ncol+i]&rmask)>>4;
			if (r > 8){r = r - 16;}
			if (im > 8){im = im - 16;}
			// printf("int8r %d, r*r = %d\n", (int8_t)r, prod);
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

void xcorr_4bit(uint8_t * data0, uint8_t * data1, float * xcorr, uint32_t nspec, uint32_t ncol)
{

	uint64_t nn = nspec * ncol;
	uint8_t imask=15;
  	uint8_t rmask=255-15;

	#pragma omp parallel for default(none) firstprivate(imask,rmask,nn) shared(data0,data1,xcorr)
	for(int i = 0; i<nn; i++)
	{
		int8_t im0=data0[i]&imask;
		int8_t r0=(data0[i]&rmask)>>4;
		if (r0 > 8){r0 = r0 - 16;}
		if (im0 > 8){im0 = im0 - 16;}

		int8_t im1=data1[i]&imask;
		int8_t r1=(data1[i]&rmask)>>4;
		if (r1 > 8){r1 = r1 - 16;}
		if (im1 > 8){im1 = im1 - 16;}

		//printf("data %d, r %d, im %d\n", data[i], r, im);
		//printf("r %d, im %d", r, im);
		xcorr[2*i] = r0*r1 + im0*im1;
		xcorr[2*i+1] = r1*im0 - r0*im1;
		//printf("corr[i]=%d\n", corr[i]);
	}

}

void avg_xcorr_4bit(uint8_t * data0, uint8_t * data1, float * xcorr, uint32_t nspec, uint32_t ncol)
{
	/*
		Returns an array of nchan elements. Sum over all spectra for each channel. 
		Division by appropriate spectra count will be taken care by python frontend.
	*/

	uint8_t imask=15;
  	uint8_t rmask=255-15;
	int32_t sum_r, sum_im = 0; //+2.1bil to -2.4bil, should be enough, and compatible with float32

	for(int i = 0; i<ncol; i++)
	{	
		sum_r = 0;
		sum_im = 0;

		#pragma omp parallel for default(none) firstprivate(imask,rmask,nspec,ncol,i) shared(data0,data1) reduction(+: sum_r,sum_im)
		for(int j = 0; j<nspec; j++)
		{
			int8_t im0=data0[j*ncol+i]&imask;
			int8_t r0=(data0[j*ncol+i]&rmask)>>4;
			if (r0 > 8){r0 = r0 - 16;}
			if (im0 > 8){im0 = im0 - 16;}

			int8_t im1=data1[j*ncol+i]&imask;
			int8_t r1=(data1[j*ncol+i]&rmask)>>4;
			if (r1 > 8){r1 = r1 - 16;}
			if (im1 > 8){im1 = im1 - 16;}
			// printf("int8r %d, r*r = %d\n", (int8_t)r, prod);
			sum_r = sum_r + r0*r1 + im0*im1;
			sum_im = sum_im + r1*im0 - r0*im1;
		}
		// implicit barrier here
		xcorr[2*i] = sum_r;
		xcorr[2*i+1] = sum_im;
	}	
}

