#include <stdio.h>
#include <stdint.h>

void bin_crosses_1bit(const uint8_t *pol0, const uint8_t *pol1, int *sum, int ndata, int nchan, int chunk)
{
	const int nchunk = ndata/chunk;
	for (int i = 0; i < nchunk; i++)
	{
		unsigned int chunkIdx = 2 * i * nchan;
		for (int j=0;j<nchan;j++)
		{
      			sum[chunkIdx + 2 * j] = 0;
      			sum[chunkIdx + 2 * j + 1] = 0;
      		}
      	}
      	
      	unsigned int totalndata = nchunk * chunk * nchan;
      	unsigned int chunkData = chunk * nchan;
      	for (unsigned int i = 0; i < totalndata; i++)
      	{
      		int currentChannel = i % nchan;
      		int currentChunk = i / chunkData;
      		
      		int8_t pol0_re;
      		int8_t pol0_im;
      		int8_t pol1_re;
      		int8_t pol1_im;
      		switch (i % 4)
      		{
      			case 0:
      				pol0_re = (pol0[i/4] >> 7) & 1;
      				pol0_im = (pol0[i/4] >> 6) & 1;
      				pol1_re = (pol1[i/4] >> 7) & 1;
      				pol1_im = (pol1[i/4] >> 6) & 1;
      				break;
      			case 1:
      				pol0_re = (pol0[i/4] >> 5) & 1;
      				pol0_im = (pol0[i/4] >> 4) & 1;
      				pol1_re = (pol1[i/4] >> 5) & 1;
      				pol1_im = (pol1[i/4] >> 4) & 1;
      				break;
      			case 2:
      				pol0_re = (pol0[i/4] >> 3) & 1;
      				pol0_im = (pol0[i/4] >> 2) & 1;
      				pol1_re = (pol1[i/4] >> 3) & 1;
      				pol1_im = (pol1[i/4] >> 2) & 1;
      				break;
      			case 3:
      				pol0_re = (pol0[i/4] >> 1) & 1;
      				pol0_im = pol0[i/4] & 1;
      				pol1_re = (pol1[i/4] >> 1) & 1;
      				pol1_im = pol1[i/4] & 1;
      				break;
      		}
      		
      		//this sets 0 to -1 and 1 to 1
      		pol0_re = 2 * pol0_re - 1;
      		pol0_im = 2 * pol0_im - 1;
      		pol1_re = 2 * pol1_re - 1;
      		pol1_im = 2 * pol1_im - 1;
      		
      		sum[currentChunk * nchan * 2 + currentChannel * 2] += pol0_re * pol1_re + pol0_im * pol1_im;
      		sum[currentChunk * nchan * 2 + currentChannel * 2 + 1] += pol0_im * pol1_re - pol0_re * pol1_im;
      	}
}

void bin_autos_1bit(const uint8_t *data, int *sum, int ndata, int nchan, int chunk)
{
	const int nchunk = ndata/chunk;
	for (int i = 0; i < nchunk; i++)
	{
		unsigned int chunkIdx = i * nchan;
		for (int j=0;j<nchan;j++)
		{
      			sum[chunkIdx + j] = 0;
      		}
      	}
      	
      	unsigned int totalndata = nchunk * chunk * nchan;
      	unsigned int chunkData = chunk * nchan;
      	for (unsigned int i = 0; i < totalndata; i++)
      	{
      		int currentChannel = i % nchan;
      		int currentChunk = i / chunkData;
      		
      		int8_t data_re;
      		int8_t data_im;
      		switch (i % 4)
      		{
      			case 0:
      				data_re = (data[i/4] >> 7) & 1;
      				data_im = (data[i/4] >> 6) & 1;
      				break;
      			case 1:
      				data_re = (data[i/4] >> 5) & 1;
      				data_im = (data[i/4] >> 4) & 1;
      				break;
      			case 2:
      				data_re = (data[i/4] >> 3) & 1;
      				data_im = (data[i/4] >> 2) & 1;
      				break;
      			case 3:
      				data_re = (data[i/4] >> 1) & 1;
      				data_im = data[i/4] & 1;
      				break;
      		}
      		
      		//this sets 0 to -1 and 1 to 1
      		data_re = 2 * data_re - 1;
      		data_im = 2 * data_im - 1;
      		
      		sum[currentChunk * nchan + currentChannel] += data_re * data_re + data_im * data_im;
      	}
}

void bin_crosses_2bit(const uint8_t *pol0, const uint8_t *pol1, int *sum, int ndata, int nchan, int chunk)
{
	const uint8_t mask = 3;
	const int nchunk = ndata/chunk;
	for (int i = 0; i < nchunk; i++)
	{
		unsigned int chunkIdx = 2 * i * nchan;
		for (int j=0;j<nchan;j++)
		{
      			sum[chunkIdx + 2 * j] = 0;
      			sum[chunkIdx + 2 * j + 1] = 0;
      		}
      	}
      	
      	unsigned int totalndata = nchunk * chunk * nchan;
      	unsigned int chunkData = chunk * nchan;
      	for (unsigned int i = 0; i < totalndata; i++)
      	{
      		int currentChannel = i % nchan;
      		int currentChunk = i / chunkData;
      		
      		int8_t pol0_re;
      		int8_t pol0_im;
      		int8_t pol1_re;
      		int8_t pol1_im;
      		if (i % 2 == 0) //some of the data is in the beginning four bytes, and some of the data is in the ending four bytes.
      		{
      			pol0_re = ((pol0[i/2] >> 6) & mask) - 1;
      			pol0_im = ((pol0[i/2] >> 4) & mask) - 1;
      			pol1_re = ((pol1[i/2] >> 6) & mask) - 1;
      			pol1_im = ((pol1[i/2] >> 4) & mask) - 1;
      		}
      		else
      		{
      			pol0_re = ((pol0[i/2] >> 2) & mask) - 1;
      			pol0_im = (pol0[i/2] & mask) - 1;
      			pol1_re = (pol1[i/2] >> 2 & mask) - 1;
      			pol1_im = (pol1[i/2] & mask) - 1;
      		}
      		
      		if (pol0_re <= 0) {pol0_re --;}
      		if (pol0_im <= 0) {pol0_im --;}
      		if (pol1_re <= 0) {pol1_re --;}
      		if (pol1_im <= 0) {pol1_im --;}
      		
      		sum[currentChunk * nchan * 2 + currentChannel * 2] += pol0_re * pol1_re + pol0_im * pol1_im;
      		sum[currentChunk * nchan * 2 + currentChannel * 2 + 1] += pol0_im * pol1_re - pol0_re * pol1_im;
      	}
}

void bin_autos_2bit(const uint8_t *data, int *sum, int ndata, int nchan, int chunk)
{
	const uint8_t mask = 3;
	const int nchunk = ndata/chunk;
	for (int i = 0; i < nchunk; i++)
	{
		unsigned int chunkIdx = i * nchan;
		for (int j=0;j<nchan;j++)
		{
      			sum[chunkIdx + j] = 0;
      		}
      	}
      	
      	unsigned int totalndata = nchunk * chunk * nchan;
      	unsigned int chunkData = chunk * nchan;
      	for (unsigned int i = 0; i < totalndata; i++)
      	{
      		int currentChannel = i % nchan;
      		int currentChunk = i / chunkData;
      		
      		int8_t data_re;
      		int8_t data_im;
      		if (i % 2 == 0) //some of the data is in the beginning four bytes, and some of the data is in the ending four bytes.
      		{
      			data_re = ((data[i/2] >> 6) & mask) - 1;
      			data_im = ((data[i/2] >> 4) & mask) - 1;
      		}
      		else
      		{
      			data_re = ((data[i/2] >> 2) & mask) - 1;
      			data_im = (data[i/2] & mask) - 1;
      		}
      		
      		if (data_re <= 0) {data_re --;}
      		if (data_im <= 0) {data_im --;}
      		
      		sum[currentChunk * nchan + currentChannel] += data_re * data_re + data_im * data_im;
      	}
}

void bin_crosses_4bit(const uint8_t *pol0, const uint8_t *pol1, int *sum, int ndata, int nchan, int chunk)
{
	const uint8_t rmask = 15;
	const uint8_t imask = 255-15;
	const int nchunk = ndata/chunk;
	for (int i = 0; i < nchunk; i++)
	{
		unsigned int chunkIdx = 2 * i * nchan;
		for (int j=0;j<nchan;j++) 
		{
      			sum[chunkIdx + 2 * j] = 0;
      			sum[chunkIdx + 2 * j + 1] = 0;
      			for (int k = 0; k < chunk; k++)
      			{
      				unsigned int idx = i * nchan * chunk + j + k * nchan;
      				
      				//getting the values
      				int8_t pol0_re = pol0[idx] & rmask;
      				int8_t pol0_im = (pol0[idx] & imask) >> 4;
      				int8_t pol1_re = pol1[idx] & rmask;
      				int8_t pol1_im = (pol1[idx] & imask) >> 4;
      				if (pol0_re > 8) {pol0_re -= 16;}
      				if (pol0_im > 8) {pol0_im -= 16;}
      				if (pol1_re > 8) {pol1_re -= 16;}
      				if (pol1_im > 8) {pol1_im -= 16;}
      				
      				sum[chunkIdx + 2 * j] += pol0_re * pol1_re + pol0_im * pol1_im;
      				sum[chunkIdx + 2 * j + 1] += pol0_im * pol1_re - pol0_re * pol1_im;
      			}
    		}
	}
}

void bin_autos_4bit(const uint8_t *data, int *sum, int ndata, int nchan, int chunk)
{
	const uint8_t rmask = 15;
	const uint8_t imask = 255-15;
	const int nchunk = ndata/chunk;
	for (int i = 0; i < nchunk; i++)
	{
		unsigned int chunkIdx = i * nchan;
		for (int j=0;j<nchan;j++) 
		{
      			sum[chunkIdx + j] = 0;
      			for (int k = 0; k < chunk; k++)
      			{
      				unsigned int idx = i * nchan * chunk + j + k * nchan;
      				
      				//getting the values
      				int8_t data_re = data[idx] & rmask;
      				int8_t data_im = (data[idx] & imask) >> 4;
      				if (data_re > 8) {data_re -= 16;}
      				if (data_im > 8) {data_im -= 16;}
      				
      				sum[chunkIdx + j] += data_re * data_re + data_im * data_im;
      			}
    		}
	}
}

void average_cross_correlations (const uint8_t *pol0, const uint8_t *pol1, float *averages, int ndata, int nchan, int chunk, short bit_depth)
{
	int sum [2 * nchan * (ndata/chunk)];
	
	if (bit_depth == 4)
	{
		bin_crosses_4bit(pol0, pol1, sum, ndata, nchan, chunk);
	}
	else if (bit_depth == 2)
	{
		bin_crosses_2bit(pol0, pol1, sum, ndata, nchan, chunk);
	}
	else if (bit_depth == 1)
	{
		bin_crosses_1bit(pol0, pol1, sum, ndata, nchan, chunk);
	}
	else printf("Average Cross Correlations Unknown Bit Depth\n");
	
	for (int i = 0; i < 2 * nchan; i++)
	{
		int sumssum = 0;
		for (int j = 0; j < ndata/chunk; j ++)
		{
			sumssum += sum[i + j * 2 * nchan];
		}
		//averages[i] = sumssum * 1.0 / (ndata/chunk) / chunk; //I divide by chunk size as the end so that it doesn't depend as much on the chunk size
		averages[i] = sumssum;
	}
}

void average_auto (const uint8_t *data, float *averages, int ndata, int nchan, int chunk, short bit_depth)
{
	int sum [nchan * (ndata/chunk)];
	
	if (bit_depth == 4)
	{
		bin_autos_4bit(data, sum, ndata, nchan, chunk);
	}
	else if (bit_depth == 2)
	{
		bin_autos_2bit(data, sum, ndata, nchan, chunk);
	}
	else if (bit_depth == 1)
	{
		bin_autos_1bit(data, sum, ndata, nchan, chunk);
	}
	else printf("Average Cross Correlations Unknown Bit Depth\n");
	
	for (int i = 0; i < nchan; i++)
	{
		int sumssum = 0;
		for (int j = 0; j < ndata/chunk; j ++)
		{
			sumssum += sum[i + j * nchan];
		}
		//averages[i] = sumssum * 1.0 / (ndata/chunk) / chunk; //normalization optional
		averages[i] = sumssum;
	}
}
