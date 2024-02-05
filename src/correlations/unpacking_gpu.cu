#include <stdio.h>
#include <stdint.h>
#include "/usr/include/cuda_runtime.h"
#include "/usr/include/device_launch_parameters.h"

__global__ void dropped_packets_kernel(uint8_t *data, unsigned long *spec_num, unsigned int num_packets, const int spectra_per_packet, unsigned int entries_per_packet)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < num_packets - 1)
	{
		if (spec_num[i + 1] - spec_num[i] != spectra_per_packet)
		{
			const unsigned int initial = entries_per_packet * i;
			const unsigned int bound = initial + entries_per_packet;
			for (unsigned int j = initial; j < bound; j++)
			{
				data[j] = 0;
			}
		}
	}
}

extern "C" {
unsigned int dropped_packets (uint8_t *data, unsigned long *spec_num, unsigned int num_packets, const int spectra_per_packet, int nchan, short bit_depth)
{
	if (num_packets >= 2)
	{
		unsigned int num_dropped = 0;
		for (unsigned int j = 0; j < num_packets - 1; j++)
		{
			if (spec_num[j + 1] - spec_num[j] != spectra_per_packet)
			{
				num_dropped++;
			}
		}
		
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

		uint8_t* d_data = NULL;
		unsigned long int totalDataSize = num_packets * entries_per_packet * sizeof(uint8_t);
		cudaMalloc((void**)&d_data, totalDataSize); //make this size work better I think
		cudaMemcpy(d_data, data, totalDataSize, cudaMemcpyHostToDevice);
		unsigned long* d_spec_num = NULL;
		cudaMalloc((void**)&d_spec_num, num_packets * sizeof(long));
		cudaMemcpy(d_spec_num, spec_num, num_packets * sizeof(long), cudaMemcpyHostToDevice);
		dropped_packets_kernel<<<512, (num_packets + 511)/512>>> (d_data, d_spec_num, num_packets, spectra_per_packet, entries_per_packet);
		
		cudaFree(d_spec_num);
		cudaMemcpy(data, d_data, totalDataSize, cudaMemcpyDeviceToHost);
		cudaFree(d_data);
		
		return num_dropped;
	}
	else 
	{
		printf("There are fewer than 2 packets in the dropped packets function\n");
		return 0;
	}
}
}








































