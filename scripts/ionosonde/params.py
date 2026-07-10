# This file is the home for constants that don't change (or at least change very rarely)

import numpy as np
import os

c = 3e5 # km/s
len_pfb_init = 4096 # Length of the orginial PFB tap
# ipfb_chunk_size = int(5e5) # 409600 # 1000000

code0_str = '0001_0010_0001_1101' # complementary code pair, code 0
code1_str = '0001_0010_1110_0010' # complementary code pair, code 1
code_snr_boost = 6
code_len = 16
code_baudrate = 200e3 # Hz, sampling rate for code
ipp = 5.5e-3 # s, interpulse pulse period
code_repeat_num = 10 # number of repetitions of code_pattern

dac_rate = 250e6 # Hz, sampling rate of output to DAC
dac_bitwidth = 16 # bitwidth of output to DAC



# Bandwidth
# bw = 104

adc_samp_freq = 250e6

chan_res_init = adc_samp_freq / len_pfb_init # Channel resolution of original FT

num_ant = 1
num_pol = 2
num_pfb_tap = 4
cutsize = 16
# read_size = ipfb_chunk_size - 2 * cutsize
filt_thresh = 0.2
filter_len = 512
buf_len = 4096

all_freqs = np.loadtxt(f"{os.path.expanduser('~')}/albatros_analysis/scripts/ionosonde/eureka_freqs_hz.csv", skiprows = 1)
ionosonde_freqs = all_freqs[36:72]