# This file is the home for constants that don't change (or at least change very rarely)

import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser('Ionosonde')

parser.add_argument('--c', default=3e5, type=float, help='Speed of light in km/s.')

# FPGA/ADC/Inital PFB
parser.add_argument('--len_pfb_init', default=4096, type=int, help='Length of the FFTs taken by the FPGA.')
parser.add_argument('--num_pfb_tap', default=4, type=int, help='Number of taps in the inital PFB.')
parser.add_argument('--adc_samp_freq', default=250e6, type=float, help='Analog-to-digital converter (ADC) sample frequency.')
parser.add_argument('--channels', default=np.arange(64,168), type=int, nargs="+")

# Codes
parser.add_argument('--code0_str', default='0001_0010_0001_1101', type=str, help='Code 0 of the complementary code pair.')
parser.add_argument('--code1_str', default='0001_0010_1110_0010', type=str, help='Code 1 of the complementary code pair.')
parser.add_argument('--code_snr_boost', default=6, type=int, help = 'Number of times each code symbol is repeated.')
parser.add_argument('--code_len', default=16, type=int, help = 'Length of codes.')
parser.add_argument('--code_baudrate', default=200e3, type=float, help='Code baudrate.')
parser.add_argument('--template_dt', type=float, help='Sample spacing for smoothed code sequence.')
parser.add_argument('--ipp', default=115501000/1e9/21, type=float, help='Interpulse period, i.e. the time between sucessive code transmissions.')
parser.add_argument('--code_repeat_num', default=10, type=int, help='Number of repetitions of code_pattern.')
parser.add_argument('--trans_len', type=int, help='Calculate transmission length.')

parser.add_argument('--which_ant', default=2, type=float, help='Antenna number.')
parser.add_argument('--start_time', default=1746795000, type=int)
parser.add_argument('--corr_time', default=15, type=int)
parser.add_argument('--baseband_dir', default="/scratch/mohanagr/drive2_mars_spring2025/baseband/", type=str)
parser.add_argument('--out_dir', default="/scratch/mayas/ionograms_testing", type=str)
parser.add_argument('--corr_name', default="correlated_data.npz", type=str)
parser.add_argument('--num_pol', default=2, type=int, help='Number of polarizations.')
parser.add_argument('--num_ant', default=1, type=int, help='Number of anntenna.')

parser.add_argument('--cutsize', default=16, type=int, help="???????")
parser.add_argument('--filt_thresh', default=0.2, type=float, help="Filter threshold.")
parser.add_argument('--filter_len', default=512, type=int, help="Filter length.")
parser.add_argument('--buf_len', default=4096, type=int, help="Buffer length.")

parser.add_argument('--iono_freqs_path', default=f"{os.path.expanduser('~')}/albatros_analysis/scripts/ionosonde/eureka_freqs_hz.csv",
                    help='Path to CSV file with list of ionosonde frequencies.')
parser.add_argument('--freq_idx_bounds', default=[36, 72], nargs=2, type=int, help='Which frequencies to actually analyze.')
parser.add_argument('-f', '--f', type=str, help='Just here so that IPython works.')

default_args = parser.parse_args()

# Calculate transmission length
if default_args.trans_len == None:
    default_args.trans_len = (2 * default_args.code_repeat_num + 1) * default_args.ipp
# Calculate channel resolution of orginial Fourier transform
default_args.chan_res_init = default_args.adc_samp_freq / default_args.len_pfb_init

if default_args.template_dt == None:
    default_args.template_dt = 1 / default_args.code_baudrate

default_args.all_freqs = np.loadtxt(default_args.iono_freqs_path, skiprows = 1)
default_args.ionosonde_freqs = default_args.all_freqs[default_args.freq_idx_bounds[0]:default_args.freq_idx_bounds[1]]

if __name__ == "__main__":
    print(default_args)
