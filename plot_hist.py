from correlations import baseband_data_classes as bdc
import numpy as np
import argparse
from matplotlib import pyplot as plt
import os

if(__name__=='__main__'):
	"Example usage: python quick_spectra.py ~/data_auto_cross/16171/1617100000"
	parser = argparse.ArgumentParser()
	parser.add_argument("filepath", type=str, help="Baseband file locaion. Ex: ~/snap1/16171/161700000/161700026.raw")
	parser.add_argument("-o", "--output_dir", type=str, default="./", help="Output directory for plots")
	parser.add_argument("-m", "--mode", type=int, default=-1, help="0 for pol0, 1 for pol1, -1 for both")
	parser.add_argument("-r", "--rescale", action="store_true", help="Map bit values (0-15 for 4 bit data) to -ve to +ve levels.")
	args = parser.parse_args()

	obj=bdc.Baseband(args.filepath)
	hist=obj.get_hist(mode=args.mode)
	print('Hist vals: \n',hist)
	if(args.rescale):
		bins = np.fft.fftshift(np.fft.fftfreq(16)*16)
		hist = np.fft.fftshift(hist)
		
	else:
		bins = np.arange(0,16)
	
	print(f"total data points: {hist.sum()}")
	
	fig,ax = plt.subplots(1,1)
	ax.bar(bins,hist,label=f'mode={args.mode}')
	fname = os.path.join(args.output_dir,'hist.png')
	plt.savefig(fname)

	