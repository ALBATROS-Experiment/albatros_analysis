from correlations import baseband_data_classes as bdc
import numpy as np
import argparse
from matplotlib import pyplot as plt
import os
from palettable.colorbrewer.sequential import PuBuGn_8 as mycmap

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
	print('Hist vals shape: \n',hist.shape)
	np.savetxt('./hist_dump_mohan_laptop.txt',hist)
	if(args.rescale):
		bins = np.fft.fftshift(np.fft.fftfreq(16)*16)
		hist = np.fft.fftshift(hist,axes=0)
	else:
		bins = np.arange(0,16)
	
	print(f"total data points: {hist.sum()}")
	bbfile=args.filepath.split('/')[-1]
	bbfile=bbfile.split('.')[0]

	f=plt.gcf()
	f.set_size_inches(10,4)
	plt.suptitle(f'Histogram for {bbfile}.raw')
	plt.subplot(121)
	plt.imshow(hist,aspect="auto",interpolation='none',extent=[obj.channels[0],obj.channels[-1], bins[-1],bins[0]],cmap=mycmap.mpl_colormap)
	# ax=plt.gca()
	# ax.yaxis.set_major_locator(bins)
	# plt.xticks(obj.channels)
	plt.colorbar()
	plt.xlabel('channels')

	plt.subplot(122)
	hist_total = np.sum(hist,axis=1)
	plt.bar(bins,hist_total,label=f'mode={args.mode}')
	plt.tight_layout()

	
	fname = os.path.join(args.output_dir,f'hist_{bbfile}.png')
	plt.savefig(fname)
	print(fname)

	