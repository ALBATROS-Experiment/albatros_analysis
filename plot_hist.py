from correlations import baseband_data_classes as bdc
import numpy as np
import argparse
from matplotlib import pyplot as plt
import os
from palettable.colorbrewer.sequential import GnBu_9 as mycmap

if(__name__=='__main__'):
	"Example usage: python quick_spectra.py ~/data_auto_cross/16171/1617100000"
	parser = argparse.ArgumentParser()
	parser.add_argument("filepath", type=str, help="Baseband file locaion. Ex: ~/snap1/16171/161700000/161700026.raw")
	parser.add_argument("-o", "--output_dir", type=str, default="./", help="Output directory for plots")
	parser.add_argument("-m", "--mode", type=int, default=-1, help="0 for pol0, 1 for pol1, -1 for both")
	parser.add_argument("-r", "--rescale", action="store_true", help="Map bit values (0-15 for 4 bit data) to -ve to +ve levels.")
	parser.add_argument("-c", '--chans', type=int, nargs=2, help="Indices of start and end channels to print out.")
	args = parser.parse_args()

	obj=bdc.Baseband(args.filepath)
	hist=obj.get_hist(mode=args.mode)
	print('Hist vals shape: \n',hist.shape)
	# np.savetxt('./hist_dump_mohan_laptop.txt',hist) this was to check output against code on niagara. all match.
	nlevels=2**obj.bit_mode
	if(args.rescale and obj.bit_mode==4):
		bins = np.arange(-7,8)
		hist = np.fft.fftshift(hist,axes=0) #first row would correspond to -8 which is 0
		assert np.all(hist[0,:]==0)
		hist = hist[1:,:].copy()
	elif(args.rescale and obj.bit_mode==1):
		bins = [-1,1]
	else:
		bins = np.arange(0,nlevels)
	print(bins)
	print(f"total data points: {hist.sum()}")
	snap,five_digit,timestamp=args.filepath.split('/')[-3:]
	timestamp=timestamp.split('.')[0]

	f=plt.gcf()
	f.set_size_inches(10,4)
	if(args.mode in (0,1)):
		tag='pol'+str(args.mode)
	else:
		tag='both_pols'

	plt.suptitle(f'Histogram for {snap} {timestamp} {tag}')
	plt.subplot(121)
	if(args.chans):
		print("Per chan hist is", hist[:,args.chans[0]:args.chans[1]])
	else:
		print("Per chan hist is", hist[:,:])
	plt.imshow(hist,aspect="auto",interpolation='none',cmap=mycmap.mpl_colormap)
	# ax=plt.gca()
	# ax.yaxis.set_major_locator(bins)
	freqs=obj.channels
	locs,labels=plt.xticks()
	locs=np.arange(0,len(obj.channels))
	labels=[str(x) for x in obj.channels]
	# osamp=len(obj.channels)//64
	osamp=int(obj.length_channels//32)
	print("OSAMP IS ", osamp, locs[::osamp])
	plt.xticks(locs[::osamp],labels[::osamp],rotation=-50)
	# print(locs,labels)

	locs,labels=plt.yticks()
	locs=np.arange(0,len(bins))
	labels=bins
	plt.yticks(locs,labels)
	plt.colorbar()
	plt.xlabel('channels')

	plt.subplot(122)
	hist_total = np.sum(hist,axis=1)
	plt.bar(bins,hist_total,label=f'mode={args.mode}')
	plt.xticks(bins,bins)
	plt.tight_layout()

	
	fname = os.path.join(args.output_dir,f'hist_{snap}_{timestamp}_{tag}.png')
	plt.savefig(fname)
	print(fname)

	