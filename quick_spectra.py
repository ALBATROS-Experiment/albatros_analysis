import numpy as np
import os
import matplotlib.pyplot as plt
from  scio import scio
import argparse
import pytz
import datetime as dt

def get_acctime(fpath):
	dat = np.fromfile(fpath,dtype='uint32')
	diff = np.diff(dat)
	acctime = np.mean(diff[(diff>0)&(diff<100)]) #sometimes timestamps are 0, which causes diff to be huge.
	return acctime


if __name__ == "__main__":
	"Example usage: python quick_spectra.py ~/data_auto_cross/16171/1617100000"
	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", type=str, help="Auto/cross-spectra location. Ex: ~/data_auto_cross/16171/161700000")
	parser.add_argument("-o", "--output_dir", type=str, default="./", help="Output directory for plots")
	parser.add_argument("-l", "--logplot", action="store_true", help="Plot in logscale")
	parser.add_argument("-s", "--show", action="store_true", help="Show final plot")
	parser.add_argument("-tz", "--timezone", type=str, default='US/Eastern', help="Valid timezone of the telescope recognized by pytz. E.g. US/Eastern. Default is US/Eastern.")
	args = parser.parse_args()

	#data_dir = pathlib.Path(args.data_dir)
	#output_dir = pathlib.Path(args.output_dir)

	pol00 = scio.read(args.data_dir + "/pol00.scio.bz2")
	pol11 = scio.read(args.data_dir + "/pol11.scio.bz2")
	pol01r = scio.read(args.data_dir + "/pol01r.scio.bz2")
	pol01i = scio.read(args.data_dir + "/pol01i.scio.bz2")
	acctime = get_acctime(args.data_dir + 'time_gps_start.raw')
	# Remove starting data chunk if it's bad :(
	pol00 = pol00[1:,:]
	pol11 = pol11[1:,:]
	pol01r = pol01r[1:,:]
	pol01i = pol01i[1:,:]
	# Add real and image for pol01	
	pol01 = pol01r + 1J*pol01i

	freq = np.linspace(0, 125, np.shape(pol00)[1])

	pol00_med = np.median(pol00, axis=0)
	pol11_med = np.median(pol11, axis=0)
	pol00_mean = np.mean(pol00, axis=0)
	pol11_mean = np.mean(pol11, axis=0)
	pol00_max = np.max(pol00, axis=0)
	pol11_max = np.max(pol11, axis=0)
	pol00_min = np.min(pol00, axis=0)
	pol11_min = np.min(pol11, axis=0)

	med = np.median(pol00)

	xx=np.ravel(pol00).copy()
	u=np.percentile(xx,99)
	b=np.percentile(xx,1)
	xx_clean=xx[(xx<=u)&(xx>=b)] # remove some outliers for better plotting
	stddev = np.std(xx_clean)
	vmin= max(med - 2*stddev,10**7)
	vmax = med + 2*stddev
	
	pmax = np.max(pol00)
	axrange = [0, 125, 0, pmax]

	if args.logplot == True:
		pol00 = np.log10(pol00)
		pol11 = np.log10(pol11)
		pol00_med = np.log10(pol00_med)
		pol11_med = np.log10(pol11_med)
		pol00_mean = np.log10(pol00_mean)
		pol11_mean = np.log10(pol11_mean)
		pol00_max = np.log10(pol00_max)
		pol11_max = np.log10(pol11_max)
		pol00_min = np.log10(pol00_min)
		pol11_min = np.log10(pol11_min)
		med = np.log10(med)
		vmin = np.log10(vmin)
		vmax = np.log10(vmax)
		pmax = np.log10(pmax)
		axrange = [0, 125, 6.5, pmax]

	print("Estimated accumulation time from timestamp file: ", acctime)
	tot_minutes = int(np.ceil(acctime * pol00.shape[0]/60))
	myext = np.array([0, 125, tot_minutes, 0])

	plt.figure(figsize=(18,10), dpi=200)

	plt.subplot(2,3,1)
	plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
	plt.title('pol00')
	cb00 = plt.colorbar()
	cb00.ax.plot([0, 1], [7.0]*2, 'w')

	plt.subplot(2,3,4)
	plt.imshow(pol11, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
	plt.title('pol11')
	plt.colorbar()

	plt.subplot(2,3,2)
	plt.title('Basic stats for frequency bins')
	plt.plot(freq, pol00_max, 'r-', label='Max')
	plt.plot(freq, pol00_min, 'b-', label='Min')
	plt.plot(freq, pol00_mean, 'k-', label='Mean')
	plt.plot(freq, pol00_med, color='#666666', linestyle='-', label='Median')
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('pol00')
	plt.axis(axrange)

	plt.subplot(2,3,5)
	plt.plot(freq, pol11_max, 'r-', label='Max')
	plt.plot(freq, pol11_min, 'b-', label='Min')
	plt.plot(freq, pol11_mean, 'k-', label='Mean')
	plt.plot(freq, pol11_med, color='#666666', linestyle='-', label='Median')
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('pol11')
	plt.axis(axrange)
	plt.legend(loc='lower right', fontsize='small')

	plt.subplot(2,3,3)
	plt.imshow(np.log10(np.abs(pol01)), vmin=3, vmax=8, aspect='auto', extent=myext)
	plt.title('pol01 magnitude')
	plt.colorbar()

	plt.subplot(2,3,6)
	plt.imshow(np.angle(pol01), vmin=-np.pi, vmax=np.pi, aspect='auto', extent=myext, cmap='RdBu')
	plt.title('pol01 phase')
	plt.colorbar()

	args.data_dir=os.path.abspath(args.data_dir)
	timestamp = args.data_dir.split('/')[-1]
	mytz = pytz.timezone(args.timezone)
	utctime = dt.datetime.fromtimestamp(int(timestamp),tz=pytz.utc) # this might be a safer way compared to passing tz directly to fromtimestamp
	localtimestr = utctime.astimezone(tz=mytz).strftime("%b-%d %H:%M:%S")
	plt.suptitle(f"Minutes since {localtimestr} localtime. File ctime {timestamp}")

	outfile = os.path.normpath(args.output_dir + '/' + timestamp + '.png')
	plt.savefig(outfile)
	print('Wrote ' + outfile)
	if args.show == True:
		plt.show()
	plt.close()


