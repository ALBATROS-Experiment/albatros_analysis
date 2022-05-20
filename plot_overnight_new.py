import os, sys
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	mpl.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
import scio, datetime, time, re
import SNAPfiletools as sft
import argparse
from datetime import datetime
import matplotlib.dates as mdates

#============================================================
def get_data_arrs(data_dir, ctime_start, ctime_stop):
	'''
	Given the path to a Big data directory (i.e. directory contains the directories 
	labeled by the first 5 digits of the ctime date), gets all the data in some time interval.

	Parameters:
	-----------

	data_dir: str
		path to data directory

	ctime_start, ctime_stop: str
		desired start and stop time in ctime

	Returns:
	--------

	cimte_start, ctime_stop: int
		start and stop times in ctime

	pol00,pol11,pol01r,pol01i: array
		2D arrays containing the data for given time interval for autospectra 
		as well as cross spectra. pol00 corresponds to adc0 and pol11 to adc3
	'''

	print("\n################### READING DATA ###################")
	print(f'Getting data from timestamps {ctime_start} to {ctime_stop}')
	print(f"In UTC time: {datetime.utcfromtimestamp(ctime_start)} to {datetime.utcfromtimestamp(ctime_stop)}")
	print(f"In local time: {datetime.fromtimestamp(ctime_start)} to {datetime.fromtimestamp(ctime_stop)}")

	#all the dirs between the timestamps. read all, append, average over chunk length
	data_subdirs = sft.time2fnames(ctime_start, ctime_stop, data_dir)
	
	new_dirs = [d+'/pol00.scio.bz2' for d in data_subdirs]
	datpol00 = scio.read_files(new_dirs)
	new_dirs = [d+'/pol11.scio.bz2' for d in data_subdirs]
	datpol11 = scio.read_files(new_dirs)
	new_dirs = [d+'/pol01r.scio.bz2' for d in data_subdirs]
	datpol01r = scio.read_files(new_dirs)
	new_dirs = [d+'/pol01i.scio.bz2' for d in data_subdirs]
	datpol01i = scio.read_files(new_dirs)

	t1=time.time()
	pol00=datpol00[0]
	for d in datpol00[1:]:
		pol00=np.append(pol00,d,axis=0)
	

	pol11=datpol11[0]
	for d in datpol11[1:]:
		pol11=np.append(pol11,d,axis=0)

	pol01r=datpol01r[0]
	for d in datpol01r[1:]:
		pol01r=np.append(pol01r,d,axis=0)

	pol01i=datpol01i[0]
	for d in datpol01i[1:]:
		pol01i=np.append(pol01i,d,axis=0)

	t2=time.time()
	print('Time taken to concatenate data:',t2-t1)
	print("pol00, pol11,pol01r, pol01i shape:", pol00.shape,pol11.shape,pol01r.shape,pol01i.shape)
	
	return pol00, pol11, pol01r, pol01i


#============================================================
def get_avg(arr,block=50):
	'''
	Averages some array over a given block size
	'''
	
	iters=arr.shape[0]//block
	leftover=arr.shape[0]%block
	print(arr.shape, iters, leftover)
	if(leftover>0.5*block):
	    result=np.zeros((iters+1,arr.shape[1]))
	else:
	    result=np.zeros((iters,arr.shape[1]))
	
	for i in range(0,iters):
	    result[i,:] = np.mean(arr[i*block:(i+1)*block,:],axis=0)
    
	if(leftover>0.5*block):
	    result[-1,:] = np.mean(arr[iters*block:,:],axis=0)
        
	return result

#============================================================
def get_stats(data_arr):
	'''
	Given a 2D array containing some data chunk, returns the 
	min, median, mean, and max over that chunk.
	'''
	if logplot:
		stats = {"min":np.log10(np.min(data_arr,axis=0)), "median":np.log10(np.median(data_arr,axis=0)), 
			    "mean":np.log10(np.mean(data_arr,axis=0)), "max":np.log10(np.max(data_arr,axis=0))}
	else:
		stats = {"min":np.min(data_arr,axis=0), "median":np.median(data_arr,axis=0), 
			    "mean":np.mean(data_arr,axis=0), "max":np.max(data_arr,axis=0)}
	return stats

def get_vmin_vmax(data_arr):
	'''
	Automatically gets vmin and vmax for colorbar
	'''
	med = np.median(data_arr)
	pmax = np.max(data_arr)

	xx=np.ravel(data_arr).copy()
	u=np.percentile(xx,99)
	b=np.percentile(xx,1)
	xx_clean=xx[(xx<=u)&(xx>=b)] # remove some outliers for better plotting
	stddev = np.std(xx_clean)
	vmin= max(med - 2*stddev,10**7)
	vmax = med + 2*stddev

	return vmin,vmax   

def get_ylim_times(t_i,t_f,tz):
	'''
	Gets the y limits in matplotlib's date format for a given initial time
	and final time. t_i and t_f must be given in ctime
	'''

	if tz=="utc":
		y_lims = list(map(datetime.utcfromtimestamp, [t_i, t_f]))
	elif tz=="local":
		y_lims = list(map(datetime.fromtimestamp, [t_i, t_f]))
	else:
		print("Invalid timezone")

	y_lims_plt = mdates.date2num(y_lims)
	return y_lims_plt

#================= plotting functions =======================
def full_plot(data_arrs):
	'''
	Makes a plot that contains autospectra waterfalls for each pol, as well
	as some statistics (min,max,med,mean spectra), and cross spectra
	'''

	pol00,pol11,pol01 = data_arrs

	pol00_stats = get_stats(pol00)
	pol11_stats = get_stats(pol00)
	
	
	if logplot is True:
		pol00 = np.log10(pol00)
		pol11 = np.log10(pol11)
	
	y_extent = get_ylim_times(ctime_start,ctime_stop,timezone)
	myext = np.array([0,125,y_extent[1],y_extent[0]])
		
	plt.figure(figsize=(18,10), dpi=200)
	plt.subplot(2,3,1)

	plt.imshow(pol00, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
	plt.title('pol00')
	cb00 = plt.colorbar()
	
	plt.subplot(2,3,4)
	plt.imshow(pol11, vmin=vmin, vmax=vmax, aspect='auto', extent=myext)
	plt.title('pol11')
	plt.colorbar()

	plt.subplot(2,3,2)
	plt.title('Basic stats for frequency bins')
	plt.plot(freq, pol00_stats["max"], 'r-', label='Max')
	plt.plot(freq, pol00_stats["min"], 'b-', label='Min')
	plt.plot(freq, pol00_stats["mean"], 'k-', label='Mean')
	plt.plot(freq, pol00_stats["median"], color='#666666', linestyle='-', label='Median')
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('pol00')
	

	plt.subplot(2,3,5)
	plt.plot(freq, pol11_stats["max"], 'r-', label='Max')
	plt.plot(freq, pol11_stats["min"], 'b-', label='Min')
	plt.plot(freq, pol11_stats["mean"], 'k-', label='Mean')
	plt.plot(freq, pol11_stats["median"], color='#666666', linestyle='-', label='Median')
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('pol11')
	
	plt.legend(loc='lower right', fontsize='small')

	plt.subplot(2,3,3)
	plt.imshow(np.log10(np.abs(pol01)), vmin=3,vmax=8,aspect='auto', extent=myext)
	plt.title('pol01 magnitude')
	plt.colorbar()

	plt.subplot(2,3,6)
	plt.imshow(np.angle(pol01), vmin=-np.pi, vmax=np.pi, aspect='auto', extent=myext, cmap='RdBu')
	plt.title('pol01 phase')
	plt.colorbar()

	plt.suptitle('Averaged over {} chunks'.format(blocksize))

	outfile = os.path.join(outdir,'output'+ '_' + str(ctime_start) + '_' + str(ctime_stop) + '.png')
	plt.savefig(outfile)
	
	print('Wrote ' + outfile)



#============================================================
def main():

	parser = argparse.ArgumentParser()
	# parser.set_usage('python plot_overnight_data.py <data directory> <start time as YYYYMMDD_HHMMSS or ctime> <stop time as YYYYMMDD_HHMMSS or ctime> [options]')
	# parser.set_description(__doc__)
	parser.add_argument('data_dir', type=str,help='Direct data directory')
	parser.add_argument("time_start", type=str, help="Start time YYYYMMDD_HHMMSS or ctime")
	parser.add_argument("time_stop", type=str, help="Stop time YYYYMMDD_HHMMSS or ctime")
	parser.add_argument('-o', '--outdir', dest='outdir',type=str, default='.',
			  help='Output plot directory [default: .]')
	
	parser .add_argument('-n', '--length', dest='readlen', type=int, default=1000, help='length of integration time in seconds')
	parser.add_argument("-a", "--avglen",dest="blocksize",default=0,type=int,help="number of chunks (rows) of direct spectra to average over. One chunk is roughly 6 seconds.")

	parser.add_argument("-l", "--logplot", dest='logplot', default = True, action="store_true", help="Plot in logscale")
	parser.add_argument("-p", "--plottype",dest="plottype",default="full",type=str,
		help="Type of plot to generate. 'full': pol00 and pol11 waterfall autospectra, min/max/mean/med autospectra, waterfall cross spectra. 'autospec': same as 1, but no cross spectra")
	parser.add_argument("-t", "--timezone", dest='timezone', default = "utc", type=str, help="Timezone to use for plot axis. Can do 'utc' or 'local'")
	parser.add_argument("-vmi", "--vmin", dest='vmin', default = None, type=int, help="minimum for colorbar. if nothing is specified, vmin is automatically set")
	parser.add_argument("-vma", "--vmax", dest='vmax', default = None, type=int, help="maximum for colorbar. if nothing is specified, vmax is automatically set")
	

	args = parser.parse_args()

	#defining some global variables 
	global freq, timezone, logplot, vmin, vmax, ctime_start, ctime_stop, blocksize, outdir
	
	timezone = args.timezone
	vmin = args.vmin
	vmax = args.vmax
	logplot=args.logplot
	blocksize = args.blocksize
	outdir = args.outdir

	
	# figuring out if human time or ctime was passed with pattern matching
	human_time_regex = re.compile((r'\d\d\d\d\d\d\d\d_\d\d\d\d\d\d'))
	mo = human_time_regex.search(args.time_start)

	try:
		match = mo.group() # time_start matches pattern, must convert to ctime
		ctime_start = sft.timestamp2ctime(time_start)
		ctime_stop = sft.timestamp2ctime(time_stop)
	except:
		ctime_start = int(args.time_start)
		ctime_stop = int(args.time_stop)
		
	
	pol00,pol11,pol01r,pol01i = get_data_arrs(args.data_dir, ctime_start, ctime_stop)
	
	if blocksize != 0: #averages over given blocksize
		pol00=get_avg(pol00,block=args.blocksize)
		pol11=get_avg(pol11,block=args.blocksize)
		pol01r=get_avg(pol01r,block=args.blocksize)
		pol01i=get_avg(pol01i,block=args.blocksize)

	pol01 = pol01r + 1J*pol01i
	
	freq = np.linspace(0, 125, np.shape(pol00)[1]) #125 MHz is max frequency


	if vmin==None and vmax==None:
		vmin,vmax = get_vmin_vmax(pol00)
		
	if logplot==True:
		vmin = np.log10(vmin)
		vmax = np.log10(vmax)

	if args.plottype == "full":
		full_plot([pol00,pol11,pol01])
	
if __name__ == '__main__':
	main()