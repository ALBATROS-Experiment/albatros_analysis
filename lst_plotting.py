import numpy as np
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

def get_localtime_from_UTC(tstamp, mytz):
    return datetime.fromtimestamp(int(tstamp),tz=pytz.utc).astimezone(tz=mytz)

def get_vmin_vmax(data_arr,log=True):
    '''
    Automatically gets vmin and vmax for colorbar
    '''
    # print("shape of passed array", data_arr.shape, data_arr.dtype)
    xx=data_arr.copy()
    med = np.percentile(xx,50)
    # print(med, "median")
    u=np.percentile(xx,99)
    b=np.percentile(xx,1)
    xx_clean=xx[(xx<=u)&(xx>=b)] # remove some outliers for better plotting
    stddev = np.std(xx_clean)
    vmin= max(med - 1*stddev,10**7)
    vmax = med + 1*stddev
    print(vmax,stddev)
    # print("vmin, vmax are", vmin, vmax)
    return np.log10(vmin),np.log10(vmax)

fpath='/project/s/sievers/mohanagr/lst_720_median_1661011607_1666620593_uapishka.npz'

#Data file also has mean if you want to plot that. Refer to lst_binning.py to see field names.
with np.load(fpath) as npz:
        pol00 = npz['p00median']
        pol11 = npz['p11median']
        pol01r = npz['p01rmedian']
        pol01i = npz['p01imedian']
        counts = npz['counts']

tags=fpath.split('_')
plot_type='median'
ctime_start=tags[3]
ctime_stop=tags[4]
mytz=pytz.timezone('US/Eastern')
sttime=get_localtime_from_UTC(ctime_start,mytz).strftime("%b-%d %H:%M")
entime=get_localtime_from_UTC(ctime_stop,mytz).strftime("%b-%d %H:%M")

pol01 = pol01r + 1J*pol01i
nbins=720
f=plt.gcf()
f.set_size_inches(15,15)

plt.suptitle(f'Plotting from: {sttime} to {entime}, plot type: {plot_type}')
myext=[0, 125, 24, 0]
plt.subplot(321)
plt.title("Pol00")
vmin,vmax=get_vmin_vmax(pol00)
plt.imshow(np.log10(pol00),vmin=vmin,vmax=vmax,extent=myext,aspect='auto')
plt.colorbar()

plt.subplot(323)
plt.title("Pol11")
plt.imshow(np.log10(pol11),vmin=vmin,vmax=vmax,extent=myext,aspect='auto')
plt.colorbar()

plt.subplot(322)
plt.title("Pol01 mag")
plt.imshow(np.log10(np.abs(pol01)),vmin=3,vmax=8,extent=myext,aspect='auto')
plt.colorbar()

plt.subplot(324)
plt.title("Pol01 phase")
plt.imshow(np.angle(pol01),extent=myext,aspect='auto',cmap='RdBu')
plt.colorbar()

plt.subplot(313)
plt.title('bin count')
plt.plot(np.arange(0,nbins),counts)

output_path = f'/home/s/sievers/mohanagr/lst_{nbins}.jpg'
plt.savefig(output_path)
print(output_path)