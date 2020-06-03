import numpy as np
import albatrostools
import glob
import time
from matplotlib import pyplot as plt
import read_4bit
from importlib import reload
import scio


plt.ion()

fnames=glob.glob('../../baseband/15634/*.raw')
fnames.sort()
fnames=fnames[3:-2]
#fnames=fnames[:5]                                                                                                                                                                                       
#fnames_autocross=glob.glob('../../data_auto_cross/15634/*')
#fnames_autocross.sort()
froot='../../data_auto_cross/15634/1563419861/'
spec0_snap=scio.read(froot+'pol00.scio')
spec1_snap=scio.read(froot+'pol11.scio')
spec01_snap=scio.read(froot+'pol01r.scio')+1J*scio.read(froot+'pol01i.scio')
t_start=np.fromfile(froot+'time_gps_start.raw',dtype='double')
t_stop=np.fromfile(froot+'time_gps_stop.raw',dtype='double')

dt=t_stop-t_start
#dt=np.diff(t_start) #this seems be more accurate than t_stop-t_start
t_per_spec=2048*2/250e6
nspec_approx=np.median(dt)/t_per_spec




items=-1
bin_size=np.int(nspec_approx)


for file_name in fnames:
    #print(file_name)
    t1=time.time();
    header,stuff=albatrostools.get_data(file_name,unpack_fast=True,float=True,byte_delta=-8)
    t2=time.time()
    spec0=albatrostools.bin_autos(stuff['pol0'],bin_size)
    spec1=albatrostools.bin_autos(stuff['pol1'],bin_size)
    #pol1_rollx=np.empty(stuff['pol1'].shape,dtype=stuff['pol1'].dtype)
    #pol1_rolly=np.empty(stuff['pol1'].shape,dtype=stuff['pol1'].dtype)
    #pol1_rollx[:]=np.roll(stuff['pol1'],1,axis=1)
    #pol1_rolly[:]=np.roll(stuff['pol1'],1,axis=0)
    spec01=albatrostools.bin_crosses(stuff['pol1'],stuff['pol0'],bin_size)
    #spec01x=albatrostools.bin_crosses(stuff['pol0'],pol1_rollx,bin_size)
    #spec01y=albatrostools.bin_crosses(stuff['pol0'],pol1_rolly,bin_size)
    t3=time.time()
    try:
        big_spec0=np.append(big_spec0,spec0,axis=0)
        big_spec1=np.append(big_spec1,spec1,axis=0)
        big_spec01=np.append(big_spec01,spec01,axis=0)
        #big_spec01x=np.append(big_spec01x,spec01x,axis=0)
        #big_spec01y=np.append(big_spec01y,spec01y,axis=0)
    except:
        big_spec0=spec0
        big_spec1=spec1
        big_spec01=spec01
        #big_spec01x=spec01x
        #big_spec01y=spec01y
    t4=time.time();
    print(file_name,t2-t1,t3-t2,t4-t3,t4-t1)


i1=header['channels'][0]
i2=np.int(header['length_channels']+i1)
bit_guess=11.2

plt.clf();
plt.plot(np.median(np.real(big_spec01),axis=0))
plt.plot(np.median(np.real(spec01_snap[:,i1:i2]),axis=0)/2**bit_guess)

plt.plot(np.median(np.imag(big_spec01),axis=0))
plt.plot(np.median(np.imag(spec01_snap[:,i1:i2]),axis=0)/2**bit_guess)
plt.legend(['Real, 4bit','Real, Direct','Im, 4 bit','Im, Direct'])
plt.title('XCorr Real/Im, 4bit vs. Direct')

