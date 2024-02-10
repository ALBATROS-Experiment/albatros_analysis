import sys
import time
import os
sys.path.insert(0, "/home/s/sievers/mohanagr/")
from albatros_analysis.src.utils import baseband_utils as butils
from albatros_analysis.src.utils import orbcomm_utils as outils
from albatros_analysis.src.correlations import baseband_data_classes as bdc
import numpy as np
from scipy import stats

T_SPECTRA = 4096/250e6
T_ACCLEN = 393216 * T_SPECTRA

# get a list of all direct spectra files between two timestamps

t1 = 1627453681
t2 = int(t1 + 560*T_SPECTRA)

direct_files = butils.time2fnames(t1,t2,"/project/s/sievers/albatros/uapishka/202107/data_auto_cross/snap3")
print(direct_files)

#all the sats we track
satlist = [40086, 40087, 40091, 41179, 41182, 41183, 41184, 41185, 41186, 41187, 41188, 41189, 25338, 28654, 33591, 40069]
satmap = {}
assert(min(satlist)>len(satlist)) # to make sure there are no collisions, we'll never have an i that's also a satnum
for i,satnum in enumerate(satlist):
    satmap[i] = satnum
    satmap[satnum]=i
# print(satmap)

#for each file get the risen sats and divide them up into unique transits
a1_coords = [51.4646065, -68.2352594, 341.052] #north antenna
a2_coords = [51.46418956, -68.23487849, 338.32526665] #south antenna

for file in direct_files:
    tstart = butils.get_tstamp_from_filename(file)
    nrows=560
    tle_path = "/home/s/sievers/mohanagr/albatros_analysis/scripts/orbcomm/orbcomm_28July21.txt"
    arr = np.zeros((nrows,len(satlist)),dtype='int64')
    rsats = outils.get_risen_sats(tle_path,a1_coords,tstart,niter=nrows)
    print(rsats)
    for i,row in enumerate(rsats):
        for satnum, satele in row:
            arr[i][satmap[satnum]]=1
    print(arr)
    pulses = outils.get_simul_pulses(arr)
    print(pulses)

    for (pstart, pend), sats in pulses:
        print(pstart,pend,sats)
        numsats_in_pulse = len(sats)
        t1 = tstart + pstart*T_ACCLEN
        t2 = tstart + pend*T_ACCLEN
        print(t1,t2)
        files_a1,idx1=butils.get_init_info(t1,t2,'/project/s/sievers/albatros/uapishka/202107/baseband/snap3/')
        files_a2,idx2=butils.get_init_info(t1,t2,'/project/s/sievers/albatros/uapishka/202107/baseband/snap1/')
        print(files_a1)
        print(bdc.get_header(files_a1[0]))
        channels = bdc.get_header(files_a1[0])['channels']
        chanstart = np.where(channels==1834)[0][0]
        chanend = np.where(channels==1854)[0][0]
        nchans = chanend-chanstart
        # #a1 = antenna 1 = SNAP3
        # #a2 = antenna 2 = SNAP1
        size=3000000
        # #dont impose any chunk num, continue iterating as long as a chunk with small enough missing fraction is found.
        # #have passed enough files to begin with. should not run out of files.
        ant1=bdc.BasebandFileIterator(files_a1,0,idx1,size,None,chanstart=chanstart,chanend=chanend,type="float")
        ant2=bdc.BasebandFileIterator(files_a2,0,idx2,size,None,chanstart=chanstart,chanend=chanend,type="float")

        p0_a1 = np.zeros((size,nchans),dtype='complex64')
        p0_a2 = np.zeros((size,nchans),dtype='complex64')
        freq=250e6*(1-np.arange(chanstart,chanend)/4096).reshape(-1,nchans) # get actual freq from aliasedprint("FREQ",freq/1e6," MHz")
        niter=int(t2-t1)+1 #run it for an extra second to avoid edge effects
        print(niter)
        delays=np.zeros((size,len(sats)))
        for i,satID in enumerate(sats):
            d = outils.get_sat_delay(a1_coords, a2_coords,'./data/orbcomm_28July21.txt', tstart, niter, satmap[satID])
            delays[:, i] = np.interp(np.arange(0,size)*dt, np.arange(0,niter),d)
        for i, (chunk1,chunk2) in enumerate(zip(ant1,ant2)):
            perc_missing_a1 = (1 - len(chunk1["specnums"]) / size) * 100
            perc_missing_a2 = (1 - len(chunk2["specnums"]) / size) * 100
            if perc_missing_a1 > 5 or perc_missing_a2 > 5:
                continue
            outils.make_continuous(p0_a1, ['pol0'], a1['specnums']-a1['specnums'][0])
            outils.make_continuous(p0_a2, a2['pol0'], a2['specnums']-a2['specnums'][0])
        cx = []
        N=2*size
        dN=min(100000,int(0.3*N))
        stamp=slice(N//2-dN,N//2+dN)
        print(N,dN)
        cx.append(np.fft.fftshift(outils.get_coarse_xcorr(p0_a1,p0_a2),axes=1)[:,stamp]) #no correction
        for i,satID in enumerate(sats):
            print("processing satellite:", satmap[satID])
            phase_delay = 2*np.pi*delays[:,i:i+1]@freq
            print(phase_delay.shape)
            cx.append(np.fft.fftshift(outils.get_coarse_xcorr(p0_a1,p0_a2*np.exp(1J*phase_delay)),axes=1)[:,stamp])
        snr_arr = np.zeros((len(sats), nchans),dtype='float64')
        # save the SNR for each channel for each satellite
        for i in range(len(sats)+1):
            snr_arr[i,:] = np.max(np.abs(cx[i]),axis=1)/stats.median_abs_deviation(np.abs(cx[i]),axis=1)