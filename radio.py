import pylab

import numpy as np
import scipy.linalg as la

import pfb_helper as pfb


from scipy import signal
from scipy.io.wavfile import write


##bandpass functions copied from scipy doc
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def demodulate_chunk(data, fs, freq, audio_rate):
    """
        AM demodulates a chunk of time domain data sent to it 
            - data: self explanatory
            -fs: sample rate in Hz
            -freq: is carrier frequency in Hz
            -audio_rate: is the sample rate for the audio return in Hz
        Returns NumPy array of int16 to be saved to a .wav file
    """
    ##bandpass first
    band = 1e6
    radio_signal = np.clip(butter_bandpass_filter(data, freq-band, freq +band, fs, order=6),-1,1)
    #find endtime using sample rate
    endtime = np.shape(data)[0]/fs


    ##demodulation code!
    fc =5000  # Cut-off frequency of the filter in hz
    filtered =butter_lowpass_filter(np.abs(radio_signal), fc, fs/2) ##low pass the abs of the signal

    ##downsampling code
    sampleRate = audio_rate ##for audio here
    downsample = signal.resample(filtered, int(endtime*sampleRate))
    #mx = np.median(downsample) + 2*np.std(downsample)
    #mn = np.median(downsample) - 2*np.std(downsample)
    audio = np.clip(downsample, -1, 1)
    audio = audio - np.mean(audio)
    #next two lines is to get it windows readable
    audio = audio/ (max(np.max(audio),np.abs(np.min(audio)))) * 32767 
    audio = np.int16(audio)
    return audio




import albatrostools

#start by unpacking data with c code (will need to compile and set LD_LIBRARY_PATH)
#export LD_LIBRARY_PATH={LD_LIBRARY_PATH}:/mnt/c/Users/simta/Documents/GitHub/mars_2019_tools
#gcc -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp

fname = "data/1563719256.raw"
header, data = albatrostools.get_data(fname, items=100000, unpack_fast=False, float=True)

n_channels = header['channels'][-1]
#freq = np.linspace(0,(n_channels / 2048) * 128e6, n_channels)
fs = 125e6
time_stream = {
    'pol0' : [],
    'pol1' : []
}


spec_pfb = np.zeros((np.shape(data['pol0'])[0], n_channels),dtype=np.complex)
spec_pfb[:,int(header['channels'][0]) -1:] = data['pol0']
rts = np.clip(pfb.inverse_pfb(spec_pfb, 4).ravel(), -1,1)
time_stream['pol0'] = rts

sampleRate = 1500
audio = demodulate_chunk(time_stream['pol0'],fs, 10e6, sampleRate )
##write to file
write('test.wav',sampleRate,audio)
pylab.plot(audio)
pylab.show()

























#mn = np.median(time_stream['pol0']) - 2*np.std(time_stream['pol0'])
#mx = np.median(time_stream['pol0']) + 2*np.std(time_stream['pol0'])
#time_stream['pol0'] = np.clip(time_stream['pol0'], mn, mx)


# for i in range(np.shape(data['pol0'])[0]//5):
#     spec_pfb = np.zeros((5, n_channels),dtype=np.complex)
#     spec_pfb[:,int(header['channels'][0]) -1:] = data['pol0'][i:i+5,:]
#     rts = np.clip(pfb.inverse_pfb(spec_pfb, 4).ravel(), -1,1)
#     time_stream['pol0'] = np.append(time_stream['pol0'], rts)




# pylab.plot(time_stream['pol0'])
# pylab.show()
# for i in range (10):        
#     fname = "data/1563719256.raw"
#     header, data = albatrostools.get_data(fname, items=1, unpack_fast=False, float=True)
#     checksum = np.sum(np.abs(data['pol0']))%100
#     print(checksum)


# rts = pfb.inverse_pfb(data['pol0'], 4).ravel()
# spec_pfb = pfb.pfb(rts, 120, ntap=2)
# print(np.shape(rts))
#pylab.imshow(np.abs(spec_pfb), aspect='auto', interpolation='nearest')
#pylab.imshow(np.abs(data['pol0']), aspect='auto', interpolation='nearest')



































# fs = 2e6 #sample rate in Hz
# endtime = 999600/fs
# ta = np.linspace(0.0, endtime, int(endtime * fs), endpoint=False)
# sampleRate = 10000

# ##### fake data generation 
# ts = (1 + 0.8 *np.sin(2* np.pi * 500 * ta)) * np.sin(2* np.pi * 100e3 * ta) + (1 + 0.8 *np.sin(2* np.pi * 1500 * ta)) * np.sin(2* np.pi * 120e3 * ta)
# print("fake data done")
# spec_pfb = pfb.pfb(ts, 120, ntap=4)
# print("pfb done")
# print(np.shape(spec_pfb))
# rts = pfb.inverse_pfb(spec_pfb, 4).ravel()
# print(np.shape(rts))
# print("and back")
# ###fake data generation done

# audio = demodulate_chunk(rts, fs, 100e3, sampleRate)



# ##write to file
# write('test.wav',sampleRate,audio)




















# #print(filtered)
# #pylab.plot(ta, rts)
# pylab.plot(audio)
# #pylab.plot(ta[:100000], filtered[:100000])
# #pylab.plot(ta, ts)
# #pylab.show()


# fs = 10000 #sample rate in Hz
# endtime = 10000/fs
# ta = np.linspace(0.0, endtime, int(endtime * fs), endpoint=False)

# ts = np.sin(2*np.pi * ta * 122.0) + np.sin(2*np.pi * ta * 378.1 + 1.0)

# spec_pfb = pfb.pfb(ts, 50, ntap=4)

# print(np.shape(spec_pfb))

# pylab.plot(np.abs(spec_pfb[10]))
# #pylab.show()


# signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
# fourier = np.fft.fft(signal)
# n = signal.size
# timestep = 0.1
# freq = np.fft.rfftfreq(n, d=timestep)
# print(freq)
