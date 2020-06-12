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
    band = 5e3
    radio_signal = butter_bandpass_filter(rts, freq-band, freq +band, fs, order=5)

    #find endtime using sample rate
    endtime = np.shape(data)[0]/fs


    ##demodulation code!
    fc =1000  # Cut-off frequency of the filter in hz
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    filtered = signal.filtfilt(b,a,np.abs(radio_signal)) ##low pass the abs of the signal

    ##downsampling code
    sampleRate = audio_rate ##for audio here
    downsample = signal.resample(filtered, int(endtime*sampleRate))
    mx = np.median(downsample) + 2*np.std(downsample)
    mn = np.median(downsample) - 2*np.std(downsample)
    audio = np.clip(downsample, mn, mx)
    audio = audio - np.mean(audio)
    #next two lines is to get it windows readable
    audio = audio/ (max(np.max(audio),np.abs(np.min(audio)))) * 32767 
    audio = np.int16(audio)
    return audio




#import albatrostools

##start by unpacking data with c code (will need to compile and set LD_LIBRARY_PATH)
##export LD_LIBRARY_PATH={LD_LIBRARY_PATH}:/mnt/c/Users/simta/Documents/GitHub/mars_2019_tools
##gcc -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp

# fname = "data/1563719256.raw"
# header, data = albatrostools.get_data(fname, items=1000, unpack_fast=True, float=True)

# rts = pfb.inverse_pfb(data['pol0'], 4)
# print(header['bit_mode'])

endtime = 2 #end time in seconds
fs = 2e6 #sample rate in Hz
ta = np.linspace(0.0, endtime, int(endtime * fs), endpoint=False)
sampleRate = 10000

##### fake data generation 
ts = (1 + 0.8 *np.sin(2* np.pi * 500 * ta)) * np.sin(2* np.pi * 100e3 * ta) + (1 + 0.8 *np.sin(2* np.pi * 1500 * ta)) * np.sin(2* np.pi * 120e3 * ta)
print("fake data done")
spec_pfb = pfb.pfb(ts, 17, ntap=4)
print("pfb done")
rts = pfb.inverse_pfb(spec_pfb, 4).ravel()
print("and back")
###fake data generation done

audio = demodulate_chunk(rts, fs, 120e3, sampleRate)



##write to file
write('test.wav',sampleRate,audio)

#print(filtered)
#pylab.plot(ta, rts)
pylab.plot(audio)
#pylab.plot(ta[:100000], filtered[:100000])
#pylab.plot(ta, ts)
pylab.show()


