import pylab

import numpy as np
import scipy.linalg as la

import pfb_helper as pfb
import SNAPfiletools as sft

from scipy import signal
from scipy.io.wavfile import write
from multiprocessing import Pool, set_start_method, get_context


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

def resample(data, fs, sample_rate):
    q = fs / sample_rate
    if int(np.log10(q)) - np.log10(q) > 0.1:
        print("problems ahead")
    n_times = int(np.log10(q))
    #print(n_times)
    for i in range(n_times):
        data = signal.decimate(data, 10)
    return data


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
    band = 30e3
    radio_signal = np.clip(butter_bandpass_filter(data, freq-band, freq +band, fs, order=3),-1,1)
    #find endtime using sample rate
    endtime = np.shape(data)[0]/fs
    print("bandpass")

    # ta = np.linspace(0.0, len(time_stream['pol0'])/fs, len(time_stream['pol0']), endpoint=False)
    # ts = 0.18 * np.sin(2* np.pi * freq * ta)
    # data = data + ts

    print("injecting sound")



    ##demodulation code!
    fc =5000  # Cut-off frequency of the filter in hz
    filtered =butter_lowpass_filter(np.abs(radio_signal), fc, fs/2) ##low pass the abs of the signal
    print("lowpass")
    ##downsampling code
    sampleRate = audio_rate ##for audio here
    downsample = resample(filtered, fs, sampleRate)

    audio = np.clip(downsample, -1, 1)
    print("downslample")
    audio = audio - np.mean(audio)
    #next two lines is to get it windows readable
    audio = audio/ (max(np.max(audio),np.abs(np.min(audio)))) * 32767 
    audio = np.int16(audio)
    print("data stuff")
    return audio

def do_inverse(info):
    data = info[0]
    n_channels = info[1]  
    s_channel = info[2]  
    spec_pfb = np.zeros((np.shape(data)[0], n_channels),dtype=np.complex)
    spec_pfb[:,int(s_channel) -1:] = data
    rts = np.clip(pfb.inverse_pfb(spec_pfb, 4).ravel(), -1,1)
    return rts

import albatrostools

#start by unpacking data with c code (will need to compile and set LD_LIBRARY_PATH)
#export LD_LIBRARY_PATH={LD_LIBRARY_PATH}:/mnt/c/Users/simta/Documents/GitHub/mars_2019_tools
#gcc -O3 -o libalbatrostools.so -fPIC --shared albatrostools.c -fopenmp
if __name__ == "__main__":
    #fname = "/data/cynthia/albatros/mars2019/baseband/15632/1563273072.raw"
    #fname = "data/15632/1563273072.raw"
    items_per_core=10000
    total_items = 100000
    n_cores = 4

    ctime_start = 1563273072
    ctime_stop = ctime_start + (60 * 4)
    data_dir = "/data/cynthia/albatros/mars2019/baseband"
    #fnames =  sft.time2fnames(ctime_start, ctime_stop, data_dir)

    fnames = ["data/15632/1563273072.raw"]

    if len(fnames) == 0:
        print('No files found in time range')
        exit(0)
    else:
        print("we have", len(fnames), 'files')
    audio = np.array([])

    
    for fname in fnames:
        print("working on" , fname)
        header, data = albatrostools.get_data(fname, items=-1, unpack_fast=False, float=True)
        print("unpacked")
        # pool = Pool(processes=n_cores)
        
        n_channels = header['channels'][-1]
        s_channel = header['channels'][0]
        #freq = np.linspace(0,(n_channels / 2048) * 128e6, n_channels)
        fs = 2* n_channels * 125e6 / 2048
        time_stream = {
            'pol0' : [],
            'pol1' : []
        }

        

        with get_context("spawn").Pool(processes=n_cores) as pool:
            time_stream['pol0'] = np.array(pool.map(do_inverse, [(data['pol0'][x:x+(5*items_per_core),:], n_channels, s_channel) for x in range(0,np.shape(data['pol0'])[0],5*items_per_core) if x+(5*items_per_core) < np.shape(data['pol0'])[0]])).ravel()

        print("inverse done")



        sampleRate = fs/(1e4)
        audio = np.append(audio,demodulate_chunk(time_stream['pol0'],fs, 13.35e6, sampleRate ))
    
    ##write to file
    write('music.wav',int(sampleRate),np.int16(audio))
    print("demodulated")
    





