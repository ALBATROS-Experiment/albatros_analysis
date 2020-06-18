import pylab

import numpy as np
import scipy.linalg as la

import pfb_helper as pfb


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
    band = 20e3
    radio_signal = np.clip(butter_bandpass_filter(data, freq-band, freq +band, fs, order=3),-1,1)
    #find endtime using sample rate
    endtime = np.shape(data)[0]/fs
    print("bandpass")

    # ta = np.linspace(0.0, len(time_stream['pol0'])/fs, len(time_stream['pol0']), endpoint=False)
    # ts = 0.18 * np.sin(2* np.pi * freq * ta)
    # data = data + ts

    print("injecting sound")



    ##demodulation code!
    fc =15000  # Cut-off frequency of the filter in hz
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
    fname = "/data/cynthia/albatros/mars2019/baseband/15632/1563273072.raw"
    fname = "data/1563719256.raw"
    items_per_core=10000
    total_items = 50000
    n_cores = 4

    header, data = albatrostools.get_data(fname, items=total_items, unpack_fast=False, float=True)
    print("unpacked")
    # pool = Pool(processes=n_cores)
    
    n_channels = header['channels'][-1]
    print(n_channels)
    s_channel = header['channels'][0]
    #freq = np.linspace(0,(n_channels / 2048) * 128e6, n_channels)
    fs = 2* n_channels * 125e6 / 2048
    print(fs)
    time_stream = {
        'pol0' : [],
        'pol1' : []
    }

    

    with get_context("spawn").Pool(processes=n_cores) as pool:
        time_stream['pol0'] = np.array(pool.map(do_inverse, [(data['pol0'][x:x+(5*items_per_core),:], n_channels, s_channel) for x in range(0,np.shape(data['pol0'])[0],5*items_per_core) if x+(5*items_per_core) < np.shape(data['pol0'])[0]])).ravel()

    print("inverse done")

    # f, t, Sxx = signal.spectrogram(time_stream['pol0'], fs)
    # pylab.pcolormesh(t, f, Sxx)
    # pylab.ylabel('Frequency [Hz]')
    # pylab.xlabel('Time [sec]')
    # pylab.show()
    print(len(time_stream['pol0']))
    sampleRate = fs/(1e4)
    audio = demodulate_chunk(time_stream['pol0'],fs, 5.7e6, sampleRate )
    print(len(audio))
    ##write to file
    write('test.wav',int(sampleRate),audio)
    print("demodulated")
    pylab.plot(audio)
    #pylab.show()












    # spec_pfb = np.zeros((np.shape(data['pol0'])[0], n_channels),dtype=np.complex)
    # spec_pfb[:,int(header['channels'][0]) -1:] = data['pol0']
    # rts = np.clip(pfb.inveprse_pfb(spec_pfb, 4).ravel(), -1,1)
    # time_stream['pol0'] = rts




    #print(np.shape(downsample))
    #downsample = signal.resample(filtered, int(endtime*sampleRate))
    #print(np.shape(downsample))
    #mx = np.median(downsample) + 2*np.std(downsample)
    #mn = np.median(downsample) - 2*np.std(downsample)


   # f, t, Sxx = signal.spectrogram(time_stream['pol0'], fs)
    # pylab.pcolormesh(t, f, Sxx)
    # pylab.ylabel('Frequency [Hz]')
    # pylab.xlabel('Time [sec]')
    # pylab.show()
    # ta = np.linspace(0.0, len(time_stream['pol0'])/fs, len(time_stream['pol0']), endpoint=False)
    # ts = 0.1 * (1 + 0.8 *np.sin(2* np.pi * 500 * ta)) * np.sin(2* np.pi * 10e6 * ta)
    # time_stream['pol0'] = time_stream['pol0'] + ts
    # spectro = pfb.pfb(time_stream['pol0'], 500, ntap=4)
    # pylab.imshow(np.abs(spectro), aspect='auto', interpolation='nearest', extent=[0, (n_channels / 2048) * 125e6, 0,1])
    # pylab.show()
    # pylab.imshow(np.abs(data['pol0']), aspect='auto', interpolation='nearest', extent=[12.63e6, (n_channels / 2048) * 125e6, 0,1])
    # pylab.show()





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
#pylab.imshow(np.abs(spec_pfb), aspect='auto', interpolation='nearest', extent=[0, (n_channels / 2048) * 125e6, 0,1])
#pylab.imshow(np.abs(data['pol0']), aspect='auto', interpolation='nearest')
#pylab.show()


#rsync -auv --progress simont@niagara.scinet.utoronto.ca:/project/s/sievers/mars2019/MARS1/albatros_north_baseband/ /data/cynthia/albatros/mars2019/baseband































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
