from scipy.io.wavfile import read, write


rate, audio = read('music.wav')
write('trying.wav',int(rate/2),audio)