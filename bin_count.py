import numpy as np
import SNAPfiletools as sft
import albatrostools as alb

start_time = "20190721_052255"
stop_time = "20190721_150000"
data_dir = "/project/s/sievers/mars2019/MARS1/albatros_north_baseband"
ctime_start = sft.timestamp2ctime(start_time)
ctime_stop = sft.timestamp2ctime(stop_time)

files = sft.time2fnames(ctime_start, ctime_stop, data_dir)

for file in files:
    print("working on:", file)
    header, data = alb.get_data(file, items = -1,unpack_fast=True, float= True,byte_delta= -8)
    if header["bit_mode"] != 4:
        print("mama mia!!! ur bit-ochini is undercooked")
        continue
    print("fmin:",header['channels'][0]*125.0/2048)
    print("fmax", header['channels'][-1]*125.0/2048)
    for i in range(np.shape(data["pol0"])[1]):
        unique, counts = np.unique(np.real(data["pol0"][:,i]), return_counts=True)
        if np.std(counts) > 10:
            print("the counts were strange for pol0 real at frequency", str((i+header['channels'][0])/125.0/2048))
            print(dict(zip(unique,counts)))
        unique, counts = np.unique(np.imag(data["pol0"][:,i]), return_counts=True)
        if np.std(counts) > 10:
            print("the counts were strange for pol0 imaginary at frequency", str((i+header['channels'][0])/125.0/2048))
            print(dict(zip(unique,counts)))