#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:26:50 2023

@author: eamon
"""

import numpy as np
from os.path import isfile, join
from os import listdir
import matplotlib.pyplot as plt

def getstats(file, stat_type):
    dataset = np.load(filepath + '/' + file)
    freq = dataset['freq']
    
    pol00_stat = dataset['pol00_' + stat_type]    
    pol11_stat = dataset['pol11_' + stat_type]    
    return freq, pol00_stat, pol11_stat

filepath = '/home/eamon/harmfinder'
filenames = [f for f in listdir(filepath) if isfile(join(filepath,f))]

# pick the .npz files in the specified filepath
files = []
for file in filenames:
    if file.endswith(".npz"):
        files.append(file)

#plt.figure()
fig, axs = plt.subplots(2)
for f in files:
    freq, pol00_stat, pol11_stat = getstats(f, 'med')
    axs[0].plot(freq, pol00_stat)
    axs[1].plot(freq, pol11_stat)

plt.show()