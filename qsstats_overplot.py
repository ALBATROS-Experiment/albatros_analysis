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

import argparse

def getstats(file, stat_type):
    dataset = np.load(filepath + '/' + file)
    freq = dataset['freq']
    
    pol00_stat = dataset['pol00_' + stat_type]    
    pol11_stat = dataset['pol11_' + stat_type]    
    return freq, pol00_stat, pol11_stat, dataset['tstart'], dataset['tstop'], dataset['timestamp'], dataset['desc']

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="location of .npz files written by quick_spectra")
parser.add_argument("-s", "--stat", dest='stat', default = 'med', type=str, help="stat type: min, max, mean, med=default")
parser.add_argument("-d", "--desc", dest='top_desc', default = None, type=str, help="optional description")
args = parser.parse_args()

filepath = args.data_dir
filenames = [f for f in listdir(filepath) if isfile(join(filepath,f))]

# pick the .npz files in the specified filepath
files = []
for file in filenames:
    if file.endswith(".npz"):
        files.append(file)

#plt.figure()
fig, axs = plt.subplots(2)
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2, sharex=ax1)

for f in files:
    freq, pol00_stat, pol11_stat, tstart, tstop, timestamp, desc = getstats(f, args.stat)
    if desc == '':
        label = "{0}_{1}-{2}".format(timestamp, tstart, tstop)
    else:
        label = desc
    ax1.plot(freq, pol00_stat, label=label)
    ax2.plot(freq, pol11_stat, label=label)

ax1.set_title('pol00 {0}'.format(args.stat))
ax2.set_title('pol11 {0}'.format(args.stat))

ax1.set(xlabel='frequency (MHz)', ylabel='uncal log(power)')
ax2.set(xlabel='frequency (MHz)', ylabel='uncal log(power)')
if args.top_desc:
    fig.suptitle(args.top_desc, fontsize=16)
ax1.legend()
ax2.legend()
fig.tight_layout()
plt.show()
