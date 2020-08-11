import numpy as np
nm = np

import pickle

import matplotlib.pyplot as plt

import matplotlib

from scipy.optimize import curve_fit

def find_index(array, val):
    '''
        returns the index of the closest matching element in an array
    '''
    for i in range(len(array)):
        if array[i] >= val:
            return i
        else:
            pass
    print("went to end of array")
    return len(array)-1



def collapse_nan(array):
    result = []
    for row in array:
        if ~np.isnan(row[50]):
            result.append(row)
        else:
            pass
    if len(result) ==0:
        print("bad frequency range selection")
    return result

def collapse_nan_loc(array): ###returns array of locations 
    loc = []
    for index,row in  enumerate(array):  
        if ~np.isnan(row[50]):
            loc.append(index)
        else:
            pass
    if len(loc) ==0:
        print("bad freq range")
    return loc

def trim_freq(array, fmin_index, fmax_index):
    result = []
    for row in array:
        result.append(row[fmin_index:fmax_index])
    return result

def lineMult(x,a):
    return x*a

save_loc = "data/temp_plot.pick"
print("reading from: " + save_loc)
with open(save_loc, 'rb') as dic_save:
    results = pickle.load(dic_save)
freqs = np.linspace(0,125,len(results['auto']['pol00'][0]), endpoint= False)


##real values :)
freq_ranges = [[5.31,12.63],[12.7,20.02]]
##pol00 low freq
fmin_index = find_index(freqs, freq_ranges[0][0])
fmax_index = find_index(freqs, freq_ranges[0][1])
base_low_pol00 = trim_freq(results['base']['pol00'],fmin_index, fmax_index)
loc_low_pol00 = collapse_nan_loc(base_low_pol00)
base_low_pol00 = np.abs(np.array(collapse_nan(base_low_pol00))[0:-8,:])
auto_low_pol00 = trim_freq(results['auto']['pol00'],fmin_index, fmax_index)
auto_low_pol00 = np.abs(np.array(collapse_nan(auto_low_pol00))[0:-8,:])/218193584
w_low_pol00 = np.where(np.mean(base_low_pol00, axis=0)> np.mean(base_low_pol00) + np.std(base_low_pol00),np.zeros(np.shape(base_low_pol00)[1]), np.ones(np.shape(base_low_pol00)[1]))
p_low_pol00, _ = curve_fit(lineMult, np.mean(base_low_pol00, axis=0), np.mean(auto_low_pol00, axis =0),sigma= w_low_pol00)
auto_low_pol00 = auto_low_pol00*p_low_pol00[0]
div_low_pol00 = max(np.max(auto_low_pol00),np.max(base_low_pol00))
auto_low_pol00 /= div_low_pol00
base_low_pol00 /= div_low_pol00
auto_low_pol00_mean = np.mean(auto_low_pol00, axis =0)
base_low_pol00_res = np.abs(np.mean(np.array(results['high']['pol00'])[loc_low_pol00,:],axis = 0))/div_low_pol00
##pol00 high freq
fmin_index = find_index(freqs, freq_ranges[1][0])
fmax_index = find_index(freqs, freq_ranges[1][1])-1
base_high_pol00 = trim_freq(results['base']['pol00'],fmin_index, fmax_index)
loc_high_pol00 = collapse_nan_loc(base_high_pol00)
base_high_pol00 = np.abs(np.array(collapse_nan(base_high_pol00)))
auto_high_pol00 = trim_freq(results['auto']['pol00'],fmin_index, fmax_index)
auto_high_pol00 = np.abs(np.array(collapse_nan(auto_high_pol00)))/218193584
# plt.plot(np.mean(base_high_pol00, axis=0))
# plt.plot(np.mean(auto_high_pol00, axis =0))
# plt.show()
# print(np.mean(base_high_pol00, axis=0))
# print(np.mean(auto_high_pol00, axis=0))
w_high_pol00 = np.where(np.mean(base_high_pol00, axis=0)> np.mean(base_high_pol00) + np.std(base_high_pol00),np.zeros(np.shape(base_high_pol00)[1]), np.ones(np.shape(base_high_pol00)[1]))
p_high_pol00, _ = curve_fit(lineMult, np.mean(base_high_pol00, axis=0), np.mean(auto_high_pol00, axis =0), sigma=w_high_pol00)
auto_high_pol00 = auto_high_pol00*p_high_pol00[0]
div_high_pol00 = max(np.max(auto_high_pol00),np.max(base_high_pol00))
auto_high_pol00 /= div_high_pol00
base_high_pol00 /= div_high_pol00
auto_high_pol00_mean = np.mean(auto_high_pol00, axis =0)
base_high_pol00_res = np.abs(np.mean(np.array(results['high']['pol00'])[loc_high_pol00,:],axis = 0))/div_high_pol00





##pol11 low freq
fmin_index = find_index(freqs, freq_ranges[0][0])
fmax_index = find_index(freqs, freq_ranges[0][1])
base_low_pol11 = trim_freq(results['base']['pol11'],fmin_index, fmax_index)
loc_low_pol11 = collapse_nan_loc(base_low_pol11)
base_low_pol11 = np.array(collapse_nan(base_low_pol11))
auto_low_pol11 = trim_freq(results['auto']['pol11'],fmin_index, fmax_index)
auto_low_pol11 = np.array(collapse_nan(auto_low_pol11))
auto_low_pol11_mean = np.mean(auto_low_pol11, axis =0)
base_low_pol11_res = np.mean(np.array(results['high']['pol11'])[loc_low_pol11,:],axis = 0)
##pol11 high freq
fmin_index = find_index(freqs, freq_ranges[1][0])
fmax_index = find_index(freqs, freq_ranges[1][1])
base_high_pol11 = trim_freq(results['base']['pol11'],fmin_index, fmax_index)
loc_high_pol11 = collapse_nan_loc(base_high_pol11)
base_high_pol11 = np.array(collapse_nan(base_high_pol11))
auto_high_pol11 = trim_freq(results['auto']['pol11'],fmin_index, fmax_index)
auto_high_pol11 = np.array(collapse_nan(auto_high_pol11))
auto_high_pol11_mean = np.mean(auto_high_pol11, axis =0)
base_high_pol11_res = np.mean(np.array(results['high']['pol11'])[loc_high_pol11,:],axis = 0)

# Dummy values
dummy_array = nm.random.randn(100,100)
dummy_spec1 = nm.random.randn(1000)
dummy_spec2 = nm.random.randn(100)
x1 = nm.linspace(0, 1, 1000)
x2 = nm.linspace(0, 1, 100)

# Set some global font sizes
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

# Constrained layout for the win
# https://matplotlib.org/3.1.1/tutorials/intermediate/constrainedlayout_guide.html
fig, axs = plt.subplots(3, 2, figsize=(16, 10), constrained_layout=True)
ax = axs.flat[0]
im = ax.imshow(auto_low_pol00, vmin=0, vmax =1, cmap='YlOrRd', aspect='auto')
# For 5.3-12.6 MHz plot, trim off last ~5 minutes
ax.set_title('Direct', loc='left', horizontalalignment='right', verticalalignment='bottom')
ax.yaxis.set_label_coords(-0.1,1.04)

ax = axs.flat[1]
im = ax.imshow(auto_high_pol00, vmin=0, vmax =1,cmap='YlOrRd', aspect='auto')

ax = axs.flat[2]
im = ax.imshow(base_low_pol00,vmin=0, vmax =1, cmap='YlOrRd', aspect='auto')
# For 5.3-12.6 MHz plot, trim off last ~5 minutes
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Time (Minutes)')
ax.set_title('Baseband', loc='left', horizontalalignment='right', verticalalignment='bottom')
fig.colorbar(im, ax=axs[:2,:], shrink=0.8)

ax = axs.flat[3]
im = ax.imshow(base_high_pol00,vmin=0, vmax =1, cmap='YlOrRd', aspect='auto')



ax = axs.flat[4]
ax.plot(np.linspace(freqs[fmin_index], freqs[fmax_index], len(base_low_pol00_res)),base_low_pol00_res, 'k-', label='Baseband')
ax.plot(np.linspace(freqs[fmin_index], freqs[fmax_index], len(auto_low_pol00_mean)),auto_low_pol00_mean, color='#f03523', linestyle='-', linewidth=3, label='Direct')
# ax.set_xlim([5.3, 12.6])
# ax.set_ylim([0, 0.8])
ax.legend(loc='upper left')
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Magnitude')

ax = axs.flat[5]
ax.plot(np.linspace(freqs[fmin_index], freqs[fmax_index], len(base_high_pol00_res)),base_high_pol00_res, 'k-')
ax.plot(np.linspace(freqs[fmin_index], freqs[fmax_index], len(auto_high_pol00_mean)),auto_high_pol00_mean, color='#f03523', linestyle='-', linewidth=3)
# ax.set_xlim([12.6, 20.0])
# ax.set_ylim([0, 0.8])

plt.savefig('spiffy.png')
