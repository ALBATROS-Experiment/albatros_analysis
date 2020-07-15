running_local = False

import numpy as np

import copy
import pickle

import sys, os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt

import scio
import SNAPfiletools as sft
import albatrostools as alb
from scipy import signal

import cmath

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


def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def get_comparison(start_time, stop_time, baseband_dir, auto_dir):
    '''
        Gets the same data representation coming from auto and baseband
        returns dictionary of dictionaries of arrays:
        top dictrionary has "dif", "auto", "base" the next layer has
        "pol00", "pol11", "pol01" each is an array of numpy arrays that represnet
        the squashed down set of computed correlations for a time chunk of baseband (about a minute)
        and the numpy array contains values of different frequencies as given by np.linspace(0,125,auto_channels)
    '''
    ctime_start = sft.timestamp2ctime(start_time)
    ctime_stop = sft.timestamp2ctime(stop_time)

    auto_files = sft.time2fnames(ctime_start, ctime_stop, auto_dir)
    ##get ctimes as integers 
    auto_times = np.array([int(os.path.basename(path)) for path in auto_files])
    baseband_files = sft.time2fnames(ctime_start, ctime_stop, baseband_dir)
    if len(baseband_files) == 0: #switch from MARS1 to MARS2
        baseband_dir = baseband_dir.replace("MARS1", "MARS2")
        baseband_files = sft.time2fnames(ctime_start, ctime_stop, baseband_dir)
        if len(baseband_files) == 0:
            print("no baseband found. Sad life")
            exit(0)
    ##also gets ctimes as integers for the baseband files 
    # (im implicitly assuming later on that these files span a shorter time)
    baseband_times = np.array([int(os.path.basename(path).split('.')[0]) for path in baseband_files])

    ##initalize the result arrays 
    #dict on the 3 correlations
    res_temp = {
        'pol00': [],
        'pol11': [],
        'pol01': []
    }

    ##populate a results dictoinary for all the quantities of interest
    results = {
        'dif' : copy.deepcopy(res_temp),
        'auto': copy.deepcopy(res_temp),
        'base': copy.deepcopy(res_temp)
    }

    print("auto time length:", len(auto_files))
    print("baseband length:" , len(baseband_files))

    ##loop through the auto computed spectra and extract the data
    for index, auto_file in enumerate(auto_files):
        print("reading from", str(auto_file))
        pol00 = scio.read(os.path.join(auto_file,'pol00.scio'))
        pol11 = scio.read(os.path.join(auto_file,'pol11.scio'))
        pol01r = scio.read(os.path.join(auto_file,'pol01r.scio'))
        pol01i = scio.read(os.path.join(auto_file,'pol01i.scio'))
        pol01 =  pol01r + 1J*pol01i
        
        ##get the start and end time and generate the time scale to match up with baseband later
        start_time_auto = auto_times[index]
        try:
            end_time_auto = auto_times[index + 1]
        except IndexError:
            end_time_auto = start_time_auto + 60*60 ##add an hour
        time_delta_auto = (end_time_auto  - start_time_auto)
        time_scale_auto = np.linspace(start_time_auto,  end_time_auto, np.shape(pol00)[0])

        #number of channels and freq range
        auto_channels = np.shape(pol00)[1]
        print("auto channels number should be 2048", auto_channels)
        freq_auto = np.linspace(0,125,auto_channels, endpoint=True)

        ##get the baseband files in the range of the extracted autos
        ##this is why we implcitly assumed that the autos have more time per file than the baseband
        baseband_in_range = []
        for ind, b_time in enumerate(baseband_times):
            if b_time >= start_time_auto and b_time <= end_time_auto:
                baseband_in_range.append(baseband_files[ind])
        
        if len(baseband_in_range) == 0:
            print("no baseband in range:", start_time_auto, end_time_auto)
            continue
        
        ##loop through relevant files of baseband
        for base_n, base_file in enumerate(baseband_in_range):
            print("reading", str(base_file))
            header, data = alb.get_data(base_file, items = 1000,unpack_fast=True, float= True) #change that back to -1 eventually
            fmin = header['channels'][0]*125.0/2048
            fmax = header['channels'][-1]*125.0/2048
            n_chan = header['channels'][-1] - header['channels'][0]

            ##used to check if frequency bins are the same from auto to baseband
            freq_base = np.linspace(fmin,fmax,n_chan, endpoint= True)
            start_time_base = int(os.path.basename(base_file).split(".")[0])
            print(start_time_base)
            try:
                some_name = np.where(baseband_times == start_time_base)[0][0]
                end_time_base = baseband_times[some_name +1]
            except IndexError:
                end_time_base = start_time_base + 60*2 ##add a miniute or so
            print(end_time_base)
            time_delta_base = end_time_base -start_time_base

            # Calculate auto and cross correlations
            corr = data['pol0']*np.conj(data['pol1'])
            ##calculate compress the data into one row
            mean_pol00 = np.mean(np.abs(data['pol0'])**2, axis=0)
            mean_pol11 = np.mean(np.abs(data['pol1'])**2, axis=0)
            mean_pol01 = np.mean(corr, axis = 0)

            #get the index in auto spectra to which this time chunk of baseband corresponds
            start_index_auto = find_index(time_scale_auto, start_time_base)
            end_index_auto = find_index(time_scale_auto, end_time_base)

            ##lets check if the frequencies line up (if not we are in for a world of pain) we kinda are but i decided to ignore it :)
            fmin_index_auto = find_index(freq_auto, fmin) - 1 ##wtffffff think about this tmr 
            fmax_index_auto = find_index(freq_auto, fmax) 

            ##trying someting 
            fmin_index_auto = int(header['channels'][0])
            fmax_index_auto = int(header['channels'][-1]+1)

            ##soooo the frequencies dont seem to align ill look at it again in the morning :)

            # if set(freq_base).issubset(set(freq_auto)):
            #     pass
            # else:
            #     print("fuck my life will need serious rebinnig")
            #     print("freq_base:")
            #     print(freq_base)
            #     print("freq_auto:")
            #     print(freq_auto)
            #     ##exit(1)
            

            #lets check that there is the correct number of bins
            if n_chan != (fmax_index_auto - fmin_index_auto) -1:
                print("will need to re-bin not the end of the world")
                print("n_chan: " + str(n_chan))
                print("auto_chan: " + str(fmax_index_auto - fmin_index_auto))
                #exit(1)
            
            ##Realsitically if one of these checks fail both will.... 
            ##to fix should sue scipy.signal.resample() to match it up


            # fmin_index_auto = np.where(freq_auto == fmin)[0][0]
            # fmax_index_auto = np.where(freq_auto == fmax)[0][0]

            print("comparing min freqeuncies: " + str(freq_auto[fmin_index_auto]) + " with " + str(freq_base[0]))
            print("comparing max freqeuncies: " + str(freq_auto[fmax_index_auto]) + " with " + str(freq_base[-1])) 

            ##get corresponding auto data to the extracted baseband
            print("sqish")
            auto_pol00 = np.mean(pol00[start_index_auto:end_index_auto, fmin_index_auto:fmax_index_auto], axis=0)
            auto_pol11 = np.mean(pol11[start_index_auto:end_index_auto, fmin_index_auto:fmax_index_auto], axis=0)
            auto_pol01 = np.mean(pol01[start_index_auto:end_index_auto, fmin_index_auto:fmax_index_auto], axis=0)
            print("squish")
            print("indecess", start_index_auto, end_index_auto)
            if np.isnan(auto_pol00).all():
                print("skipping data cause simon doesnt know how to code or msthing")
                continue 

                
                #print(pol00[start_index_auto:end_index_auto, fmin_index_auto:fmax_index_auto])
        
            ##keep it non numpy when appending since np appends in a very shitty way
            ##start with difference code
            ##create a array of nans
            nan_holder = np.empty(auto_channels, dtype= complex)
            nan_holder[:] = np.nan

            temp = nan_holder.copy()  
            ##little pause to mention how the fact that python hides pointers is stupid
            ##chased my tail there
            temp[fmin_index_auto:fmax_index_auto] = auto_pol00 - mean_pol00
            results['dif']['pol00'].append(temp)
            temp = nan_holder.copy() 
            temp[fmin_index_auto:fmax_index_auto] = auto_pol11 - mean_pol11
            results['dif']['pol11'].append(temp)
            temp = nan_holder.copy() 
            temp[fmin_index_auto:fmax_index_auto] = auto_pol01 - mean_pol01
            results['dif']['pol01'].append(temp)

            ##now lets do autos
            temp = nan_holder.copy()  
            temp[fmin_index_auto:fmax_index_auto] = auto_pol00
            results['auto']['pol00'].append(temp)
            temp = nan_holder.copy()  
            temp[fmin_index_auto:fmax_index_auto] = auto_pol11
            results['auto']['pol11'].append(temp)
            temp = nan_holder.copy()  
            temp[fmin_index_auto:fmax_index_auto] = auto_pol01
            results['auto']['pol01'].append(temp)

            ##and now bsaeband
            temp = nan_holder.copy()  
            temp[fmin_index_auto:fmax_index_auto] = mean_pol00
            results['base']['pol00'].append(temp)
            temp = nan_holder.copy()  
            temp[fmin_index_auto:fmax_index_auto] = mean_pol11
            results['base']['pol11'].append(temp)
            temp = nan_holder.copy()  
            temp[fmin_index_auto:fmax_index_auto] = mean_pol01
            results['base']['pol01'].append(temp)
            
            print("appended some data we now have: " + str(len(results['base']['pol01'])) + " lines of goodness")

    return results, np.linspace(0,125,len(results['auto']['pol00'][0]), endpoint= True)

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

def trim_freq(array, fmin_index, fmax_index):
    result = []
    for row in array:
        result.append(row[fmin_index:fmax_index])
    return result

## at this point the 3 arrays of numpy arrays have all the differences 
## we can either im show them or collaps them 
##if we imshow them its gonna be blocks
##we can collaps the nans out and see if it looks prettier
##this is a problem for the morning
##wanna run this first to double check it works 

## should i be showing plots of the flattened arrays 
## or should i make the heatmap

##lets start with cynthia plot implementation
##that is two collumns 3 rows. Left column is 5- 13mhz right is 13-20 (why not one big column?)
##top row is direct auto 
##middle row is baseband auto
##bottom row is mean of both

out_dir = "/project/s/sievers/simont/mars_2019_tools/plots"
start_time = "20190721_052255"
stop_time = "20190721_150000"
baseband_dir = "/project/s/sievers/mars2019/MARS1/albatros_north_baseband"
auto_dir = "/project/s/sievers/mars2019/auto_cross/data_auto_cross"
save_loc = "/project/s/sievers/simont/mars_2019_tools/plots/temp_plot.pick"
compute = True
if running_local is True:
    save_loc = "data/temp_plot.pick"
    out_dir = "plots"
    compute = False

if compute is True:
    results, freqs = get_comparison(start_time, stop_time, baseband_dir, auto_dir)
    print("saving to: " + save_loc)
    with open(save_loc, 'w') as dic_save:
        pickle.dump(results, dic_save)
else:
    print("reading from: " + save_loc)
    with open(save_loc, 'r') as dic_save:
        results = pickle.load(dic_save)
    freqs = np.linspace(0,125,len(results['auto']['pol00'][0]), endpoint= True)

for val in results["auto"]['pol00'][1]:
    pass
    #print(val)

# fmin_index = find_index(freqs, 5.31)
# print(fmin_index)
# fmax_index = find_index(freqs, 12.63) -1
# print(fmax_index)
# ##now lets squash the pesky nans
# data = trim_freq(results["base"]["pol00"], fmin_index, fmax_index)
# print("data shape is", str(len(data)), str(len(data[0])))
# data = collapse_nan(data)
# print("data shape is", str(len(data)), str(len(data[0])))
# data = np.array(data)

for  pol in results["auto"]: ##should be pol00, pol01, pol11
    if  pol == "pol01":
        ##we have complex stuff lets skip it 
        continue
    ##initate the figs
    plt.figure(figsize=(16,16))

    freq_ranges = [[5.31,12.63],[12.7,20.02]]
    cell = 1
    for row in results:##should be dif, auto ,base
        if row == "dif":
            ##we arent plotting that as of now 
            continue
        for freq_range in freq_ranges: ##the two collumns

            print("plotting" + pol + row + str(cell))

            ##iterate through the subplots 
            plt.subplot(3,2,cell)
            cell += 1

            plt.title(str(pol) + str(row))
            ##start by getting the bit of results that matters (or at least the interesting indecees)
            fmin_index = find_index(freqs, freq_range[0])
            print(fmin_index)
            fmax_index = find_index(freqs, freq_range[1]) -1
            print(fmax_index)
            ##now lets squash the pesky nans
            data = trim_freq(results[row][pol], fmin_index, fmax_index)
            print("data shape is", str(len(data)), str(len(data[0])))
            data = collapse_nan(data)
            data = np.abs(np.array(data))
            myExtents = np.array([freqs[fmin_index], freqs[fmax_index], np.shape(data)[0],0])##should be in minutes give or take
            vmin = np.median(data) - 2*np.std(data)
            vmax = np.median(data) + 2*np.std(data)
            print(np.shape(data))
            plt.imshow(data, vmin=vmin, vmax=vmax, aspect='auto', extent=myExtents)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('(Down) minutes')

    ##now we are only left with the last two 
    for freq_range in freq_ranges:
        plt.subplot(3,2,cell)
        cell += 1
        plt.title("averages")

        ##start by getting the bit of results that matters (or at least the interesting indecees)
        fmin_index = find_index(freqs, freq_range[0])
        fmax_index = find_index(freqs, freq_range[1]) -1

        ##now lets squash the pesky nans
        data = trim_freq(results["auto"][pol], fmin_index, fmax_index)
        data = collapse_nan(data)
        auto_data = np.mean(np.array(data),axis = 0)
        mx = np.max(np.abs(auto_data))
        auto_data = auto_data/mx
        data = trim_freq(results["base"][pol], fmin_index, fmax_index)
        data = collapse_nan(data)
        base_data = np.mean(np.array(data),axis = 0)
        mx = np.max(np.abs(base_data))
        base_data = base_data/mx

        plt.plot(freqs[fmin_index:fmax_index],auto_data, label="auto", color="blue")
        plt.plot(freqs[fmin_index:fmax_index],base_data, label="baseband", color="red")
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude')
        plt.legend()
    if cell != 7:
        print("simon should take a comp class lol")

    plt.suptitle("comparison " + str(pol))
    plt.show()
    plt.savefig(os.path.join(out_dir, str(pol)+".png"))