running_local = True

import numpy as np
n = np
import pfb_helper as pfb
import scipy

import copy
import pickle

import sys, os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    #print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt

import scio
import SNAPfiletools as sft
import albatrostools as alb
from scipy import signal

import cmath

from multiprocessing import Pool, get_context


def re_bin(data, nbin):
    ##first switch to time domain
    ts = pfb.inverse_pfb(data,4).ravel()
    #snip off the start and end which are noizy af
    lent = len(ts)
    ts = ts[int(0.1 * lent): - int(0.1*lent)]
    #lets just hope it isnt prime :)
    #it has an fft deep inside it so this will be slow if its prime
    spec = pfb.pfb(ts, nbin, ntap=4)
    return spec

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

def get_comparison(start_time, stop_time, baseband_dir, auto_dir, freq_ranges):
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
        'base': copy.deepcopy(res_temp),
        'high': copy.deepcopy(res_temp)
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
        auto_start_times = np.fromfile(os.path.join(auto_file, 'time_gps_start.raw'))
        auto_stop_times = np.fromfile(os.path.join(auto_file, 'time_gps_stop.raw'))
        
        ##get the start and end time and generate the time scale to match up with baseband later
        start_time_auto = auto_start_times[0]
        end_time_auto = auto_stop_times[-1]

        time_delta_auto = (end_time_auto  - start_time_auto)
        time_scale_auto = np.linspace(start_time_auto,  end_time_auto, np.shape(pol00)[0])

        #number of channels and freq range
        auto_channels = np.shape(pol00)[1]
        print("auto channels number should be 2048", auto_channels)
        freq_auto = np.linspace(0,125,auto_channels, endpoint=False)

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
            header, data = alb.get_data(base_file, items = -1,unpack_fast=True, float= True,byte_delta= -8) #change that back to -1 eventually
            fmin = header['channels'][0]*125.0/2048
            fmax = header['channels'][-1]*125.0/2048
            littleflag = 0
            for freq_range in freq_ranges:
                if (np.abs(freq_range[0] - fmin) < 0.1 and  np.abs(freq_range[1] - fmax) < 0.1):
                    ##this is the data we want!!
                    littleflag += 1
            #if little flag is still zero the data is trash
            if littleflag == 0:
                print("baseband data not in frequnecy range so skipping")
                continue
            n_chan = header['channels'][-1] - header['channels'][0]

            ##used to check if frequency bins are the same from auto to baseband
            start_time_base = int(os.path.basename(base_file).split(".")[0])
            #print(start_time_base)
            try:
                some_name = np.where(baseband_times == start_time_base)[0][0]
                end_time_base = baseband_times[some_name +1]
            except IndexError:
                end_time_base = start_time_base + 30 ##add a miniute or so
                print("made up end point")
            #print(end_time_base)


            time_delta_base = end_time_base -start_time_base
            if time_delta_base > 360:
                end_time_base = start_time_base + 30
                print("made up end point")

            # Calculate auto and cross correlations
            corr = data['pol0']*np.conj(data['pol1'])
            ##calculate compress the data into one row
            mean_pol00 = np.mean(np.abs(data['pol0'])**2, axis=0)
            mean_pol11 = np.mean(np.abs(data['pol1'])**2, axis=0)
            mean_pol01 = np.mean(corr, axis = 0)

            #get the index in auto spectra to which this time chunk of baseband corresponds using real math this time woop woop
            start_index_auto = find_index(auto_start_times, start_time_base)
            end_index_auto = find_index(auto_stop_times, end_time_base)


            ##trying someting should correspond according to jon
            ##the plus 1 is cause : is non inclusive on right side
            fmin_index_auto = int(header['channels'][0])
            fmax_index_auto = int(header['channels'][-1] + 1)

            

            #lets check that there is the correct number of bins
        
            if n_chan != (fmax_index_auto - fmin_index_auto) -1:
                print("will need to re-bin not the end of the world")
                print("n_chan: " + str(n_chan))
                print("auto_chan: " + str(fmax_index_auto - fmin_index_auto))
                exit(1)
            


            print("comparing min freqeuncies: " + str(freq_auto[fmin_index_auto]) + " with " + str(fmin))
            print("comparing max freqeuncies: " + str(freq_auto[fmax_index_auto - 1]) + " with " + str(fmax))
            #minus 1 cause the right bound is excluded 

            ##get corresponding auto data to the extracted baseband
            auto_pol00 = np.mean(pol00[start_index_auto:end_index_auto, fmin_index_auto:fmax_index_auto], axis=0)
            auto_pol11 = np.mean(pol11[start_index_auto:end_index_auto, fmin_index_auto:fmax_index_auto], axis=0)
            auto_pol01 = np.mean(pol01[start_index_auto:end_index_auto, fmin_index_auto:fmax_index_auto], axis=0)

            print("indecess in time domain: ", start_index_auto, end_index_auto)
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

            ##difference which im not using so why waste memory on it:) and its redundent too.....
            # temp[fmin_index_auto:fmax_index_auto] = auto_pol00 - mean_pol00
            # results['dif']['pol00'].append(temp)
            # temp = nan_holder.copy() 
            # temp[fmin_index_auto:fmax_index_auto] = auto_pol11 - mean_pol11
            # results['dif']['pol11'].append(temp)
            # temp = nan_holder.copy() 
            # temp[fmin_index_auto:fmax_index_auto] = auto_pol01 - mean_pol01
            # results['dif']['pol01'].append(temp)

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
            n_cores = 80
            print("moving on to high res of baseband")
            lines_per_core = int(1000)
            high_res_bins = 500
            with get_context("spawn").Pool() as pool:
                for pol in ['pol0', 'pol1']:
                    print("re-binning:", [pol])
                    job = [(data[pol][x:(x+lines_per_core),:], high_res_bins) for x in range(0, np.shape(data[pol])[0]-1,lines_per_core)]
                    result = pool.starmap(re_bin, job)
                    result_key ='pol' + pol.split('l')[-1] + pol.split('l')[-1]
                    results['high'][result_key].append(np.mean(np.abs(np.vstack(result)**2),axis = 0))
            print("appended some data we now have: " + str(len(results['base']['pol01'])) + " lines of goodness")

    return results, np.linspace(0,125,len(results['auto']['pol00'][0]), endpoint= False)

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
az = []
ap = []
plotz = {}
if __name__ == "__main__":

    out_dir = "/project/s/sievers/simont/mars_2019_tools/plots"
    start_time = "20190721_052255"
    stop_time = "20190721_150000"
    baseband_dir = "/project/s/sievers/mars2019/MARS1/albatros_north_baseband"
    auto_dir = "/project/s/sievers/mars2019/auto_cross/data_auto_cross"
    save_loc = "/project/s/sievers/simont/mars_2019_tools/plots/temp_plot.pick"
    compute = True

    freq_ranges = [[5.31,12.63],[12.7,20.02]]
    

    if running_local is True:
        save_loc = "data/temp_plot.pick"
        out_dir = "plots"
        compute = False

    if compute is True:
        results, freqs = get_comparison(start_time, stop_time, baseband_dir, auto_dir, freq_ranges)
        print("saving to: " + save_loc)
        with open(save_loc, 'wb') as dic_save:
            pickle.dump(results, dic_save)
    else:
        print("reading from: " + save_loc)
        with open(save_loc, 'rb') as dic_save:
            results = pickle.load(dic_save)
        freqs = np.linspace(0,125,len(results['auto']['pol00'][0]), endpoint= False)

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


    freq_shift = 0 ##this better get switched to zero sometime in the near future nivek im looking at u
    # Set some global font sizes
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}
    mpl.rc('font', **font)
    for  pol in results["auto"]: ##should be pol00, pol01, pol11
        #freq_ranges = [[79.96,87.28]]
        if  pol == "pol01":
            print("working on phases")
            ##we have complex stuff lets skip it
            #time to plot the angles
            # so lets go
            plt.figure(figsize=(16,16))
            
            cell = 1
            for freq_range in freq_ranges:
                fmin_index = find_index(freqs, freq_range[0])
                fmax_index = find_index(freqs, freq_range[1])
                auto = trim_freq(results['auto'][pol], fmin_index + freq_shift, fmax_index)
                auto = collapse_nan(auto)
                auto = np.array(auto)
                auto = np.angle(auto)
                auto[auto < 0] += 2* np.pi
                base = trim_freq(results['base'][pol], fmin_index , fmax_index- freq_shift)
                base = collapse_nan(base)
                base = np.array(base)
                base = np.angle(base)
                base[base<0] += 2 * np.pi
                data = auto - base
                plt.subplot(1,len(freq_ranges), cell)
                cell += 1

                myExtent = np.array([freq_range[0], freq_range[1], np.shape(data)[0], 0])
                plt.imshow(data, vmin=-np.pi, vmax= np.pi, extent=myExtent, aspect = 'auto')
                plt.colorbar()
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('Minutes (down)')
                plt.title("Pol01 phase")
            plt.suptitle("phase difference auto-cross")
            plt.savefig(os.path.join(out_dir, str(pol)+".png"))
            cell = 0
            fig, axs = plt.subplots(2, len(freq_ranges), figsize=(16, 10), constrained_layout=True)
            for row in results:
                if row == "dif":
                    continue
                if row == "high":
                    continue
                for freq_range in freq_ranges:
                    fmin_index = find_index(freqs, freq_range[0])
                    fmax_index = find_index(freqs, freq_range[1])
                    
                    if row == 'auto':
                        fmin_index += freq_shift
                    elif row == 'base':
                        fmax_index -= freq_shift
            
                    data = trim_freq(results[row][pol], fmin_index, fmax_index)
                    data = collapse_nan(data)
                    data = np.array(data)
                    data=np.angle(data)
                    #data[ data< 0 ] += 2 *np.pi
                    ax = axs.flat[cell]

                    if cell ==0:
                        ax.set_title('Direct', loc='left', horizontalalignment='right', verticalalignment='bottom')
                        ax.yaxis.set_label_coords(-0.1,1.04)
                    elif cell ==2:
                        ax.set_xlabel('Frequency (MHz)')
                        ax.set_ylabel('Time (Minutes)')
                        ax.set_title('Baseband', loc='left', horizontalalignment='right', verticalalignment='bottom')
                    if cell%2 ==0:
                        data = data[0:-8,:]
                    else:
                        data = data[:,0:-1]

                    myExtent = np.array([freq_range[0], freq_range[1], np.shape(data)[0], 0])
                    im = ax.imshow(data, vmin = -np.pi ,vmax =np.pi, extent=myExtent, aspect = 'auto', cmap='coolwarm')
                    #ax.colorbar()
                    #ax.set_xlabel('Frequency (MHz)')
                    #ax.set_ylabel('Minutes (down)')


                    cell += 1
                    #plt.title(row + " phase, pol01")
            #plt.suptitle("phase difference auto-cross")
            fig.colorbar(im, ax=axs[:2,:], shrink=0.8)
            plt.savefig(os.path.join(out_dir, str(pol)+"indi"+".png"))
            continue

            
        ##initate the figs
        fig, axs = plt.subplots(3, 2, figsize=(16, 10), constrained_layout=True)

        cell = 0
        vmax = 0
        for row in results:##should be dif, auto ,base
            if row == "dif":
                ##we arent plotting that as of now 
                continue
            if row == "high":
                continue
            for freq_range in freq_ranges: ##the two collumns

                print("plotting" , pol , row , str(cell))

                ##iterate through the subplots 
                ax = axs.flat[cell]
                cell += 1

                #plt.title(str(pol) + str(row))
                ##start by getting the bit of results that matters (or at least the interesting indecees)
                fmin_index = find_index(freqs, freq_range[0])
                fmax_index = find_index(freqs, freq_range[1]) -1
                if row == "auto":
                    fmin_index += freq_shift
                elif row == "base":
                    fmax_index -= freq_shift
                # print(fmin_index)
                # print(fmax_index)
                ##now lets squash the pesky nans
                data = trim_freq(results[row][pol], fmin_index, fmax_index)
                print("data shape is", str(len(data)), str(len(data[0])))
                data = collapse_nan(data)
                data = np.abs(np.array(data))
                if (cell-1)%2 ==0:
                    data = data[0:-8,:]
                    vmax = 6e10
                else:
                    vmax = 2e10

                
                vmin = 0 #np.median(data) - 2*np.std(data)
                
                   
                if row == "base":
                    base_data = np.mean(data, axis=0)
                    auto_data = trim_freq(results["auto"][pol], fmin_index, fmax_index)
                    auto_data = collapse_nan(auto_data)
                    auto_data = np.abs(auto_data)
                    if (cell-1)%2 ==0:
                        auto_data = auto_data[0:-8,:]
                    auto_data = np.mean(auto_data, axis= 0)
                    p = np.polyfit(base_data, auto_data, 1)
                    data = data*p[0] + p[1]
                    if (cell-1)%2 ==0:
                        data = data/6e10
                    else:
                        data = data/2e10
                    myExtents = np.array([freq_range[0], freq_range[1], np.shape(data)[0],0])##should be in minutes give or take
                    im = ax.imshow(data, vmin=vmin, vmax = 1, aspect='auto', extent=myExtents, cmap='YlOrRd')
                else:
                    if (cell-1)%2 ==0:
                        data = data/6e10
                    else:
                        data = data/2e10
                    myExtents = np.array([freq_range[0], freq_range[1], np.shape(data)[0],0])##should be in minutes give or take
                    im = ax.imshow(data, vmin=vmin, vmax = 1, aspect='auto', extent=myExtents, cmap='YlOrRd')
                plotz[str(freq_range[0])+row+pol] = data
                #print(np.shape(data))
                
                
                if ((cell-1)%2)==0:
                    if row == 'base':
                        word = "Baseband"
                    elif row == 'auto':
                        word = "Direct"
                    ax.set_title(word, loc='left', horizontalalignment='right', verticalalignment='bottom')
                if cell -1 ==0:
                    ax.yaxis.set_label_coords(-0.1,1.04)
                if cell-1 == 2:
                    ax.set_xlabel('Frequency (MHz)')
                    ax.set_ylabel('Time (Minutes)')  
                
                #plt.yaxis.set_label_coords(-0.1,1.04)
        fig.colorbar(im, ax=axs[:2,:], shrink=0.8)
        ##now we are only left with the last two 
        for freq_range in freq_ranges:
            ax = axs.flat[cell]
            cell += 1
        

            ##start by getting the bit of results that matters (or at least the interesting indecees)
            fmin_index = find_index(freqs, freq_range[0])
            fmax_index = find_index(freqs, freq_range[1]) -1

            ##now lets squash the pesky nans
            data = trim_freq(results["auto"][pol], fmin_index + freq_shift, fmax_index)
            loc = collapse_nan_loc(data) ##should get the index of fine shit
            
            data = collapse_nan(data)
            auto_data = np.mean(np.array(data),axis = 0)
            mx = np.max(np.abs(auto_data))
            print("auto max", mx)
            auto_dataN = auto_data*5.84552513e-09 + 1.07022679e+01
            
            # data = trim_freq(results["base"][pol], fmin_index, fmax_index -  freq_shift)
            # data = collapse_nan(data)
            base_data = np.mean(np.array(results["high"][pol])[loc,:],axis = 0)
            mx = np.max(np.abs(base_data))
            base_data = base_data 
            print("base max", mx)
            if cell == 5:
                auto_dataN = auto_dataN/1200
                base_dataN = base_data/1200
            else:
                auto_dataN = auto_dataN/190
                base_dataN = base_data/190
            dec_base = np.empty_like(auto_data)
            for i in range(dec_base.size):
                k = base_data.size/dec_base.size
                dec_base[i] = np.mean(base_data[int(i*k):int((i+1)*k)])
            p = np.polyfit(dec_base, auto_data, 1)
            auto_data = (auto_data-p[1])/p[0]

            ax.plot(np.linspace(freqs[fmin_index], freqs[fmax_index], len(auto_dataN)),auto_dataN,  color='#f03523', linestyle='-', linewidth=3, label='Direct')
            ax.plot(np.linspace(freqs[fmin_index], freqs[fmax_index], len(base_dataN)),base_dataN, 'k-', label='Baseband', linewidth=0.7)
            ax.set_ylim(0,1)
            ax.set_xlim(freq_range)
            #ax.plot(np.linspace(freqs[fmin_index], freqs[fmax_index], len(dec_base)),dec_base, label='trash', color="pink")
            if cell-1 ==4:
                ax.legend(loc='upper left')
                ax.set_xlabel('Frequency (MHz)')
                ax.set_ylabel('Magnitude')

            if cell-1 ==5:
                np.save('base.npy', base_data)
                np.save('auto.npy', auto_data)
                np.save('downsample.npy', dec_base)
            ##curve fitting here 
            

            
                
            
            data = trim_freq(results["base"][pol], fmin_index + freq_shift, fmax_index)
            data = collapse_nan(data)
            fit_data = np.array(np.mean(data, axis =0))
            # if freq_range = [5.31,12.63]:
            #     fit_data = fit_[0:-8,:]
            #     auto_data = []
            p = np.polyfit( np.abs(auto_data),np.abs(fit_data),1)
            az.append(p)

            # #from scipy import signal
            # dec_base = congrid(base_data.reshape(-1,1), (auto_data.size,1)).ravel()
            # for i in range(dec_base.size):
            #     k = base_data.size/dec_base.size
            #     dec_base[i] = np.mean(base_data[int(i*k):int((i+1)*k)])
            # #dec_base = signal.resample(np.abs(base_data), len(auto_data))
            # pp = np.polyfit(np.abs(auto_data), np.abs(dec_base),1)
            # ap.append(pp)

        if cell != 6:
            print("simon should take a comp class lol")

        #plt.suptitle("comparison " + str(pol))
        #plt.show()
        plt.savefig(os.path.join(out_dir, str(pol)+".png"))
    print(az)
    print(np.mean(np.array(az),axis = 0))
    print(ap)
    print(np.mean(np.array(ap),axis = 0))
    plt.clf()
    plt.plot(np.mean(plotz["5.31basepol00"],axis=0), label="base")
    plt.plot(np.mean(plotz["5.31autopol00"], axis=0), label="direct")
    plt.legend()
    plt.savefig("testing.png")
    plt.clf()
    plt.imshow(plotz["5.31basepol00"],vmin=0,vmax=6e10)
    plt.colorbar()
    plt.savefig("basetest.png")
    plt.clf()
    plt.imshow(plotz["5.31autopol00"],vmin=0, vmax=6e10)
    plt.colorbar()
    plt.savefig("autotest.png")


    plt.clf()
    

