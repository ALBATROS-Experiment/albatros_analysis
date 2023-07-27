from plot_overnight_new import get_data_arrs
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tqdm import tqdm


def live_plot(data_arr, t_i,t_f, vmin=None,vmax=None):
    '''
    Use this to show the nice interactive matplotlib plot
    where you can zoom into thing and all that.
    '''

    y_lims = list(map(dt.datetime.fromtimestamp, [t_i, t_f]))
    y_lims_plt = mdates.date2num(y_lims)
    myext =  [0,125,  y_lims_plt[1], y_lims_plt[0]]

    date_format = mdates.DateFormatter('%Hh%M:%S')

    ext = np.array( [0,125,  y_lims_plt[1], y_lims_plt[0]])
    
    fig,ax=plt.subplots()
    
    ax.imshow(data_arr,vmin=vmin,vmax=vmax,extent=ext,aspect="auto", interpolation = None)
    
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(date_format)

    # plt.show()


if True:

    data_dir = "../../uapishka2022/data_auto_cross_uapishka/"

    #=============== first night =============#
    

    # time_start = 1652582604 
    # time_stop = 1652640343

    # ctime_start, ctime_stop, pol00,pol11,pol01r,pol01i = get_data_arrs(data_dir, time_start, time_stop,ctime=True)


    #=============== second night =============#

    
    # full night
    # time_start = 1652653779
    # time_stop = 1652725966

    # full night + 1hr cut on each side
    # time_start = 1652657389
    # time_stop = 1652722358


    # 12 am EDT to 6 am EDT
    # time_start = 1652686267
    # time_stop=1652707921


    #=============== third night =============#

    # full night
    ctime_start = 1652753630 # 22:17 May 16
    ctime_stop = 1652790294 # 10:25 May 17


    # 3 am EDT to 6 am EDT
    # time_start = 1652772228
    # time_stop=1652779457

    #=============== fourth night =============#

    # time_start = 1652834611 # 8:43 pm
    # time_stop=1652877953 # 8:46 am

    #=============== fifth night =============#
    # ctime_start = 1652895489
    # ctime_stop = 1652967727

    #=============== sixth night =============#
    ctime_start = 1653015714 #May 19 23:01 EDT
    ctime_stop = 1653055436# May 20 10:03 EDT


    #=============== cage_dipole =============#
    # data_dir = "../../uapishka2022/cage_dipole/data_auto_cross_uapishka/"
    
    # ctime_start = 1653192596
    # ctime_stop = 1653210638

    pol00,pol11,pol01r,pol01i = get_data_arrs(data_dir, ctime_start, ctime_stop)
def index_to_hours_since_start(ind):
    return entry_rate*ind/3600

def get_min_med_mean_max(data_arr):
    
    return np.min(data_arr,axis=0),np.median(data_arr,axis=0), np.mean(data_arr,axis=0), np.max(data_arr,axis=0)

arr_inds = np.arange(pol00.shape[0])
entry_rate = 6.44 #s

seconds_since_start = arr_inds*entry_rate #in seconds
hours_since_start = seconds_since_start/3600

freq = np.linspace(0, 125, np.shape(pol00)[1])
# live_plot(np.log10(pol00), ctime_start, ctime_stop, vmin=7,vmax=np.max(np.log10(pol00)))
# live_plot(np.log10(pol00), ctime_start, ctime_stop, vmin=6,vmax=10)

# print(np.max(np.log10(pol00)))
# print(np.max(np.log10(pol11)))
# live_plot(np.log10(pol11), ctime_start, ctime_stop, vmin=7,vmax=np.max(np.log10(pol11)))


# fig, ax = plt.subplots()

# ax.imshow(np.log10(pol00), extent = [0,125,  y_lims_plt[1], y_lims_plt[0]], 
#           aspect='auto')

# ax.yaxis_date()



def mean_15_mins(data_arr):
    chunk_size = 15/(entry_rate/60)
    print(np.round(chunk_size))

# mean_15_mins(pol00)

vmin=6
vmax=12

vmin2 = 6.5
vmax2 = 7.5
big_plot=True
if big_plot:


    y_lims = list(map(dt.datetime.fromtimestamp, [ctime_start, ctime_stop]))
    y_lims_plt = mdates.date2num(y_lims)
    myext =  [0,125,  y_lims_plt[1], y_lims_plt[0]]

    date_format = mdates.DateFormatter('%Hh%M:%S')

    pol00_med = np.median(pol00, axis=0)
    pol11_med = np.median(pol11, axis=0)
    pol00_mean = np.mean(pol00, axis=0)
    pol11_mean = np.mean(pol11, axis=0)
    pol00_max = np.max(pol00, axis = 0)
    pol11_max = np.max(pol11, axis = 0)
    pol00_min = np.min(pol00, axis = 0)
    pol11_min = np.min(pol11, axis = 0)

    fig = plt.figure()

    #top left
    ax0=fig.add_axes([0,1.2,1,1])
    cbarax = fig.add_axes([1.05,1.2,0.03,1])
    im = ax0.imshow(np.log10(pol00),vmin=vmin2, vmax = vmax2, extent = myext, aspect = "auto")
    plt.colorbar(im, cax = cbarax)

    
    # plt.colorbar(im, cax = cbarax)
    ax0.set_title(f"New FEE, starting at {datetime.datetime.fromtimestamp(ctime_start)} Local time")
    ax0.set_xlabel("Frequency [MHz]")
    ax0.yaxis_date()
    ax0.yaxis.set_major_formatter(date_format)

    ax0.set_xlim(110,120)

    #bottom left
    # ax2=fig.add_axes([0,0,1,1])
    # ax2.plot(freq, np.log10(pol00_max), label = "max")
    # ax2.plot(freq, np.log10(pol00_min), label = "min")
    # ax2.plot(freq, np.log10(pol00_mean), label = "mean")
    # ax2.plot(freq, np.log10(pol00_mean), label = "median")
    # ax2.set_xlabel("Frequency [MHz]")
    # ax2.set_ylabel("log10 10 (Power)")
    
    # ax2.set_xlim(110,120)

    # ax1.legend()

    
    #top right
    ax1 = fig.add_axes([1.2, 1.2, 1,1])
    cbarax2 = fig.add_axes([1.2+1.05,1.2,0.03,1])
    im2 = ax1.imshow(np.log10(pol11),vmin=vmin, vmax = vmax, aspect = "auto",extent = myext)
    plt.colorbar(im2, cax = cbarax2)
    ax1.set_title(f"LWA FEE, starting at {datetime.datetime.fromtimestamp(ctime_start)} Local time")
    ax1.set_xlabel("Frequency [MHz]")
    ax1.yaxis_date()
    ax1.yaxis.set_major_formatter(date_format)
    ax1.set_yticklabels([])
    
    ax1.set_xlim(110,120)

    # bottom right
    # ax3 = fig.add_axes([1.06, 0, 1,1])
    # ax3.plot(freq, np.log10(pol11_max), label = "max")
    # ax3.plot(freq, np.log10(pol11_min), label = "min")
    # ax3.plot(freq, np.log10(pol11_mean), label = "mean")
    # ax3.plot(freq, np.log10(pol11_mean), label = "median")
    # ax3.set_xlabel("Frequency [MHz]")
    # ax3.set_ylabel("log10 10 (Power)")
    # ax3.set_yticklabels([])
    # ax3.legend()

    # ax3.set_xlim(110,120)

    start_str = dt.datetime.fromtimestamp(ctime_start).strftime("%m-%d-%Y-%H:%M:%S")
    end_str = dt.datetime.fromtimestamp(ctime_stop).strftime("%m-%d-%Y-%H:%M:%S")
    outfile = f"../../uapishka2022/figures/{start_str}_{end_str}"
    print(f"Writing {outfile}")
    plt.savefig("test.png", bbox_inches="tight")
    plt.savefig(outfile, bbox_inches="tight")
    
    