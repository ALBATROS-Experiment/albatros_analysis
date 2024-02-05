import datetime, tabulate, sys, os
import numpy as np

# Just a stupid convenience script for listing ctime-stamped
# directories and printing out equivalent UTC in human-friendly
# format.  Usage is
#    python utc_ls.py /path/to/directories/*



#============================================================  
def files_to_human_time(parent_dir):
    '''
    Author: Joelle, May 2022

    Given path to a directory that contains timestamp directories
    (e.g. 16528/ which contains 1652700704/ and etc), gives the 
    human time of each timestamp directory it contains.
    '''

    files = np.sort(np.array(os.listdir(parent_dir),dtype=int))
    utc_dates = [datetime.datetime.utcfromtimestamp(i).strftime("%m-%d-%Y  %H:%M:%S")for i in files]
    local_dates = [datetime.datetime.fromtimestamp(i).strftime("%m-%d-%Y  %H:%M:%S")for i in files]
    
    d = np.vstack((files,utc_dates,local_dates)).T
    print(tabulate.tabulate(d,headers=["Timestamp","UTC date","Local timezone date"],stralign="center"))


if __name__ == '__main__':

    args = sys.argv[1]
    files_to_human_time(args)
