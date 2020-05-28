import numpy as nm
import os, sys, glob, datetime

# Just a stupid convenience script for listing ctime-stamped
# directories and printing out equivalent UTC in human-friendly
# format.  Usage is
#    python utc_ls.py /path/to/directories/*

if __name__ == '__main__':

    args = sys.argv[1:]
    dlist = []
    for arg in args:
        dlist.append(glob.glob(arg))
        
    for d in dlist:
        d = d[0]
        t = d.split('/')[-1]
        # Look for coarse 5-digit time fragments and just add zero padding...
        if len(t) == 5:
            t = t+'00000'
        t = float(t)
        print d+'\t\t'+str(datetime.datetime.utcfromtimestamp(t))
