from math import *
import numpy as nm



def find_flag_blocks(flag,count=False):
    """Copied from find_blocks function in Cyrille's glitchtools.py.
    Find the blocks of adjacent flagged points and return first point of each zone
    (and optionally the number of points in the block)."""
    marks = flag[1:]*1-flag[:-1]
    start = nm.where(marks == 1)[0]+1
    if flag[0]:
        start = nm.concatenate([[0],start])
    if count:
        end = nm.where(marks == -1)[0]
        if flag[-1]:
            end = nm.concatenate([end,[len(flag)-1]])
        n = end-start+1
        return start,n
    else:
        return start
        
​
def expandflag(flag, before=5, after=5):
​
    """Expand flagged samples to include specified number of before/after samples.
    Modified from rosset's extflag function in glitchtools.py.
​
    - flag: flag field
    - before: # of samples to flag before each flagged point
    - after: # of samples to flag after each flagged point
​
    Returns expanded flag field."""
​
    inds = nm.where(nm.asarray(flag) > 0)[0]
    flag2 = nm.asarray(flag).copy()
    for i in inds:
        flag2[max(i-before,0):min(i+after+1,flag2.size)] = 1
    return flag2
​
​
def linefill(data,flag):
​
    """Given data and flag field, find blocks of adjacent flagged points and
    draw lines through them.
    
    Returns filled timestream."""
​
    filldata = data.copy()
    start,npts = find_flag_blocks(flag,True)
​
    for i,n in zip(start,npts):
        # Deal with endpoints
        if i == 0:
            # ... Change this into a flagged fraction warning later on...
            if n == nm.size(data):
                exit('linefill: all data points are flagged, FAIL !!!')
            a = data[i+n]
            b = data[i+n]
        elif i+n == len(data):
            a = data[i-1]
            b = data[i-1]
        # Linterp from one point before and after flagged block
        else:
            a = data[i-1]
            b = data[i+n]
        filldata[i:i+n] = nm.linspace(a,b,n+2)[1:n+1]
    
    return filldata
​
​
def gapfill(data,flag,order=3,wn=False):
​
    """A stupid gap filler.  Given data and flag field, fit unflagged data to
    a polynomial baseline and draw the line through the flagged bits.  Option 
    to add white noise based on rms of non-flagged samples
    
    Returns gapfilled timestream."""
​
    from scipy import polyfit
​
    inds = nm.where(flag == 0)[0]
    # Get rms of nonflagged data
    if wn:
        from numpy.random import randn
        rms = nm.std(data[inds])
    # Fit for a baseline
    filldata = data.copy()
    x = nm.array(range(len(filldata)))
    par = polyfit(x[inds], filldata[inds], order)
    p = nm.poly1d(par)
    base = p(x)
    # Put in fake data
    i = nm.where(flag == 1)[0]
    if wn:
        filldata[i] = base[i] + rms*randn(len(i))
    else:
        filldata[i] = base[i]
    
    return filldata