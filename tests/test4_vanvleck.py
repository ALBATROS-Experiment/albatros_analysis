from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
import numpy as np
import time

files=['/project/s/sievers/albatros/uapishka/202107/baseband/snap3/16275/1627503651.raw']
fileidx=0
idxstart=0
acclen=10000
nchunks=1
nchans=12
ant1 = bdc.BasebandFileIterator(files,fileidx,idxstart,acclen,nchunks=nchunks,chanstart=422,chanend=434)

R0=np.zeros((nchunks,nchans),dtype='float64',order='c')
R1=np.zeros((nchunks,nchans),dtype='float64',order='c')
I0=np.zeros((nchunks,nchans),dtype='float64',order='c')
I1=np.zeros((nchunks,nchans),dtype='float64',order='c')
for i, chunk in enumerate(ant1):
    print(chunk)
    x= cr.avg_xcorr_1bit(chunk['pol0'], chunk['pol1'],chunk['specnums'],nchans)
    print(i+1,"CHUNK READ")