import numpy as np
import time

if __name__=="__main__":
    from correlations import baseband_data_classes as bdc
    from correlations import correlations as cr
else:
    from .correlations import baseband_data_classes as bdc
    from .correlations import correlations as cr


def fillarr(arr):
    x=np.array([],dtype='int64')
    for block in arr:
        x=np.append(x,np.arange(block[0],block[1]))
    return x

if __name__ == "__main__":
    obj=bdc.BasebandPacked('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')
    # arr1=[[0,20],[40,70],[90,100]]
    # arr2=[[0,30],[60,100]]
    arr1=[[30,80],[90,100]] #expect 50 rows. 0:10, 20:60
    arr2=[[0,40],[50,100]] #expect 50 rows. 30:70, 80:90
    # arr1=[[30,80],[90,100]] #expect 0 rows
    # arr2=[[0,30]] #expect 0
    a1=fillarr(arr1)+250 #const idx offset doesn't make a difference. pass it below.
    a2=fillarr(arr2)+350
    print(a1,a2)
    p0=np.ones((len(a1),1),dtype='uint8')*17
    p1=np.ones((len(a2),1),dtype='uint8')*18
    print(cr.avg_xcorr_4bit_2ant(p0, p1, a1, a2, 250,350))
