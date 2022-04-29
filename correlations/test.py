 #temporary testing file
 
import baseband_data_classes as bdc
import numpy as np
# obj=bdc.Baseband('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')
# hist=obj.get_hist(mode=0)
# print(hist, hist.sum())

obj=bdc.BasebandPacked('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')
check=np.sum(obj.pol0,axis=1)==0
print(check.sum()) # should be 27990