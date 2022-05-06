 #temporary testing file
 
import baseband_data_classes as bdc
import numpy as np

# histogram check 

# obj=bdc.Baseband('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')
# obj=bdc.Baseband('/project/s/sievers/albatros/uapishka/baseband/snap3/16276/1627622856.raw')
# hist=obj.get_hist(mode=-1)
# hist_str = ','.join([str(n) for n in hist])
# print("hist vals:",hist_str,"\n","Total:", hist.sum())

#missing spectra check
obj=bdc.BasebandPacked('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')
xx=np.sum(obj.pol0,axis=1)
wherezero=np.where(xx==0)
# np.savetxt('/scratch/s/sievers/mohanagr/dump.txt',wherezero)
# check=np.sum(obj.pol0,axis=1)==0
# print(check.sum()) # should be 27990

