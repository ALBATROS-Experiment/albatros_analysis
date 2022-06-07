 #temporary testing file
 
from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
import numpy as np
import time


objnew = bdc.BasebandPacked("/project/s/sievers/albatros/uapishka/baseband/snap1/16275/1627528540.raw")
z=cr.avg_xcorr_1bit(objnew.pol0[0,:].reshape(1,-1),objnew.pol0[0,:].reshape(1,-1),436)
print(z)
print(objnew.pol0[0,-1], "last byte")

# histogram check 
# obj=bdc.BasebandPacked('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')
# obj=bdc.Baseband('/project/s/sievers/albatros/uapishka/baseband/snap3/16276/1627622856.raw')
# hist=obj.get_hist(mode=-1)
# hist_str = ','.join([str(n) for n in hist])
# print("hist vals:",hist_str,"\n","Total:", hist.sum())

#missing spectra check
# obj=bdc.BasebandPacked('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')
# xx=np.sum(obj.pol0,axis=1)
# wherezero=np.where(xx==0)
# np.savetxt('/scratch/s/sievers/mohanagr/dump.txt',wherezero)
# check=np.sum(obj.pol0,axis=1)==0
# print(check.sum()) # should be 27990

# obj=bdc2.BasebandPacked('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')

# newauto = cr.autocorr_4bit(obj.pol0)

# t1=time.time()
# manual_sum = np.sum(newauto,axis=0)
# t2=time.time()
# print(f"time taken for manual sum {t2-t1:5.3f}")

# new_avgauto = cr.avg_autocorr_4bit(obj.pol0)

# print(manual_sum-new_avgauto)


# print(new_avgauto)

