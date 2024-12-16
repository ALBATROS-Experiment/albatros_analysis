import numpy as np
import sys
import os

sys.path.insert(0,'/home/mohan/Projects/albatros_analysis/')

print(sys.path)

from correlations import baseband_data_classes as bdc

obj=bdc.Baseband('./data/1627202039.raw')

hdrbytes = obj.header_bytes #includes initial bytes

with open('./data/1627202039.raw',mode='rb') as f:
    hdr=f.read(hdrbytes)

print(obj.spec_num)
NPACKETS = 10000
trig_packet = 900
trig_spec = 4 #spectra 4 of packet 900 is where it reaches 2**32 = 0

dd = 2**32-obj.spec_num[900] - 4
new_specnum = (obj.spec_num + dd).astype("uint32")

data = np.empty(NPACKETS, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(obj.bytes_per_packet-4))])
data['spec_num']=new_specnum[:NPACKETS]
data['spectra']=obj.raw_data[:NPACKETS]
with open('./data/1627202039_wrap.raw',mode='wb') as f:
    f.write(hdr)
    data.tofile(f)
    



