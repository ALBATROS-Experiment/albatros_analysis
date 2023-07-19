import numpy as np
import sys
import os

sys.path.insert(0, "/home/mohan/Projects/albatros_analysis/")

print(sys.path)

from correlations import baseband_data_classes as bdc

obj = bdc.Baseband("./data/1667664784.raw")

hdrbytes = obj.header_bytes  # includes initial bytes

# with open('./data/1667664784.raw',mode='rb') as f:
#    hdr=f.read(hdrbytes)

# first chunk
nbytes = 10 * obj.bytes_per_packet
with open("./data/1667664784.raw", mode="rb") as f:
    dat1 = f.read(hdrbytes + nbytes)


with open("./data/1667664784_1.raw", mode="wb") as f:
    f.write(dat1)
