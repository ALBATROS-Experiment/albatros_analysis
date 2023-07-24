import numpy as np
import sys
import os

sys.path.insert(0, "/home/mohan/Projects/albatros_analysis/")

print(sys.path)

from correlations import baseband_data_classes as bdc

obj = bdc.Baseband("./data/1627202039.raw")

hdrbytes = obj.header_bytes  # includes initial bytes

with open("./data/1627202039.raw", mode="rb") as f:
    hdr = f.read(hdrbytes)

# first chunk
nbytes = 65 * obj.bytes_per_packet
with open("./data/1627202039.raw", mode="rb") as f:
    dat1 = f.read(hdrbytes + nbytes)
    dat2 = f.read()


with open("./data/1627202039_1.raw", mode="wb") as f:
    f.write(dat1)
# sec chunk
with open("./data/1627202039_2.raw", mode="wb") as f:
    f.write(hdr)
    f.write(dat2)
