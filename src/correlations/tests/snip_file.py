# What does this file do?
# Does it get called?
# How is it run?

import numpy as np
import sys
from os.path import dirname, realpath, join


# add project root directory to path
# sys.path.insert(0, "/home/mohan/Projects/albatros_analysis/")
test_path = dirname(realpath(__file__))  # correlations/test absolute path
data_path = join(test_path, "data")
root_path = realpath(
    join(test_path, "..", "..")
)  # warning breaks if not in albatros_analysis/correlations/test/ (or two levels deep)
sys.path.insert(0, root_path)

from correlations import baseband_data_classes as bdc


obj = bdc.Baseband(join(data_path, "1627202039.raw"))

hdrbytes = obj.header_bytes  # includes initial bytes

with open(join(data_path, "1627202039.raw"), mode="rb") as f:
    hdr = f.read(hdrbytes)

# first chunk
nbytes = 65 * obj.bytes_per_packet
with open(join(data_path, "1627202039.raw"), mode="rb") as f:
    dat1 = f.read(hdrbytes + nbytes)
    dat2 = f.read()


with open(join(data_path, "1627202039_1.raw"), mode="wb") as f:
    f.write(dat1)
# sec chunk
with open(join(data_path, "1627202039_2.raw"), mode="wb") as f:
    f.write(hdr)
    f.write(dat2)
