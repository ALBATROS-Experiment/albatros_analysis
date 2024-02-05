import numpy as np
import sys
from os.path import dirname, realpath, join

# sys.path.insert(0, "/home/mohan/Projects/albatros_analysis/")
corr_path = dirname(realpath(__file__))  # corr path
data_path = join(corr_path, "data")  # test data path
root_path = join(corr_path, "..", "..")  # project root path
# warning: above line breaks if not in albatros_analysis/correlations/test/ (or two levels deep)
sys.path.insert(0, root_path)

from correlations import baseband_data_classes as bdc

obj = bdc.Baseband(join(data_path, "1667664784.raw"))

hdrbytes = obj.header_bytes  # includes initial bytes

# with open(join(data_path, '1667664784.raw'),mode='rb') as f:
#    hdr=f.read(hdrbytes)

# first chunk
nbytes = 10 * obj.bytes_per_packet
with open(join(data_path, "1667664784.raw"), mode="rb") as f:
    dat1 = f.read(hdrbytes + nbytes)


with open(join(data_path, "1667664784_1.raw"), mode="wb") as f:
    f.write(dat1)
