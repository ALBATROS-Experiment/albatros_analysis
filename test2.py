from correlations import baseband_data_classes as bdc
from correlations import correlations as cr
import numpy as np

obj = bdc.BasebandPacked('/project/s/sievers/albatros/uapishka/baseband/snap1/16272/1627202039.raw')

new_avgxcorr = cr.avg_xcorr_4bit(obj.pol0,obj.pol1)

