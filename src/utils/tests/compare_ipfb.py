from matplotlib import pyplot as plt
import numpy as np
import sys
from os import path

sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.utils import pfb_utils as pu
import cupy as cp
import time


nant = 1
npol = 2
osamp = 64
pfb_size = 65536
channels = np.arange(1830, 1840)
lblock = 4096
cutsize = 10
read_size = pfb_size - 2 * cutsize
timestream_size = read_size * lblock
nchan = len(channels)
new_channels = np.arange(osamp) + channels[:, None] * osamp
new_channels = new_channels.ravel()
new_nchan = len(new_channels)
# needs channels that are in the read data
# ipfb1 = pu.StreamingIPFB(
#     nant,
#     npol,
#     channels,
#     nblock=pfb_size,
#     lblock=4096,
#     ntap=4,
#     window="hamming",
#     cut=cutsize,
# )
# fpfb1 = pu.StreamingPFB(
#     nant, npol, timestream_size=timestream_size, lblock=lblock * osamp
# )

ipfb2 = pu.StreamingIPFB_IQ(
    nant,
    npol,
    channels,
    nblock=pfb_size,
    lblock=4096,
    ntap=4,
    window="hamming",
    cut=cutsize,
)
new_lblock = ipfb2.lblock
print("new lblock is", new_lblock)
fpfb2 = pu.StreamingPFB(
    nant, npol, timestream_size=timestream_size, lblock=new_lblock * osamp, dtype='complex'
)

write = True

with np.load("/project/rrg-sievers/mohanagr/albatros_test_data/spectra_1830_1840_4096_24000_0.05_1.00e-10_20260408_192234.npz") as f:
        spec1 = f["spectra1"]
        spec2 = f["spectra2"]
        delays = f["delays"]

# needs channels you want to cross-correlate in re-PFB'd data

carrier = 1834.1888
new_st_chan = int((carrier-2)*osamp)
new_en_chan = int((carrier+2)*osamp)
new_st_chanidx = new_st_chan - channels[0]*osamp
new_en_chanidx = new_en_chan - channels[0]*osamp
print("st idx", new_st_chanidx, "en idx", new_en_chanidx)
assert new_channels[new_st_chanidx]==new_st_chan
assert new_channels[new_en_chanidx]==new_en_chan

new_spec1 = np.zeros((spec1.shape[0] // osamp, new_en_chan-new_st_chan), dtype="complex64")
new_spec2 = new_spec1.copy()

nchunks = spec1.shape[0] // pfb_size

print("new spec shape", new_spec1.shape, "dumping channels", new_st_chan/osamp, "to", new_en_chan/osamp)
print("to process", nchunks, "chunks")

filt_thresh = 0.08
idx1 = 0
idx2 = 0
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()
for chunk_idx in range(nchunks):
    if chunk_idx % 10 == 0:
        print(f"processed {chunk_idx} chunks")
    pol0 = cp.asarray(
        spec1[chunk_idx * read_size : (chunk_idx + 1) * read_size, :], dtype="complex64"
    )
    pol1 = cp.asarray(
        spec2[chunk_idx * read_size : (chunk_idx + 1) * read_size, :], dtype="complex64"
    )
    # start_event.record()
    # ts0 = ipfb1.ipfb(0, 0, pol0, thresh=filt_thresh) # this IPFBs the whole nyquist bandwidth
    # ts1 = ipfb1.ipfb(0, 1, pol1, thresh=filt_thresh)
    ts0 = ipfb2.ipfb(0, 0, pol0, thresh=filt_thresh) # this IPFBs only the channels passed
    ts1 = ipfb2.ipfb(0, 1, pol1, thresh=filt_thresh)
    # end_event.record()
    # end_event.synchronize()
    # print("tot ipfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    # start_event.record()
    pol0_new = fpfb2.pfb(0, 0, ts0)
    pol1_new = fpfb2.pfb(0, 1, ts1)
    # print(pol0_new.shape)
    # print(pol1_new.shape)
    # end_event.record()
    # end_event.synchronize()
    # print("tot pfb time", cp.cuda.get_elapsed_time(start_event, end_event)/1000)
    
    n = pol0_new.shape[0]
    # new_spec1[idx1 : idx1 + n, :] = cp.asnumpy(
    #     pol0_new[:, new_st_chan : new_en_chan]
    # )
    # new_spec2[idx2 : idx2 + n, :] = cp.asnumpy(
    #     pol1_new[:, new_st_chan : new_en_chan]
    # )
    new_spec1[idx1 : idx1 + n, :] = cp.asnumpy(
        pol0_new[:, new_st_chanidx : new_en_chanidx]
    )
    new_spec2[idx2 : idx2 + n, :] = cp.asnumpy(
        pol1_new[:, new_st_chanidx : new_en_chanidx]
    )
    idx1 += n
    idx2 += n
print("final idxs", idx1, idx2)

if write:
    import datetime
    tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(
        f"/scratch/mohanagr/raw_small_{osamp}_{new_st_chan}_{new_en_chan}_{filt_thresh}_{tstamp}.npz",
        spec1=new_spec1,
        spec2=new_spec2,
        channels=np.arange(new_st_chan,new_en_chan),
    )




