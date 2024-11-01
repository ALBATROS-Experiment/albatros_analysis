import os
import sys
sys.path.insert(0,os.path.join(os.path.expanduser("~"),'Projects'))
import albatros_analysis.src.baseband_data_classes as bdc
import albatros_analysis.src.utils.pfb_utils as pu

ant=bdc.BasebandIterator()
acclen=cut*1000
pfb_size = acclen + 2*cut
to_ipfb = cp.empty((pfb_size,nchans),dtype='complex64')

# antennas = [ant1,ant2,...]
# for i, chunks in enumerate(zip(*antennas)):
    # for chunk in chunks:
for i, chunk in enumerate(ant):
    pol0=make_continuous(chunk.pol0)
    if i==0:
        raw_pol0 = pu.cupy_ipfb(pol0) 
    else:
        to_ipfb[2*cut:] = pol0
        assert to_ipfb.base is None
        raw_pol0 = ipfb(to_ipfb)
    pol0_new = pu.cupy_pfb(raw_pol0[cut:-cut]) #size a bit smaller acclen - 2*cut for first chunk
    to_ipfb[:2*cut] = pol0[-2*cut:] #store for next iteration

    #do stuff with new pol0