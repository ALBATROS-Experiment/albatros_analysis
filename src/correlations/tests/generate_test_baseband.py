import numpy as np
import time
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, "data")

def get_channels_from_str(chan, nbits):
    """This function is copied from albatros_daq_utils, used in dump_baseband and config_fpga. Originally writenn by Jon(?). The channels returned by this function essentially correspond to the channel for each byte in the spectrum.
    This is probably due to FPGA file requiring that order. See set_channel_order() in albatrosdigitizer.py
    The analysis code baseband_data_classes.py then compensates for this structure and gives correct number of channels.

    Example
    -------
    If chan = "452:732", and nbits = 1
    You'll get an array that looks like [452, 454, 456,...] because each byte includes two channels

    If chan = "452:732", and nbits = 4
    You'll get an array that looks like [452, 452, 453, 453,...] because you need two bytes for each channel (pol0, pol1)
    """
    new_chans = np.empty(0, dtype=">H")
    multi_chan = chan.split(" ")
    chan_start_stop = []
    for single_chan in multi_chan:
        start_stop = map(int, single_chan.split(":"))
        chan_start_stop.extend(start_stop)
    if nbits == 1:
        for i in range(len(chan_start_stop) // 2):
            new_chans = np.append(
                new_chans,
                np.arange(
                    chan_start_stop[2 * i], chan_start_stop[2 * i + 1], 2, dtype=">H"
                ),
            )
    elif nbits == 2:
        for i in range(len(chan_start_stop) // 2):
            new_chans = np.append(
                new_chans,
                np.arange(
                    chan_start_stop[2 * i], chan_start_stop[2 * i + 1], dtype=">H"
                ),
            )
    else:
        for i in range(len(chan_start_stop) // 2):
            chans = np.arange(
                chan_start_stop[2 * i], chan_start_stop[2 * i + 1], dtype=">H"
            )
            new_chans = np.append(new_chans, np.ravel(np.column_stack((chans, chans))))
    return new_chans


def write_header(file_object, chans, spec_per_packet, bytes_per_packet, bits):
    have_trimble = True
    header_bytes = 8 * 10 + 8 * len(
        chans
    )  # 8 bytes per element in the header. Should be 11.
    # gpsread = lbtools_l.lb_read()
    # gps_time = gpsread[0]
    gps_time = None
    if gps_time is None:
        print("File timestamp coming from RPi clock. This is unreliable.")
        have_trimble = False
        gps_time = time.time()
    print("GPS time is now ", gps_time)
    print("LENGTH OF CHANNELS IS", len(chans))
    print("HEADER BYTES IS", header_bytes)
    file_header = np.asarray(
        [
            header_bytes,
            bytes_per_packet,
            len(chans),
            spec_per_packet,
            bits,
            have_trimble,
        ],
        dtype=">Q",
    )
    file_header.tofile(file_object)
    print("The channels being dumped are:", np.asarray(chans, dtype=">Q"))
    np.asarray(chans, dtype=">Q").tofile(file_object)
    gps_time = np.asarray(
        [0, gps_time], dtype=">Q"
    )  # setting gps_week = 0 to flag the new header format with GPS ctime timestamp
    gps_time.tofile(file_object)
    # lat_lon = gpsread[1]
    lat_lon = None
    if lat_lon is None:
        print("Can't speak to LB, so no position information")
        latlon = {}
        latlon["lat"] = 0
        latlon["lon"] = 0
        latlon["elev"] = 0
    else:
        latlon = {}
        latlon["lat"] = lat_lon[3]
        latlon["lon"] = lat_lon[2]
        latlon["elev"] = lat_lon[4]
        print("lat/lon/elev are ", latlon["lat"], latlon["lon"], latlon["elev"])

    latlon = np.asarray([latlon["lat"], latlon["lon"], latlon["elev"]], dtype=">d")
    latlon.tofile(file_object)


def dump_data(start_specnum, spec_per_packet, num_packets, missing_mask, filename, bits=4):
    """Dump baseband with all spectra bytes set to 1. Allows custom-crafting of missing packets.
    Parameters
    ----------
    start_specnum : int64
        Starting specnum
    num_packets : int
        Total number of packets
    missing_mask : list
        Array of 0's and 1's. numpy.ma format: 1 if missing
        Choose which packets to save and which to "drop" in order to simulate missing packets
    spec_per_packet : int64
        Spectra per packet
    filename : str
        Name of baseband file to save to data directory
    bits : int, optional
        Bit depth of the fake baseband data, by default 4.
        Unpacking codes will use this parameter when decompressing the baseband 
        into floating point complex numbers.
    """
    bits = 4
    chans = get_channels_from_str("452:454", bits)
    bytes_per_spectra = len(chans)
    bytes_per_packet = bytes_per_spectra * spec_per_packet + 4
    print("bytes per packet", bytes_per_packet)
    # spectra = np.ones(bytes_per_packet-4, dtype="B")
    if isinstance(start_specnum,np.ndarray):
        specnums = start_specnum.astype(">I")
    else:
        specnums = (start_specnum + np.arange(num_packets,dtype='int64')*spec_per_packet).astype(">I") #big endian uint32
    print(specnums)
    print(missing_mask)
    with open(
        os.path.join(data_dir, filename), "wb"
    ) as f:
        write_header(f, chans, spec_per_packet, bytes_per_packet, bits)
        for pnum in range(num_packets):
            if not missing_mask[pnum]:
                # chan 0 = 0+1j chan 1 = 1+0j
                # pol0 = pol1
                spectrum=np.ones(bytes_per_spectra,dtype='u1') # 
                spectrum[2:]=spectrum[2:]<<4
                packet = np.asarray(
                    (specnums[pnum], np.tile(spectrum, spec_per_packet)),
                    dtype=[("spec_num", ">I"), ("spectra", "%dB" % (bytes_per_packet-4))],
                )
                packet.tofile(f)
                print(packet)

if __name__ == "__main__":
    # chans = get_channels_from_str("452:454", 4)
    # bytes_per_spectra = len(chans)
    # num_packets=1
    # spec_per_packet=5
    # bytes_per_packet=bytes_per_spectra*spec_per_packet+4
    # print(bytes_per_packet)
    # with open(
    #     os.path.join(data_dir, "test_baseband.raw"), "wb"
    # ) as f:

    #     write_header(f, chans, spec_per_packet, bytes_per_packet, 4)
    #     packet = np.asarray(
    #         (500, np.ones(20,dtype='u1')),
    #         dtype=[("spec_num", ">I"), ("spectra", "%dB" % (20))],
    #     )
    #     packet.tofile(f)
    #     print(packet)
    dump_data(2**32-10,5,7,[0,0,1,0,0,0,0],"specnum_wrap1.raw") #start at 4294967286
    # (4294967286, [ 1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16])
    # (4294967291, [ 1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16])
    # (5,          [ 1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16])
    # (10,         [ 1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16])
    # (15,         [ 1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16])
    # (20,         [ 1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16,  1,  1, 16, 16])
    dump_data(25,5,3,[0,0,0],"specnum_wrap2.raw") #start at 25 continuing from previous file
    dump_data(np.asarray([40,5,10]),5,3,[0,0,0],"specnum_wrap3.raw") #introduce another wrap