import numpy as np
import time


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


def dump_raw_data():
    spec_per_packet = 1
    bits = 4
    chans = get_channels_from_str("452:732", bits)
    bytes_per_spectra = len(chans)
    bytes_per_packet = bytes_per_spectra * spec_per_packet + 4
    spectra = np.ones(bytes_per_spectra, dtype="B")
    specnum = np.asarray([123456789], dtype=">I")
    packet = np.array(
        (specnum, spectra),
        dtype=[("spec_num", ">I"), ("spectra", "%dB" % (len(spectra)))],
    )
    print(packet)
    with open(
        "/home/mohan/Projects/albatros_analysis/correlations/tests/header.raw", "w"
    ) as f:
        write_header(f, chans, spec_per_packet, bytes_per_packet, bits)
        packet.tofile(f)


if __name__ == "__main__":
    dump_raw_data()
