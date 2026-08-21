#!/usr/bin/env python3
"""
probe_baseband.py
-----------------
A standalone command-line tool to quickly probe and summarize essential
information from ALBATROS Baseband .raw files.

Key information summarized:
  - Header parameters & Bit mode (1-bit, 2-bit, 4-bit)
  - Timestamps (Unix ctime & UTC datetime, GPS week/seconds, Trimble lock status)
  - GPS Coordinates (Latitude, Longitude, Elevation)
  - File Duration & Total Spectra count
  - Packet count & Missing packet statistics (missing packet count, missing spectra, fraction missing)
  - Specnum 32-bit integer overflow / wrap detection & correction
  - Reconstructed channel count & distinct contiguous frequency bands (MHz)

Usage:
  python probe_baseband.py <file.raw> [file2.raw ...]
  python probe_baseband.py /path/to/data/*.raw
  python probe_baseband.py -t /path/to/data/*.raw     # Compact table summary
  python probe_baseband.py -v <file.raw>              # Verbose mode (packet gaps & channels)
  python probe_baseband.py --json <file.raw>          # Structured JSON output
"""

import argparse
import datetime
import glob
import json
import os
import struct
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Telescope & digitizer hardware parameters
CLOCK_FREQ_HZ = 250e6      # 250 MSPS digitizer sampling clock
FFT_POINTS = 4096          # 4096-point PFB/FFT
TOTAL_CHANNELS = 2048      # Channels spanning 0 to 125 MHz (Nyquist)
CHAN_BIN_WIDTH_MHZ = (CLOCK_FREQ_HZ / 2.0 / TOTAL_CHANNELS) / 1e6  # 125 / 2048 MHz = 0.06103515625 MHz
DT_SPEC_SEC = FFT_POINTS / CLOCK_FREQ_HZ  # 4096 / 250e6 = 16.384 microseconds per spectrum
GPS_EPOCH_UNIX = 315964800 # 1980-01-06 00:00:00 UTC


@dataclass
class ChannelBand:
    band_index: int
    chan_start: int
    chan_end: int
    num_channels: int
    freq_start_mhz: float
    freq_end_mhz: float
    bandwidth_mhz: float


@dataclass
class PacketGap:
    packet_index: int
    gap_packets: int
    gap_spectra: int
    start_spec_num: int
    end_spec_num: int


@dataclass
class BasebandSummary:
    file_path: str
    file_name: str
    file_size_bytes: int
    file_size_human: str
    header_bytes: int
    bytes_per_packet: int
    spectra_per_packet: int
    bit_mode: int
    have_trimble: bool
    gps_week: int
    gps_timestamp_raw: int
    unix_timestamp: Optional[float]
    utc_time_str: str
    local_time_str: str
    gps_latitude: float
    gps_longitude: float
    gps_elevation: float
    num_packets: int
    unaligned_trailing_bytes: int
    total_recorded_spectra: int
    recorded_duration_sec: float
    has_missing_packets: bool
    missing_packet_count: int
    missing_spectra_count: int
    expected_packets: int
    expected_spectra: int
    missing_fraction: float
    missing_percentage: float
    wall_clock_span_sec: float
    specnum_start: Optional[int]
    specnum_end: Optional[int]
    specnum_overflow_count: int
    num_channels: int
    contiguous_bands: List[ChannelBand] = field(default_factory=list)
    total_bandwidth_mhz: float = 0.0
    packet_gaps: List[PacketGap] = field(default_factory=list)
    channels_list: List[int] = field(default_factory=list)
    raw_header: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def format_bytes(size: int) -> str:
    """Format byte count into human-readable string (B, KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:3.2f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds into human-readable string."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60.0:
        return f"{seconds:.4f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{seconds:.2f} s ({mins}m {secs:04.1f}s)"


def parse_baseband_header(f, file_size: int) -> Dict[str, Any]:
    """Parse the raw baseband binary header from an open binary file object."""
    if file_size < 8:
        raise ValueError(f"File size ({file_size} B) is smaller than 8 bytes; cannot read header size.")

    header_payload_len = struct.unpack(">Q", f.read(8))[0]
    total_header_bytes = 8 + header_payload_len

    if file_size < total_header_bytes:
        raise ValueError(
            f"File size ({file_size} B) is smaller than total header size ({total_header_bytes} B). Header is truncated."
        )

    bytes_per_packet = struct.unpack(">Q", f.read(8))[0]
    length_channels_hdr = struct.unpack(">Q", f.read(8))[0]
    spectra_per_packet = struct.unpack(">Q", f.read(8))[0]
    bit_mode = struct.unpack(">Q", f.read(8))[0]
    have_trimble_val = struct.unpack(">Q", f.read(8))[0]
    have_trimble = bool(have_trimble_val)

    # Number of channel words in header = (header_payload_len - 8*10) / 8
    # 8*10 represents the 10 other 8-byte uint64/double fields in the payload
    n_chan_words = int((header_payload_len - 80) // 8)
    if n_chan_words < 0:
        raise ValueError(f"Invalid header: negative channel count ({n_chan_words}).")

    raw_channels = np.fromfile(f, count=n_chan_words, dtype=">Q")

    gps_week = struct.unpack(">Q", f.read(8))[0]
    gps_timestamp_raw = struct.unpack(">Q", f.read(8))[0]
    gps_latitude = struct.unpack(">d", f.read(8))[0]
    gps_longitude = struct.unpack(">d", f.read(8))[0]
    gps_elevation = struct.unpack(">d", f.read(8))[0]

    # Reconstruct channel array according to bit_mode (consistent with Baseband data classes)
    if bit_mode == 1:
        channels = np.ravel(np.column_stack((raw_channels, raw_channels + 1))).astype(int)
    elif bit_mode == 4:
        channels = raw_channels[::2].astype(int)
    else:  # 2-bit or other bit modes
        channels = raw_channels.astype(int)

    return {
        "header_bytes": total_header_bytes,
        "bytes_per_packet": bytes_per_packet,
        "spectra_per_packet": spectra_per_packet,
        "length_channels_hdr": length_channels_hdr,
        "bit_mode": bit_mode,
        "have_trimble": have_trimble,
        "raw_channels": raw_channels,
        "channels": channels,
        "gps_week": gps_week,
        "gps_timestamp_raw": gps_timestamp_raw,
        "gps_latitude": gps_latitude,
        "gps_longitude": gps_longitude,
        "gps_elevation": gps_elevation,
    }


def find_contiguous_bands(channels: np.ndarray) -> Tuple[List[ChannelBand], float]:
    """Identify distinct contiguous channel bands and compute frequency ranges in MHz."""
    if len(channels) == 0:
        return [], 0.0

    sorted_chans = np.sort(channels)
    diffs = np.diff(sorted_chans)
    split_indices = np.where(diffs != 1)[0] + 1
    band_arrays = np.split(sorted_chans, split_indices)

    bands: List[ChannelBand] = []
    total_bw = 0.0

    for idx, band_arr in enumerate(band_arrays, 1):
        c_start = int(band_arr[0])
        c_end = int(band_arr[-1])
        n_chans = len(band_arr)
        f_start = c_start * CHAN_BIN_WIDTH_MHZ
        f_end = c_end * CHAN_BIN_WIDTH_MHZ
        bw = n_chans * CHAN_BIN_WIDTH_MHZ
        total_bw += bw

        bands.append(
            ChannelBand(
                band_index=idx,
                chan_start=c_start,
                chan_end=c_end,
                num_channels=n_chans,
                freq_start_mhz=f_start,
                freq_end_mhz=f_end,
                bandwidth_mhz=bw,
            )
        )

    return bands, total_bw


def analyze_packets(
    file_path: str,
    header_bytes: int,
    bytes_per_packet: int,
    spectra_per_packet: int,
    num_packets: int,
) -> Dict[str, Any]:
    """Inspect packet spectrum numbers for gaps, missing packets, and 32-bit counter wraps."""
    if num_packets == 0:
        return {
            "has_missing_packets": False,
            "missing_packet_count": 0,
            "missing_spectra_count": 0,
            "expected_packets": 0,
            "expected_spectra": 0,
            "missing_fraction": 0.0,
            "missing_percentage": 0.0,
            "wall_clock_span_sec": 0.0,
            "specnum_start": None,
            "specnum_end": None,
            "specnum_overflow_count": 0,
            "packet_gaps": [],
        }

    try:
        # Memory-map the packet stream to read only the spec_num headers without loading spectra payload
        mm = np.memmap(
            file_path,
            dtype=[("spec_num", ">I"), ("_pad", f"V{bytes_per_packet - 4}")],
            mode="r",
            offset=header_bytes,
            shape=(num_packets,),
        )
        spec_nums = np.array(mm["spec_num"], dtype="int64")
    except Exception:
        # Fallback to buffered sequential reads if memmap is unsupported
        spec_nums = np.empty(num_packets, dtype="int64")
        with open(file_path, "rb") as fp:
            fp.seek(header_bytes)
            for p_idx in range(num_packets):
                raw_pkt = fp.read(bytes_per_packet)
                if len(raw_pkt) < 4:
                    spec_nums = spec_nums[:p_idx]
                    break
                spec_nums[p_idx] = struct.unpack(">I", raw_pkt[:4])[0]

    # Detect and correct for 32-bit unsigned counter overflow / wrap
    wraps = np.where(np.diff(spec_nums) < 0)[0]
    overflow_count = len(wraps)
    for w in wraps:
        spec_nums[w + 1 :] += 1 << 32

    specnum_start = int(spec_nums[0])
    specnum_end = int(spec_nums[-1])

    if len(spec_nums) <= 1:
        total_recorded_spectra = len(spec_nums) * spectra_per_packet
        return {
            "has_missing_packets": False,
            "missing_packet_count": 0,
            "missing_spectra_count": 0,
            "expected_packets": len(spec_nums),
            "expected_spectra": total_recorded_spectra,
            "missing_fraction": 0.0,
            "missing_percentage": 0.0,
            "wall_clock_span_sec": total_recorded_spectra * DT_SPEC_SEC,
            "specnum_start": specnum_start,
            "specnum_end": specnum_end,
            "specnum_overflow_count": overflow_count,
            "packet_gaps": [],
        }

    specdiff = np.diff(spec_nums)
    missing_mask = specdiff != spectra_per_packet
    missing_indices = np.where(missing_mask)[0]

    gaps: List[PacketGap] = []
    if len(missing_indices) > 0:
        missing_spectra_arr = specdiff[missing_indices] - spectra_per_packet
        total_missing_spectra = int(np.sum(missing_spectra_arr))
        total_missing_packets = int(np.sum(missing_spectra_arr // spectra_per_packet))
        expected_spectra = int(specnum_end - specnum_start + spectra_per_packet)
        expected_packets = num_packets + total_missing_packets
        missing_frac = total_missing_spectra / expected_spectra if expected_spectra > 0 else 0.0

        for m_idx in missing_indices:
            m_spec = int(specdiff[m_idx] - spectra_per_packet)
            m_pkt = m_spec // spectra_per_packet
            gaps.append(
                PacketGap(
                    packet_index=int(m_idx),
                    gap_packets=m_pkt,
                    gap_spectra=m_spec,
                    start_spec_num=int(spec_nums[m_idx]),
                    end_spec_num=int(spec_nums[m_idx + 1]),
                )
            )
    else:
        total_missing_spectra = 0
        total_missing_packets = 0
        expected_spectra = num_packets * spectra_per_packet
        expected_packets = num_packets
        missing_frac = 0.0

    wall_clock_span_sec = expected_spectra * DT_SPEC_SEC

    return {
        "has_missing_packets": len(missing_indices) > 0,
        "missing_packet_count": total_missing_packets,
        "missing_spectra_count": total_missing_spectra,
        "expected_packets": expected_packets,
        "expected_spectra": expected_spectra,
        "missing_fraction": missing_frac,
        "missing_percentage": missing_frac * 100.0,
        "wall_clock_span_sec": wall_clock_span_sec,
        "specnum_start": specnum_start,
        "specnum_end": specnum_end,
        "specnum_overflow_count": overflow_count,
        "packet_gaps": gaps,
    }


def probe_baseband(file_path: str, check_packets: bool = True) -> BasebandSummary:
    """Probe a Baseband .raw file and return a comprehensive BasebandSummary object."""
    abs_path = os.path.abspath(file_path)
    file_name = os.path.basename(abs_path)

    if not os.path.exists(abs_path):
        return BasebandSummary(
            file_path=abs_path,
            file_name=file_name,
            file_size_bytes=0,
            file_size_human="0 B",
            header_bytes=0,
            bytes_per_packet=0,
            spectra_per_packet=0,
            bit_mode=0,
            have_trimble=False,
            gps_week=0,
            gps_timestamp_raw=0,
            unix_timestamp=None,
            utc_time_str="N/A",
            local_time_str="N/A",
            gps_latitude=0.0,
            gps_longitude=0.0,
            gps_elevation=0.0,
            num_packets=0,
            unaligned_trailing_bytes=0,
            total_recorded_spectra=0,
            recorded_duration_sec=0.0,
            has_missing_packets=False,
            missing_packet_count=0,
            missing_spectra_count=0,
            expected_packets=0,
            expected_spectra=0,
            missing_fraction=0.0,
            missing_percentage=0.0,
            wall_clock_span_sec=0.0,
            specnum_start=None,
            specnum_end=None,
            specnum_overflow_count=0,
            num_channels=0,
            error=f"File not found: {abs_path}",
        )

    file_size = os.path.getsize(abs_path)
    file_size_human = format_bytes(file_size)

    try:
        with open(abs_path, "rb") as f:
            hdr = parse_baseband_header(f, file_size)
    except Exception as e:
        return BasebandSummary(
            file_path=abs_path,
            file_name=file_name,
            file_size_bytes=file_size,
            file_size_human=file_size_human,
            header_bytes=0,
            bytes_per_packet=0,
            spectra_per_packet=0,
            bit_mode=0,
            have_trimble=False,
            gps_week=0,
            gps_timestamp_raw=0,
            unix_timestamp=None,
            utc_time_str="N/A",
            local_time_str="N/A",
            gps_latitude=0.0,
            gps_longitude=0.0,
            gps_elevation=0.0,
            num_packets=0,
            unaligned_trailing_bytes=0,
            total_recorded_spectra=0,
            recorded_duration_sec=0.0,
            has_missing_packets=False,
            missing_packet_count=0,
            missing_spectra_count=0,
            expected_packets=0,
            expected_spectra=0,
            missing_fraction=0.0,
            missing_percentage=0.0,
            wall_clock_span_sec=0.0,
            specnum_start=None,
            specnum_end=None,
            specnum_overflow_count=0,
            num_channels=0,
            error=f"Failed to read header: {str(e)}",
        )

    header_bytes = hdr["header_bytes"]
    bytes_per_packet = hdr["bytes_per_packet"]
    spectra_per_packet = hdr["spectra_per_packet"]
    bit_mode = hdr["bit_mode"]
    have_trimble = hdr["have_trimble"]
    channels = hdr["channels"]
    gps_week = hdr["gps_week"]
    gps_ts_raw = hdr["gps_timestamp_raw"]
    gps_lat = hdr["gps_latitude"]
    gps_lon = hdr["gps_longitude"]
    gps_elev = hdr["gps_elevation"]

    # Compute packet count and trailing unaligned bytes
    payload_bytes = file_size - header_bytes
    if bytes_per_packet > 0:
        num_packets = payload_bytes // bytes_per_packet
        unaligned_trailing_bytes = payload_bytes % bytes_per_packet
    else:
        num_packets = 0
        unaligned_trailing_bytes = payload_bytes

    total_recorded_spectra = num_packets * spectra_per_packet
    recorded_duration_sec = total_recorded_spectra * DT_SPEC_SEC

    # Timestamp interpretation
    # Modern baseband files write gps_week=0 and gps_timestamp_raw = Unix ctime
    # Legacy baseband files write GPS week and seconds into week
    if gps_week == 0 and gps_ts_raw > 0:
        unix_timestamp = float(gps_ts_raw)
    elif gps_week > 0:
        unix_timestamp = float(GPS_EPOCH_UNIX + gps_week * 7 * 86400 + gps_ts_raw)
    else:
        unix_timestamp = None

    if unix_timestamp is not None:
        try:
            utc_dt = datetime.datetime.fromtimestamp(unix_timestamp, tz=datetime.timezone.utc)
            utc_time_str = utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            local_dt = datetime.datetime.fromtimestamp(unix_timestamp).astimezone()
            local_time_str = local_dt.strftime("%Y-%m-%d %H:%M:%S %Z (%z)")
        except Exception:
            utc_time_str = f"Invalid timestamp ({unix_timestamp})"
            local_time_str = "N/A"
    else:
        utc_time_str = "N/A"
        local_time_str = "N/A"

    # Analyze contiguous frequency bands
    contiguous_bands, total_bw_mhz = find_contiguous_bands(channels)

    # Analyze packet stream for missing packets and counter wrap
    if check_packets and num_packets > 0:
        pkt_stats = analyze_packets(
            abs_path, header_bytes, bytes_per_packet, spectra_per_packet, num_packets
        )
    else:
        pkt_stats = {
            "has_missing_packets": False,
            "missing_packet_count": 0,
            "missing_spectra_count": 0,
            "expected_packets": num_packets,
            "expected_spectra": total_recorded_spectra,
            "missing_fraction": 0.0,
            "missing_percentage": 0.0,
            "wall_clock_span_sec": recorded_duration_sec,
            "specnum_start": None,
            "specnum_end": None,
            "specnum_overflow_count": 0,
            "packet_gaps": [],
        }

    raw_hdr_dict = {
        "header_bytes": header_bytes,
        "bytes_per_packet": bytes_per_packet,
        "length_channels_hdr": hdr["length_channels_hdr"],
        "spectra_per_packet": spectra_per_packet,
        "bit_mode": bit_mode,
        "have_trimble": have_trimble,
        "gps_week": gps_week,
        "gps_timestamp_raw": gps_ts_raw,
        "gps_latitude": gps_lat,
        "gps_longitude": gps_lon,
        "gps_elevation": gps_elev,
    }

    return BasebandSummary(
        file_path=abs_path,
        file_name=file_name,
        file_size_bytes=file_size,
        file_size_human=file_size_human,
        header_bytes=header_bytes,
        bytes_per_packet=bytes_per_packet,
        spectra_per_packet=spectra_per_packet,
        bit_mode=bit_mode,
        have_trimble=have_trimble,
        gps_week=gps_week,
        gps_timestamp_raw=gps_ts_raw,
        unix_timestamp=unix_timestamp,
        utc_time_str=utc_time_str,
        local_time_str=local_time_str,
        gps_latitude=gps_lat,
        gps_longitude=gps_lon,
        gps_elevation=gps_elev,
        num_packets=num_packets,
        unaligned_trailing_bytes=unaligned_trailing_bytes,
        total_recorded_spectra=total_recorded_spectra,
        recorded_duration_sec=recorded_duration_sec,
        has_missing_packets=pkt_stats["has_missing_packets"],
        missing_packet_count=pkt_stats["missing_packet_count"],
        missing_spectra_count=pkt_stats["missing_spectra_count"],
        expected_packets=pkt_stats["expected_packets"],
        expected_spectra=pkt_stats["expected_spectra"],
        missing_fraction=pkt_stats["missing_fraction"],
        missing_percentage=pkt_stats["missing_percentage"],
        wall_clock_span_sec=pkt_stats["wall_clock_span_sec"],
        specnum_start=pkt_stats["specnum_start"],
        specnum_end=pkt_stats["specnum_end"],
        specnum_overflow_count=pkt_stats["specnum_overflow_count"],
        num_channels=len(channels),
        contiguous_bands=contiguous_bands,
        total_bandwidth_mhz=total_bw_mhz,
        packet_gaps=pkt_stats["packet_gaps"],
        channels_list=channels.tolist(),
        raw_header=raw_hdr_dict,
        error=None,
    )


def print_detailed_summary(summary: BasebandSummary, verbose: bool = False, max_gaps: int = 10):
    """Print an aesthetically formatted terminal summary of a baseband file."""
    print("=" * 78)
    print(f" BASEBAND FILE PROBE: {summary.file_name}")
    print("=" * 78)

    if summary.error:
        print(f" ERROR: {summary.error}")
        print("=" * 78)
        return

    # Section 1: File & Acquisition Overview
    print(" [FILE & ACQUISITION]")
    print(f"  • File Path          : {summary.file_path}")
    print(f"  • File Size          : {summary.file_size_human} ({summary.file_size_bytes:,} bytes)")
    print(f"  • Header Size        : {summary.header_bytes} bytes")
    print(f"  • Bit Mode           : {summary.bit_mode}-bit ({'1 pol/byte (2 chan/byte)' if summary.bit_mode==1 else '2 pol/byte' if summary.bit_mode==4 else 'custom'})")
    print(f"  • Bytes per Packet   : {summary.bytes_per_packet:,} bytes")
    print(f"  • Spectra per Packet : {summary.spectra_per_packet}")
    print(f"  • Packets in File    : {summary.num_packets:,} packets")
    if summary.unaligned_trailing_bytes > 0:
        print(f"  • Trailing Bytes     : {summary.unaligned_trailing_bytes} unaligned bytes (incomplete packet)")

    # Section 2: Timestamps & GPS
    print("\n [TIMESTAMPS & GPS]")
    if summary.unix_timestamp is not None:
        print(f"  • Unix Timestamp     : {summary.unix_timestamp:.0f} (ctime)")
        print(f"  • UTC Time           : {summary.utc_time_str}")
        print(f"  • Local Time         : {summary.local_time_str}")
    else:
        print("  • Timestamp          : N/A")

    trimble_str = "Locked / Present (Trimble GPS)" if summary.have_trimble else "No (Raspberry Pi System Clock)"
    print(f"  • GPS Trimble Status : {trimble_str}")
    if summary.gps_week > 0:
        print(f"  • GPS Week / Seconds : Week {summary.gps_week}, {summary.gps_timestamp_raw} sec")

    if summary.have_trimble and (summary.gps_latitude != 0.0 or summary.gps_longitude != 0.0):
        print(f"  • GPS Coordinates    : Lat {summary.gps_latitude:+.6f}°, Lon {summary.gps_longitude:+.6f}°, Elev {summary.gps_elevation:.1f} m")

    # Section 3: Spectra, Duration & Packet Continuity
    print("\n [SPECTRA & DURATION]")
    print(f"  • Recorded Spectra   : {summary.total_recorded_spectra:,} spectra")
    print(f"  • Recorded Duration  : {format_duration(summary.recorded_duration_sec)} ({summary.recorded_duration_sec:.4f} s)")

    if summary.has_missing_packets:
        print(f"  • Wall-Clock Span    : {format_duration(summary.wall_clock_span_sec)} ({summary.wall_clock_span_sec:.4f} s)")
        print(f"  • Missing Packets?   : YES - {summary.missing_packet_count:,} packets missing ({summary.missing_spectra_count:,} spectra)")
        print(f"  • Missing Fraction   : {summary.missing_percentage:.4f}% ({summary.missing_fraction:.6e})")
        print(f"  • Expected Packets   : {summary.expected_packets:,} (recorded: {summary.num_packets:,})")
    else:
        print("  • Missing Packets?   : NO  (0 missing packets, 100% complete)")

    if summary.specnum_start is not None and summary.specnum_end is not None:
        print(f"  • Specnum Range      : {summary.specnum_start:,} -> {summary.specnum_end:,}")
    if summary.specnum_overflow_count > 0:
        print(f"  • Specnum 32-bit Wrap: {summary.specnum_overflow_count} overflow(s) detected and unwrapped")

    # Section 4: Channel & Frequency Bands
    print("\n [CHANNELS & FREQUENCY BANDS]")
    print(f"  • Total Channels     : {summary.num_channels} channels (Nyquist max: {TOTAL_CHANNELS})")
    print(f"  • Total RF Bandwidth : {summary.total_bandwidth_mhz:.4f} MHz ({summary.total_bandwidth_mhz*1e3:.1f} kHz)")
    print(f"  • Distinct Bands     : {len(summary.contiguous_bands)} contiguous band(s):")

    for band in summary.contiguous_bands:
        print(
            f"    └─ Band {band.band_index}: Chans [{band.chan_start:4d} .. {band.chan_end:4d}] "
            f"({band.num_channels:4d} chans) -> "
            f"[{band.freq_start_mhz:8.4f} - {band.freq_end_mhz:8.4f}] MHz "
            f"(Bandwidth: {band.bandwidth_mhz:7.4f} MHz)"
        )

    # Verbose sections: gaps and raw channels
    if verbose:
        if len(summary.packet_gaps) > 0:
            print(f"\n [PACKET DROP GAPS] (Showing up to {min(len(summary.packet_gaps), max_gaps)} of {len(summary.packet_gaps)} gap intervals):")
            for g_i, gap in enumerate(summary.packet_gaps[:max_gaps], 1):
                offset_s = gap.packet_index * summary.spectra_per_packet * DT_SPEC_SEC
                print(
                    f"    #{g_i:2d}: Gap after pkt {gap.packet_index:,} (~{offset_s:.3f} s): "
                    f"{gap.gap_packets:,} missing pkts ({gap.gap_spectra:,} spectra) "
                    f"[Specnums: {gap.start_spec_num} -> {gap.end_spec_num}]"
                )
            if len(summary.packet_gaps) > max_gaps:
                print(f"    ... and {len(summary.packet_gaps) - max_gaps} more gap(s)")

        print("\n [RAW RECONSTRUCTED CHANNELS]")
        chans = summary.channels_list
        if len(chans) <= 30:
            print(f"  {chans}")
        else:
            print(f"  First 15: {chans[:15]}")
            print(f"  Last  15: {chans[-15:]}")

        print("\n [RAW HEADER DICTIONARY]")
        for k, v in summary.raw_header.items():
            print(f"  {k:<22}: {v}")

    print("=" * 78 + "\n")


def print_table_header():
    """Print table header for multi-file summary."""
    header = (
        f"{'Filename':<24} "
        f"{'Bit':>4} "
        f"{'UTC Time':<20} "
        f"{'Duration':>9} "
        f"{'Spectra':>10} "
        f"{'Pkts':>9} "
        f"{'Miss%':>8} "
        f"{'Chans':>6} "
        f"{'Bands':>6} "
        f"{'Freq Bands (MHz)':<35}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))


def print_table_row(summary: BasebandSummary):
    """Print a single concise table row for a baseband file."""
    if summary.error:
        print(f"{summary.file_name:<24} ERROR: {summary.error}")
        return

    bands_str_parts = []
    for b in summary.contiguous_bands:
        bands_str_parts.append(f"{b.freq_start_mhz:.1f}-{b.freq_end_mhz:.1f}")
    bands_str = ", ".join(bands_str_parts)
    if len(bands_str) > 34:
        bands_str = bands_str[:31] + "..."

    dur_str = f"{summary.recorded_duration_sec:.2f}s"
    miss_str = f"{summary.missing_percentage:.3f}%" if summary.has_missing_packets else "0.0%"

    print(
        f"{summary.file_name:<24} "
        f"{summary.bit_mode:>3}b "
        f"{summary.utc_time_str:<20} "
        f"{dur_str:>9} "
        f"{summary.total_recorded_spectra:>10,d} "
        f"{summary.num_packets:>9,d} "
        f"{miss_str:>8} "
        f"{summary.num_channels:>6d} "
        f"{len(summary.contiguous_bands):>6d} "
        f"{bands_str:<35}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Probe and summarize ALBATROS Baseband .raw file headers, timing, packets, and frequency bands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python probe_baseband.py /path/to/1627202039.raw
  python probe_baseband.py /path/to/data/*.raw
  python probe_baseband.py -t /path/to/data/*.raw     # Tabular summary of all files
  python probe_baseband.py -v /path/to/1627202039.raw  # Verbose packet gap info & raw header
  python probe_baseband.py --json /path/to/1627202039.raw
""",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more baseband .raw files to probe (supports glob wildcards).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display detailed output including individual packet drop gaps, channel arrays, and raw header.",
    )
    parser.add_argument(
        "-t",
        "--table",
        action="store_true",
        help="Display compact tabular summary (useful when probing multiple files).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results in JSON format.",
    )
    parser.add_argument(
        "--no-packets",
        action="store_true",
        help="Only inspect header; skip scanning packet headers for missing packet stats.",
    )
    parser.add_argument(
        "--max-gaps",
        type=int,
        default=10,
        help="Maximum number of missing packet gap intervals to display in verbose mode (default: 10).",
    )

    args = parser.parse_args()

    # Expand any glob patterns in input files
    expanded_files = []
    for f in args.files:
        matches = glob.glob(f)
        if matches:
            expanded_files.extend(matches)
        else:
            expanded_files.append(f)

    # Deduplicate while preserving order
    unique_files = list(dict.fromkeys(expanded_files))
    if not unique_files:
        print("No files specified or matched.", file=sys.stderr)
        sys.exit(1)

    summaries: List[BasebandSummary] = []
    for fpath in unique_files:
        summary = probe_baseband(fpath, check_packets=not args.no_packets)
        summaries.append(summary)

    if args.output_json:
        data = [asdict(s) for s in summaries]
        print(json.dumps(data, indent=2))
        return

    if args.table or (len(summaries) > 1 and not args.verbose):
        print_table_header()
        for s in summaries:
            print_table_row(s)
        print("=" * 128)
        print(f"Total files probed: {len(summaries)}")
    else:
        for s in summaries:
            print_detailed_summary(s, verbose=args.verbose, max_gaps=args.max_gaps)


if __name__ == "__main__":
    main()
