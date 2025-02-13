import numpy as np

# from correlations_temp import baseband_data_classes as bdc
import time
import argparse
from os import path
import sys

sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
from albatros_analysis.src.utils import baseband_utils as butils
import json


def get_init_info_2ant(init_t, end_t, spec_offset, dir_parent0, dir_parent1):
    # spec offset definition:
    # offset in actual spdcrum numbers from two antennas that line up the two timestreams

    f_start0, idx0 = butils.get_file_from_timestamp(init_t, dir_parent0, "f")
    f_end0, _ = butils.get_file_from_timestamp(end_t, dir_parent0, "f")
    files0 = butils.time2fnames(
        butils.get_tstamp_from_filename(f_start0),
        butils.get_tstamp_from_filename(f_end0),
        dir_parent0,
        "f",
        mind_gap=True,
    )

    f0_obj = bdc.Baseband(f_start0)
    specnum0 = f0_obj.spec_num[0] + idx0

    f_start1, idx1 = butils.get_file_from_timestamp(init_t, dir_parent1, "f")
    f_end1, _ = butils.get_file_from_timestamp(end_t, dir_parent1, "f")
    files1 = butils.time2fnames(
        butils.get_tstamp_from_filename(f_start1),
        butils.get_tstamp_from_filename(f_end1),
        dir_parent1,
        "f",
        mind_gap=True,
    )

    f1_obj = bdc.Baseband(f_start1)
    specnum1 = f1_obj.spec_num[0] + idx1

    init_offset = specnum0 - specnum1
    print("before correction", idx0, idx1)
    # idx0 += (spec_offset - init_offset) #needed offset - current offset, adjust one antenna's starting
    idx1 -= spec_offset - init_offset  # the other way around.
    if idx1 < 0:
        raise NotImplementedError(
            "Edge case, idx < 0. Don't start right at the beginning of a file."
        )
    # not handling the edge case for now
    print("after correction", idx0, idx1)
    return files0, idx0, files1, idx1


def get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents):
    """_summary_

    Parameters
    ----------
    init_t : float
        Start time in unix timestamp format.
    end_t : float
        End time in unix timestamp format
    spec_offsets : list
        spectrum number offsets for each antenna with reference to the first antenna.
        first antenna's offset with itself is always set to 0.
    dir_parents : list
        List of paths where each antenna's data (5-digit dirs) resides.

    Returns
    -------
    List
        list of In-file specidx offsets for each antenna, List of files for each antenna

    Raises
    ------
    NotImplementedError
        Note that due to an edge case, the method _might_ fail
        if the start time corresponds to beginning of a file,
        and in-file index pointer needs to seek to past times due to clock offsets.
    """
    # spec offset definition:
    # offset in actual spdcrum numbers from two antennas that line up the two timestreams
    idxs = len(dir_parents) * [0]
    specnums = len(dir_parents) * [0]
    files = []
    for anum, dir_parent in enumerate(dir_parents):
        f_start, idx = butils.get_file_from_timestamp(init_t, dir_parent, "f")
        idxs[anum] = idx
        f_end, _ = butils.get_file_from_timestamp(end_t, dir_parent, "f")
        files.append(
            butils.time2fnames(
                butils.get_tstamp_from_filename(f_start),
                butils.get_tstamp_from_filename(f_end),
                dir_parent,
                "f",
                mind_gap=True,
            )
        )
        f_obj = bdc.Baseband(f_start)
        specnums[anum] = f_obj.spec_num[0] + idx

    for jj in range(1, len(idxs)):  # all except first antenna
        init_offset = specnums[0] - specnums[jj]
        print("before correction", idxs[0], idxs[jj])
        # idx0 += (spec_offset - init_offset) #needed offset - current offset, adjust one antenna's starting
        idxs[jj] -= spec_offsets[jj] - init_offset  # the other way around.
        if idxs[jj] < 0:
            raise NotImplementedError(
                "Edge case, idx < 0. Don't start right at the beginning of a file."
            )
        # not handling the edge case for now
        print("after correction", idxs[0], idxs[jj])
    return idxs, files


def get_avg_fast(
    path1, path2, init_t, end_t, delay, acclen, nchunks, chanstart=0, chanend=None
):
    files1, idxstart1, files2, idxstart2 = get_init_info_2ant(
        init_t, end_t, delay, path1, path2
    )

    print("Starting at: ", idxstart1, "in filenum: ", files1[0], "for antenna 1")
    print("Starting at: ", idxstart2, "in filenum: ", files2[0], "for antenna 2")
    # print(files[fileidx])
    fileidx1 = 0
    fileidx2 = 0
    ant1 = bdc.BasebandFileIterator(
        files1,
        fileidx1,
        idxstart1,
        acclen,
        nchunks=nchunks,
        chanstart=chanstart,
        chanend=chanend,
    )
    ant2 = bdc.BasebandFileIterator(
        files2,
        fileidx2,
        idxstart2,
        acclen,
        nchunks=nchunks,
        chanstart=chanstart,
        chanend=chanend,
    )
    ncols = ant1.obj.chanend - ant1.obj.chanstart
    npols = 2
    polmap = {0: ["pol0", "pol0"], 1: ["pol1", "pol1"]}
    pols = np.zeros((npols, nchunks, ncols), dtype="complex64", order="c")
    rowcounts = np.empty(nchunks, dtype="int64")
    m1 = ant1.spec_num_start
    m2 = ant2.spec_num_start
    st = time.time()
    for i, (chunk1, chunk2) in enumerate(zip(ant1, ant2)):
        # t1=time.time()
        # pol00[i,:] = cr.avg_xcorr_4bit_2ant(chunk1['pol0'], chunk2['pol0'],chunk1['specnums'],chunk2['specnums'],m1+i*acclen,m2+i*acclen)
        # pol00[i,:] = cr.avg_xcorr_4bit_2ant(chunk1['pol0'], chunk2['pol0'],chunk1['specnums'],chunk2['specnums'],m1+i*acclen,m2+i*acclen)
        for pp in range(npols):
            xcorr, rowcount = cr.avg_xcorr_1bit_vanvleck_2ant(
                chunk1[polmap[pp][0]],
                chunk2[polmap[pp][1]],
                ncols,
                chunk1["specnums"],
                chunk2["specnums"],
                m1 + i * acclen,
                m2 + i * acclen,
            )
            if rowcount < 100:
                pols[pp, i, :] = np.nan
            else:
                pols[pp, i, :] = cr.van_vleck_correction(
                    *xcorr, rowcount
                )  # Van Vleck needs unpacked R0,R1,I0,I1
            rowcounts[i] = rowcount
        # t2=time.time()
        # print("time taken for one loop", t2-t1)
        j = ant1.spec_num_start
        # print("After a loop spec_num start at:", j, "Expected at", m1+(i+1)*acclen)
        if i % 1000 == 0:
            print(i + 1, "CHUNK READ")
    print("Time taken final:", time.time() - st)
    pols = np.ma.masked_invalid(pols)
    return pols, rowcounts, ant1.obj.channels

def get_avg_fast2(idxs,files,acclen,nchunks,chanstart,chanend):
    nant = len(idxs)
    
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0, #fileidx is 0 = start idx is inside the first file
            idxs[i],
            acclen,
            nchunks=nchunks,
            chanstart=chanstart,
            chanend=chanend,
        )
        antenna_objs.append(aa)
    print(antenna_objs)
    ncols = aa.obj.chanend - aa.obj.chanstart
    npols = 2
    nbl = nant * (nant - 1) // 2  # 01 02 03...12, 13...
    polmap = {0: ["pol0", "pol0"], 1: ["pol1", "pol1"]}
    print("nant", nant, "nbl", nbl, "nchunks", nchunks, "ncols", ncols)
    vis = np.zeros((nchunks, nbl, npols, ncols), dtype="complex64", order="c")
    rowcounts = np.empty(nchunks, dtype="int64")
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    st = time.time()
    for i, chunks in enumerate(zip(*antenna_objs)):
        bl = 0
        for j in range(nant):
            for k in range(j+1, nant):
                for pp in range(npols):
                    # print(i,bl,pp)
                    xcorr, rowcount = cr.avg_xcorr_1bit_vanvleck_2ant(
                        chunks[j][polmap[pp][0]],
                        chunks[k][polmap[pp][1]],
                        ncols,
                        chunks[j]["specnums"],
                        chunks[k]["specnums"],
                        start_specnums[j] + i * acclen,
                        start_specnums[k] + i * acclen,
                    )
                    if rowcount < 100:
                        vis[i, bl, pp, :] = np.nan
                    else:
                        vis[i, bl, pp, :] = cr.van_vleck_correction(
                            *xcorr, rowcount
                        )  # Van Vleck needs unpacked R0,R1,I0,I1
                    rowcounts[i] = rowcount
                bl += 1
        # t2=time.time()
        # print("time taken for one loop", t2-t1)
        # j=ant1.spec_num_start
        # print("After a loop spec_num start at:", j, "Expected at", m1+(i+1)*acclen)
        if i % 1000 == 0:
            print(i + 1, "CHUNK READ")
    print("Time taken final:", time.time() - st)
    vis = np.ma.masked_invalid(vis)
    return vis, rowcounts, aa.obj.channels


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )
    dir_parents = []
    spec_offsets = []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        # if ant != ref_ant:
        print(ref_ant, ant, details)
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])
    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    print(dir_parents)
    print(spec_offsets)
    print(init_t + 5, end_t)
    idxs, files = get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print(idxs)
    # print(idxs, files)
