import numpy
import struct
import time
import numba as nb
import numpy as np
import os

if __name__ == "__main__":
    import unpacking as unpk  # If loaded as top level script
else:
    from . import unpacking as unpk  # If loaded as a module

# keeping track of number of times specnum overflows in a given long-averaging run (e.g. several days)


@nb.njit(parallel=True)
def fill_arr(myarr, specnum, spec_per_packet):
    """Fill array with ??
    
    Parameters
    ----------
    myarr: np.ndarray 
        1d array to fill with data, length is len(specnum) * spec_per_packet
    specnum: array-like
        ?? 
    spec_per_packet: int
        Length of ?? 
    """
    n = len(specnum)
    for i in nb.prange(n):
        for j in nb.prange(spec_per_packet):
            myarr[i * spec_per_packet + j] = specnum[i] + j


class Baseband:
    def __init__(self, file_name, readlen=-1, fixoverflow=True):
        """Create instance of Baseband. 
        
        TODO: Explain what the Baseband class is and what it does??
        
        Parameters
        ----------
        file_name: str
            Path to baseband binary file to be read. 
        readlen: int or float
            The number of packets to read. 
            If it is an integer >=1, specifies # of packets to read. 
            If it is a float in (0,1), reads fraction of total packets.
            Defaults to -1, in which case all packets are read. 
        fixoverflow: bool
            Defaults to True. ??
            
        Returns 
        -------
        self: Baseband
        """
        with open(file_name, "rb") as file_data:  # ,encoding='ascii')
            # Declarations [steve: I think it would be nice to have a 
            # comment explaining what each of the attributes are/do, a
            # good place for this is declarations. @Mohan, is this 
            # necessary or too verbose?]
            # Header
            self.header_bytes       = None # Header data (??)
            self.bytes_per_packet   = None # Number (int)
            self.length_channels    = None # Len or num of channels? (int)
            self.spectra_per_packet = None # Number of what per packet??
            self.bit_mode           = None # (int), four bits or one bit (is this 0,1 or 2 or 1,2 or 4??)
            self.have_trimble       = None # (bool??)
            # Data & packet info
            self.channels           = None # Is this the data??
            self.length_channels    = None # ??
            self.read_packets       = None # (int or float) 
            self.raw_data           = None # (np.ndarray)
            self.spec_num           = None # (np.ndarray)
            self.where_zero         = None # (np.ndarray)
            self.missing_num        = None # ?? [not initialized in __init__]
            self.missing_loc        = None # ?? [not initialized in __init__]
            # GPS
            self.gps_week           = None # ??
            self.gps_timestamp      = None # What's the format/units??
            self.gps_latitude       = None # What's the format/units??
            self.gps_longitude      = None # What's the format/units??
            self.gps_elevation      = None # What's the format/units??
            self.specnum_overflow   = None # ?? 
            # TODO: are the above all of the attributes in this class??
            
            header_bytes = struct.unpack(">Q", file_data.read(8))[0]
            # setting all the header values
            self.header_bytes = 8 + header_bytes # Why 8+ ??
            self.bytes_per_packet = struct.unpack(">Q", file_data.read(8))[0]
            self.length_channels = struct.unpack(">Q", file_data.read(8))[0]
            self.spectra_per_packet = struct.unpack(">Q", file_data.read(8))[0]
            self.bit_mode = struct.unpack(">Q", file_data.read(8))[0]
            self.have_trimble = struct.unpack(">Q", file_data.read(8))[0]
            self.channels = numpy.frombuffer(
                file_data.read(self.header_bytes - 88),
                ">%dQ" % (int((header_bytes - 8 * 10) / 8)),
            )[0]  # this line is sketchy but it should work as long as the header structure stays the same. I know there's 88 bytes of the header which is not the channel array, so the rest is the length of the channel array.
            self.gps_week = struct.unpack(">Q", file_data.read(8))[0]
            self.gps_timestamp = struct.unpack(">Q", file_data.read(8))[0]
            self.gps_latitude = struct.unpack(">d", file_data.read(8))[0]
            self.gps_longitude = struct.unpack(">d", file_data.read(8))[0]
            self.gps_elevation = struct.unpack(">d", file_data.read(8))[0]
            self.specnum_overflow = 0

            if self.bit_mode == 1:
                self.channels = numpy.ravel(
                    numpy.column_stack((self.channels, self.channels + 1))
                )
                self.length_channels = int(self.length_channels * 2)
            if self.bit_mode == 4:
                self.channels = self.channels[::2]
                self.length_channels = int(self.length_channels / 2)

            self.num_packets = (
                os.fstat(file_data.fileno()).st_size - self.header_bytes
            ) // self.bytes_per_packet
            if readlen >= 1:
                # interpreted as number of packets
                self.read_packets = int(readlen)
                print("Reading", self.read_packets, "packets")
            elif readlen > 0 and readlen < 1:
                # fraction of file
                self.read_packets = int(self.num_packets * readlen)
                print("Reading", self.read_packets, "packets")
            elif readlen == 0:
                print("Not reading any data")
                self.read_packets = 0
            else:
                self.read_packets = -1

            if self.read_packets != 0:
                file_data.seek(self.header_bytes)
                t1 = time.time()
                data = numpy.fromfile(
                    file_data,
                    count=self.read_packets,
                    dtype=[
                        ("spec_num", ">I"),
                        ("spectra", "%dB" % (self.bytes_per_packet - 4)),
                    ],
                )
                t2 = time.time()
                print(f"took {t2-t1:5.3f} seconds to read raw data on ", file_name)
                self.raw_data = numpy.array(data["spectra"], dtype="uint8")
                self.spec_num = numpy.array(data["spec_num"], dtype="int64")
                # check for specnum overflow in current file
                self.where_zero = np.where(np.diff(self.spec_num) < 0)[0]
                if fixoverflow:
                    if len(self.where_zero) == 1:
                        self.spec_num[self.where_zero[0] + 1 :] += 2**32
                        self._set_specidx()
                    elif len(self.where_zero) > 1:
                        raise ValueError(
                            "Why are there two -ve diffs in specnum? Investigate this file"
                        )
                self._set_specidx()
        return 

    def _set_specidx(self):
        self.spec_idx = numpy.zeros(
            self.spec_num.shape[0] * self.spectra_per_packet, dtype="int64"
        )  # keep dtype int64 otherwise numpy binary search becomes slow
        fill_arr(self.spec_idx, self.spec_num, self.spectra_per_packet)
        specdiff = numpy.diff(self.spec_num)
        idx = numpy.where(specdiff != self.spectra_per_packet)[0]
        self.missing_loc = (
            self.spec_num[idx] + self.spectra_per_packet - self.spec_num[0]
        ).astype("int64")
        self.missing_num = (specdiff[idx] - self.spectra_per_packet).astype("int64")
        return 

    def __str__(self):
        """Calls print_headers()"""
        self.print_header()
        return 

    def print_header(self):
        """Formats and prints a string displaying header info."""
        print(
            "Header Bytes = "
            + str(self.header_bytes)
            + ". Bytes per packet = "
            + str(self.bytes_per_packet)
            + ". Channel length = "
            + str(self.length_channels)
            + ". Spectra per packet: "
            + str(self.spectra_per_packet)
            + ". Bit mode: "
            + str(self.bit_mode)
            + ". Total packets = "
            + str(self.num_packets)
            + ". Read packets = "
            + str(self.read_packets)
            + ". Have trimble = "
            + str(self.have_trimble)
            + ". Channels: "
            + str(self.channels)
            + " GPS week = "
            + str(self.gps_week)
            + ". GPS timestamp = "
            + str(self.gps_timestamp)
            + ". GPS latitude = "
            + str(self.gps_latitude)
            + ". GPS longitude = "
            + str(self.gps_longitude)
            + ". GPS elevation = "
            + str(self.gps_elevation)
            + "."
        )
        return 

    def get_hist(self, mode=-1):
        """Get
        
        Parameters
        ----------
        mode: int
            mode=0 for pol0, 1 for po1, -1 for both.
        
        Returns
        -------
        histvals: ??
            ??
        """
        # mode = 0 for pol0, 1 for pol1, -1 for both
        rowstart = 0
        rowend = len(self.spec_idx)
        return unpk.hist(
            self.raw_data, rowstart, rowend, self.length_channels, self.bit_mode, mode
        )


def get_header(file_name, verbose=True):
    """Get header dictionary from (attributes of) baseband file. 
    
    Loads the data at <file_name> by instantiating a Baseband object, 
    returns the object's attributes (self.__dict__), incl header info.
    
    Parameters
    ----------
    file_name: str
        Path to baseband data file. 
    verbose: bool
        Defaults to True, in which case header data is printed.
    
    Returns
    -------
    dict
        Baseband dictionary containing header information. 
    """
    obj = Baseband(file_name, readlen=0)
    if verbose:
        obj.print_header()
    return obj.__dict__


class BasebandFloat(Baseband):
    def __init__(
        self, file_name, readlen=-1, fixoverflow=True, chanstart=0, chanend=None
    ):
        """Create instance of BasebandFloat.
        
        A child of the Baseband class. BasebandFloat... TODO: describe??
        
        Parameters
        ----------
        file_name: str
            Path to baseband binary file to be read. 
        readlen: int or float
            The number of packets to read. 
            If it is an integer >=1, specifies # of packets to read. 
            If it is a float in (0,1), reads fraction of total packets.
            Defaults to -1, in which case all packets are read. 
        fixoverflow: bool
            Defaults to True. ?? *warning: not passed to super*
        chanstart: int
            Index of channel at which to start selection. Default is 0.
        chanend: int or None
            Index of channel at which to end selection. Default is None
            in which case select up to highest frequency channel. 
        """
        super().__init__(file_name, readlen) # why not pass fixoverflow too??
        self.chanstart = chanstart
        if chanend == None:
            self.chanend = self.length_channels
        else:
            self.chanend = chanend

        if self.bit_mode == 4:
            self.pol0, self.pol1 = unpk.unpack_4bit(
                self.raw_data,
                self.length_channels,
                0,
                len(self.spec_idx),
                self.chanstart,
                self.chanend,
            )
        elif self.bit_mode == 1:
            self.pol0, self.pol1 = unpk.unpack_1bit(
                self.raw_data, self.length_channels, self.chanstart, self.chanend
            )
        else:
            print("Unknown bit depth")
        return 


class BasebandPacked(Baseband):
    # turn spec_selection to true and enter the range of spectra you want to save only part of the file
    def __init__(
        self,
        file_name,
        readlen=-1,
        fixoverflow=True, 
        rowstart=None,
        rowend=None,
        chanstart=0,
        chanend=None,
        unpack=True,
    ):
        """Create instance of BasebandPacked.
        
        A child of the Baseband class. BasebandPacked... ??
        
        Parameters
        ----------
        file_name: str
            Path to baseband binary file to be read. 
        readlen: int or float
            The number of packets to read. 
            If it is an integer >=1, specifies # of packets to read. 
            If it is a float in (0,1), reads fraction of total packets.
            Defaults to -1, in which case all packets are read. 
        fixoverflow: bool
            Defaults to True. ?? *Depricated*
        chanstart: int
            Index of channel at which to start selection. Default is 0.
        chanend: int or None
            Index of channel at which to end selection. Default is None
            in which case select up to highest frequency channel. 
        """

        super().__init__(file_name, readlen) # why not pass fixoverflow too??

        # self.spec_idx2 = self.spec_num - self.spec_num[0]
        self.chanstart = chanstart
        if chanend == None:
            self.chanend = self.length_channels
        else:
            self.chanend = chanend

        if unpack:
            if rowstart and rowend:
                self.pol0, self.pol1 = self._unpack(rowstart, rowend)
            else:
                self.pol0, self.pol1 = self._unpack(0, len(self.spec_idx))

    def _unpack(self, rowstart, rowend):
        # There should NOT be an option to modify channels you're working with in a private function.
        # If you want different set of channels, create a new object
        return unpk.sortpols(
            self.raw_data,
            self.length_channels,
            self.bit_mode,
            rowstart,
            rowend,
            self.chanstart,
            self.chanend,
        )


def get_rows_from_specnum(stidx, endidx, spec_arr):
    """??
    
    ??
    
    Parameters
    ----------
    stidx: int
        Start index of... ??
    endidx: int
        End index of... ??
    spec_arr: np.ndarray
        ??
    
    Returns
    -------
    l: int??
        Left...?? 
    r: int??
        Right...??    
    """
    # follows numpy convention
    # endidx is assumed not included
    # print("utils get_rows received:",stidx,endidx,spec_arr)
    l = np.searchsorted(spec_arr, stidx, side="left")
    r = np.searchsorted(spec_arr, endidx, side="left")
    return l, r


class BasebandFileIterator:
    def __init__(
        self,
        file_paths,
        fileidx,
        idxstart,
        acclen,
        nchunks=None,
        chanstart=0,
        chanend=None,
    ):
        """Create instance of BasebandFileIterator.
        
        ??
        
        Parameters
        ----------
        file_paths: list of str
            Paths to baseband binary files to be read. 
        fileidx: int
            ??
        idxstart: int
            ??
        acclen: int
            ??
        nchunks: int
            Defaults to None. You need to pass nchunks if you are 
            passing the iterator to zip(). Without nchunks, iteration
            won't stop. 
        chanstart: int
            Index of channel at which to start selection. Default is 0.
        chanend: int or None
            Index of channel at which to end selection. Default is None
            in which case select up to highest frequency channel. 
        """
        print("ACCLEN RECEIVED IS", acclen)
        self._OVERFLOW_DICT = {}
        self._OVERFLOW_CTR = 0
        self.acclen = acclen
        self.file_paths = file_paths
        self.fileidx = fileidx
        self.nchunks = nchunks
        self.chunksread = 0
        self.chanstart = chanstart
        self.chanend = chanend
        self.obj = BasebandPacked(
            file_paths[fileidx], chanstart=chanstart, chanend=chanend, unpack=False
        )
        self.spec_num_start = (
            idxstart + self.obj.spec_idx[0]
        )  # REPLACE SPEC_IDX to be SPEC_NUM, not 0 indexed
        print(
            "START SPECNUM IS",
            self.spec_num_start,
            "obj start at",
            self.obj.spec_num[0],
        )
        if self.obj.bit_mode == 4:
            self.ncols = (
                self.obj.chanend - self.obj.chanstart
            )  # gotta be careful with this for 1 bit and 2 bit. for 4 bits, ncols = nchans
        elif self.obj.bit_mode == 1:
            if self.obj.chanstart % 2 > 0:
                raise ValueError("ERROR: Start channel index must be even.")
            self.ncols = numpy.ceil((self.obj.chanend - self.obj.chanstart) / 4).astype(
                int
            )

    def _read_packed(
        self,
        file_name,
        readlen=-1,
        fixoverflow=True,
        rowstart=None,
        rowend=None,
        chanstart=0,
        chanend=None,
        unpack=True,
    ):
        temp_obj = BasebandPacked(
            file_name,
            fixoverflow=False,
            chanstart=chanstart,
            chanend=chanend,
            unpack=False,
        )
        temp_obj.spec_num[:] += (
            self._OVERFLOW_CNTR * 2**32
        )  # correct for past overflows in this run of averaging
        if len(temp_obj.where_zero) == 1:
            if file_name not in self._OVERFLOW_DICT.keys():
                self._OVERFLOW_DICT[file_name] = 1
                self._OVERFLOW_CTR += 1
                temp_obj.spec_num[temp_obj.where_zero[0] + 1 :] += 2**32
            elif len(temp_obj.where_zero) > 1:
                raise ValueError(
                    f"Why are there two -ve diffs in specnum? Investigate {file_name}"
                )
        temp_obj._set_specidx()
        return temp_obj

    def __iter__(self):
        return self

    def __next__(self):
        t1 = time.time()
        print(
            "Current obj first spec vs acc start",
            self.obj.spec_idx[0],
            self.spec_num_start,
        )
        if self.nchunks and self.chunksread == self.nchunks:
            raise StopIteration
        pol0 = numpy.zeros(
            (self.acclen, self.ncols), dtype="uint8", order="c"
        )  # for now take all channels. will modify to accept chanstart, chanend
        pol1 = numpy.zeros((self.acclen, self.ncols), dtype="uint8", order="c")
        specnums = numpy.array(
            [], dtype="int64"
        )  # len of this array will control everything in corr, neeeeed the len.
        rem = self.acclen
        i = 0
        while rem:
            print("Rem is", rem)
            if self.spec_num_start < self.obj.spec_num[0]:
                # we are in a gap between the files
                print("IN A GAP BETWEEN FILES")
                step = min(self.obj.spec_num[0] - self.spec_num_start, rem)
                rem -= step
                # i+=self.acclen-rem
                self.spec_num_start += step
            else:
                l = (
                    self.obj.spec_idx[-1] - self.spec_num_start + 1
                )  # length to end from the point in file we're starting from
                # print("dist to end is", l, "rem is", rem)
                if rem >= l:
                    # spillover to next file.

                    rowstart, rowend = get_rows_from_specnum(
                        self.spec_num_start, self.spec_num_start + l, self.obj.spec_idx
                    )
                    print(
                        "From if:, rowstart, rowend",
                        rowstart,
                        rowend,
                        rowend - rowstart,
                    )
                    specnums = numpy.append(
                        specnums, self.obj.spec_idx[rowstart:rowend]
                    )
                    # print("len specnum from new file", rowend-rowstart)
                    rem -= l
                    (
                        pol0[i : i + rowend - rowstart],
                        pol1[i : i + rowend - rowstart],
                    ) = self.obj._unpack(rowstart, rowend)
                    i += rowend - rowstart
                    self.spec_num_start += l
                    # print("Reading new file")
                    self.fileidx += 1
                    self.obj = BasebandPacked(
                        self.file_paths[self.fileidx],
                        chanstart=self.chanstart,
                        chanend=self.chanend,
                        unpack=False,
                    )
                    print(
                        "My specnum pointer at",
                        self.spec_num_start,
                        "first specnum of new obj",
                        self.obj.spec_num[0],
                    )
                else:
                    rowstart, rowend = get_rows_from_specnum(
                        self.spec_num_start,
                        self.spec_num_start + rem,
                        self.obj.spec_idx,
                    )
                    print(
                        "From else:, rowstart, rowend",
                        rowstart,
                        rowend,
                        rowend - rowstart,
                    )
                    specnums = numpy.append(
                        specnums, self.obj.spec_idx[rowstart:rowend]
                    )
                    # print("len specnum from else", rowend-rowstart)
                    (
                        pol0[i : i + rowend - rowstart],
                        pol1[i : i + rowend - rowstart],
                    ) = self.obj._unpack(rowstart, rowend)
                    self.spec_num_start += rem
                    rem = 0
                    i += rowend - rowstart
        # print(pol0[len(specnums)-1,:])
        # print(pol0[len(specnums),:])
        self.chunksread += 1
        data = {"pol0": pol0, "pol1": pol1, "specnums": specnums}
        t2 = time.time()
        # print("TIME TAKEN FOR RETURNING NEW OBJECT",t2-t1)
        return data
