import struct
import time
import numba as nb
import numpy as np
from .. import xp
import os

print("BDC is using", xp.__name__)

if __name__ == "__main__":
    import unpacking as unpk  # If loaded as top level script
else:
    from . import unpacking as unpk  # If loaded as a module

@nb.njit(parallel=True)
def fill_arr_cpu(specnum, spec_per_packet):
    n = len(specnum)
    arr = np.empty(n*spec_per_packet, dtype=specnum.dtype)
    for i in nb.prange(n):
        for j in range(spec_per_packet):
            # print(j, specnum[i], j+specnum[i])
            arr[i * spec_per_packet + j] = specnum[i] + j
    return arr

def fill_arr_gpu(specnum, spec_per_packet):
    print("fill arr gpu was called")
    return (specnum[:,None] + xp.arange(spec_per_packet,dtype=specnum.dtype)).ravel() #broadcast magic

def fill_arr(specnum, spec_per_packet):
    """Make the spectrum number array continuous. Raw spec_num has starting spectrum number of each packet.
    This function expands each packet. Array is required in x-corr computation for handling missing packets.

    Parameters
    ----------
    specnum: xp.ndarray
        Raw spectrum number array. If GPU enabled, it resides on device
    spec_per_packet: int
        Number of spectrums in a single packet (stored in file header).
    
    Returns
    ---------
    xp.ndarray
        Expanded array. If GPU enabled, it resides on device.
    """
    if xp.__name__=='numpy': return fill_arr_cpu(specnum, spec_per_packet)
    elif xp.__name__=='cupy': return fill_arr_gpu(specnum,spec_per_packet)

def make_continuous_gpu(spec, specnum, channels, nspec, nchans=2049, out=None):
    if len(specnum)==nspec and len(channels)==nchans:
        if out is not None:
            out[:]=spec
            return out #nothing to do if nothing's missing and all channels populated
        else:
            return spec
    if out is None:
        out=xp.zeros((nspec, nchans), dtype=spec.dtype)
    out[xp.ix_(specnum,channels)] = spec[:len(specnum)]
    # print("specnum is", specnum)
    assert out.base is None
    return out

@nb.njit(parallel=True)
def add_constant_cpu(arr,const):
    n = len(arr)
    for i in nb.prange(n):
        arr[i] += const

def add_constant_gpu(arr,const):
    return arr + const

def add_constant(arr,const):
    """Helper function to add a constant to an array. 
    Function typically used for handling spectrum number int32 wrapping.
    Parallelized since spectrum number array can be O(1e7) long, 
    and addition needs to happen for each file after a wrap is encountered.

    Parameters
    ----------
    arr: np.ndarray
        1d array to which constant will be added.
    const: int
        constant to add
    """
    if xp.__name__=='numpy': add_constant_cpu(arr, const)
    elif xp.__name__=='cupy': add_constant_gpu(arr,const)

class Baseband:
    def __init__(self, file_name, readlen=-1):
        """Create instance of Baseband object.
        Headers and spec_num always stored on host memory.
        Raw_data can be stored on either host/device depending on whether GPU is in use.
        Storing raw data on GPU enables much faster IPFB->PFB->Correlation step.
        Specnum can be passed to GPU when needed, no need to do so by default.

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
        fixoverflow: int
            Defaults to 1. Number of cycles of int32 wrap that need to be undone in spectrum numbers.

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
            self.header_bytes = None  # Header data (??)
            self.bytes_per_packet = None  # Number (int)
            self.length_channels = None  # Len or num of channels? (int)
            self.spectra_per_packet = None  # Number of what per packet??
            self.bit_mode = (
                None  # (int), four bits or one bit (is this 0,1 or 2 or 1,2 or 4??)
            )
            self.have_trimble = None  # (bool??)
            # Data & packet info
            self.channels = None  # Is this the data??
            self.length_channels = None  # ??
            self.read_packets = None  # (int or float)
            self.raw_data = None  # (np.ndarray)
            self.spec_num = None  # (np.ndarray)
            self.where_zero = None  # (np.ndarray)
            self.missing_num = None  # ?? [not initialized in __init__]
            self.missing_loc = None  # ?? [not initialized in __init__]
            # GPS
            self.gps_week = None  # ??
            self.gps_timestamp = None  # What's the format/units??
            self.gps_latitude = None  # What's the format/units??
            self.gps_longitude = None  # What's the format/units??
            self.gps_elevation = None  # What's the format/units??
            self.specnum_overflow = None  # ??
            # TODO: are the above all of the attributes in this class??

            header_bytes = struct.unpack(">Q", file_data.read(8))[0]
            # setting all the header values
            self.header_bytes = 8 + header_bytes  # Why 8+ ??
            self.bytes_per_packet = struct.unpack(">Q", file_data.read(8))[0]
            self.length_channels = struct.unpack(">Q", file_data.read(8))[0]
            self.spectra_per_packet = struct.unpack(">Q", file_data.read(8))[0]
            self.bit_mode = struct.unpack(">Q", file_data.read(8))[0]
            self.have_trimble = struct.unpack(">Q", file_data.read(8))[0]
            self.channels = np.frombuffer(
                file_data.read(self.header_bytes - 88),
                ">%dQ" % (int((header_bytes - 8 * 10) / 8)),
            )[
                0
            ]  # this line is sketchy but it should work as long as the header structure stays the same. I know there's 88 bytes of the header which is not the channel array, so the rest is the length of the channel array.
            self.gps_week = struct.unpack(">Q", file_data.read(8))[0]
            self.gps_timestamp = struct.unpack(">Q", file_data.read(8))[0]
            self.gps_latitude = struct.unpack(">d", file_data.read(8))[0]
            self.gps_longitude = struct.unpack(">d", file_data.read(8))[0]
            self.gps_elevation = struct.unpack(">d", file_data.read(8))[0]
            self._overflowed = False
            self._spec_idx = None

            if self.bit_mode == 1:
                self.channels = np.ravel(
                    np.column_stack((self.channels, self.channels + 1))
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
                data = np.fromfile(
                    file_data,
                    count=self.read_packets,
                    dtype=[
                        ("spec_num", ">I"),
                        ("spectra", "%dB" % (self.bytes_per_packet - 4)),
                    ],
                )
                t2 = time.time()
                print(f"took {t2-t1:5.3f} seconds to read raw data on ", file_name)
                self.raw_data = xp.array(data["spectra"], dtype="uint8", order='c') #only raw data in GPU (if enabled)
                self.spec_num = np.array(data["spec_num"], dtype="int64", order='c')
                # check for specnum overflow in current file
                self._wrap_loc = np.where(np.diff(self.spec_num) < 0)[0]
                if len(self._wrap_loc) == 1:
                    self.spec_num[self._wrap_loc[0] + 1 :] += 2**32
                    self._overflowed = True
                    print("file overflowed")
                elif len(self._wrap_loc) > 1:
                    raise ValueError(
                        "Why are there two -ve diffs in specnum? Investigate this file"
                    )
        return

    @property
    def spec_idx(self):
        # Lazy initialization, since no need to spend time to compute this unless accessed by someone.
        if self._spec_idx is None:
            self._spec_idx = fill_arr_cpu(self.spec_num, self.spectra_per_packet)
            # print("spec idx is", self._spec_idx)
            specdiff = np.diff(self.spec_num)
            idx = np.where(specdiff != self.spectra_per_packet)[0]
            self._missing_loc = (
                self.spec_num[idx] + self.spectra_per_packet - self.spec_num[0]
            ).astype("int64")
            self._missing_num = (specdiff[idx] - self.spectra_per_packet).astype("int64")
        return self._spec_idx
        

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
        rowend = self.spec_num.shape[0] * self.spectra_per_packet
        return unpk.hist(
            self.raw_data, rowstart, rowend, self.length_channels, self.bit_mode, mode
        )
    
    def _assign_channels(self,channels=None,chanstart=None,chanend=None):
        """
        Helper function for child classes
        """
        if isinstance(channels, np.ndarray):
                channels.sort()
                if channels.dtype == "int64":
                    self.channel_idxs = channels
                    self.chanstart = channels[0]
                    self.chanend = len(channels) + channels[0] #numpy slice convention
                else:
                    raise ValueError("channels should be integers!")
        elif isinstance(channels, list) or isinstance(channels, tuple):
            channels.sort()
            self.channel_idxs = np.asarray(channels,dtype="int64")
            self.chanstart = channels[0]
            self.chanend = len(channels) + channels[0]
        else:
            if chanstart == None:
                self.chanstart = 0
            else:
                self.chanstart = chanstart
            if chanend == None:
                self.chanend = self.length_channels
            else:
                self.chanend = chanend
            self.channel_idxs = np.arange(self.chanstart, self.chanend, dtype="int64")


def get_header(file_name, verbose=False):
    """Get header dictionary from (attributes of) baseband file.

    Only reads the header of <file_name> by instantiating a Baseband object with readlen=0.
    Returns the object's public attributes which is the header information.

    Parameters
    ----------
    file_name: str
        Path to baseband data file.
    verbose: bool
        Defaults to True, in which case header data is printed.

    Returns
    -------
    dict
        Dictionary containing header information.
    """
    obj = Baseband(file_name, readlen=0)
    if verbose:
        obj.print_header()
    return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}


class BasebandFloat(Baseband):
    def __init__(
        self,
        file_name,
        readlen=-1,
        rowstart=None,
        rowend=None,
        channels=None,
        chanstart=0,
        chanend=None,
        unpack=True,
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
        channels: array like
            List of channel indices that must be unpacked. If passed, chanstart and chanend ignored.
        chanstart: int
            Index of channel at which to start selection. Default is 0.
            Used to instantiate `channels` if `channels` is None.
        chanend: int or None
            Index of channel at which to end selection. Default is None
            in which case select up to highest available frequency channel.
            Used to instantiate `channels` if `channels` is None.
        """
        super().__init__(file_name, readlen)
        self._assign_channels(channels=channels,chanstart=chanstart,chanend=chanend)
            
        if unpack:
            if rowstart and rowend:
                self.pol0, self.pol1 = self._unpack(rowstart, rowend)
            else:
                self.pol0, self.pol1 = self._unpack(0, len(self.spec_idx))
        return

    def _unpack(self, rowstart, rowend):
        # There should NOT be an option to modify channels you're working with in a private function.
        # If you want different set of channels, create a new object
        if self.bit_mode == 4:
            return unpk.unpack_4bit(
                self.raw_data,
                rowstart,
                rowend,
                self.channel_idxs,
                self.length_channels
            )
        elif self.bit_mode == 1: #TODO: fix the function call
            return unpk.unpack_1bit(
                self.raw_data,
                self.length_channels,
                rowstart,
                rowend,
                self.chanstart,
                self.chanend,
            )

class BasebandPacked(Baseband):
    # turn spec_selection to true and enter the range of spectra you want to save only part of the file
    def __init__(
        self,
        file_name,
        readlen=-1,
        rowstart=None,
        rowend=None,
        channels=None,
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

        super().__init__(file_name, readlen)  # why not pass fixoverflow too??
        self._assign_channels(channels=channels,chanstart=chanstart,chanend=chanend)
        # self.spec_idx2 = self.spec_num - self.spec_num[0]
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
        channels=None,
        chanstart=0,
        chanend=None,
        type="packed",
    ):
        """Create an instance of BasebandFileIterator (BFI).

        An iterator for iterating through an arbitrary number of baseband files passed to it, acclen spectrums (= 1 chunk) at a time. 
        Typically, a BFI instance is associated with a particular antenna, allowing one to obtain a timestream of acclen-long chunks, which can be correlated and averaged. 
        BFI gracefully handles missing spectra and int32 overflows.

        Parameters
        ----------
        file_paths: list of str
            Paths to baseband binary files to be read.
        fileidx: int
            The index of the file in file_paths from which contains idxstart.
        idxstart: int
            The starting spectrum number.
        acclen: int
            Accumulation length, i.e. size of each chunks.
        nchunks: int
            Defaults to None. You need to pass nchunks if you are
            passing the iterator to zip(). Without nchunks, iteration
            will stop once BFI runs out of files.
        chanstart: int
            Index of channel at which to start selection. Default is 0.
        chanend: int or None
            Index of channel at which to end selection. Default is None
            in which case select up to highest frequency channel.
        """
        print("ACCLEN RECEIVED IS", acclen)
        self._OVERFLOW_CTR = 0 #keeps track of overflows encounted in a very long averaging run
        self.acclen = acclen
        self.file_paths = file_paths
        self.fileidx = fileidx
        self.nchunks = nchunks
        self.chunksread = 0
        self.type = type
        self.file_loader = self.get_file_loader() #NB: this is technically not a bound method, but it's OK b/c we don't need self to be passed to file_loader.
        self.obj = self.file_loader(
            file_paths[fileidx], channels=channels, chanstart=chanstart, chanend=chanend, unpack=False
        )
        self.channel_idxs = self.obj.channel_idxs

        self.spec_num_start = idxstart + self.obj.spec_idx[0]
        print(
            "START SPECNUM IS",
            self.spec_num_start,
            "obj start at",
            self.obj.spec_num[0],
        )
        if self.obj.bit_mode == 1 and self.type == "packed": #TODO: update for the new channel array format
            if self.obj.chanstart % 2 > 0:
                raise ValueError("ERROR: Start channel index must be even.")
            self.ncols = np.ceil((self.obj.chanend - self.obj.chanstart) / 4).astype(
                int
            )
        else:
            self.ncols = len(self.channel_idxs)
            # for 4 bits ncols = nchans regardless of packed or float, only dtype changes. 
            # For 1 bit, we want one col for each chan if float.
        # self.pol0 = np.zeros((self.acclen, self.ncols), dtype=self.dtype, order="c")
        # self.pol1 = np.zeros((self.acclen, self.ncols), dtype=self.dtype, order="c")
    def get_file_loader(self):
        if self.type == 'float':
            myclass = BasebandFloat
            self.dtype = 'complex64'
        elif self.type == 'packed':
            myclass = BasebandPacked
            self.dtype = 'uint8'
        def file_loader(*args, **kwargs):
                obj = myclass(*args, **kwargs)
                if self._OVERFLOW_CTR > 0:
                    add_constant_cpu(obj.spec_num, self._OVERFLOW_CTR*2**32) #account for all previous overflows
                if obj._overflowed: 
                    self._OVERFLOW_CTR+=1
                    print("overflow counter is ",self._OVERFLOW_CTR)
                return obj
        return file_loader

    def __iter__(self):
        return self

    def __next__(self):
        t1 = time.time()
        # print(
        #     "Current obj first spec, last spec, and acc spec start",
        #     self.obj.spec_idx[0],
        #     self.obj.spec_idx[-1],
        #     self.spec_num_start,
        # )
        if self.nchunks and self.chunksread == self.nchunks:
            raise StopIteration(f"{self.nchunks} chunks read!")
        pol0 = xp.zeros((self.acclen, self.ncols), dtype=self.dtype, order="c") #this can lead to a mem leak if ref to pol0 lies around in client code
        pol1 = xp.zeros((self.acclen, self.ncols), dtype=self.dtype, order="c")
        specnums = np.array(
            [], dtype="int64"
        )  # len of this array is required to know what % of requested spectrums are present. Could also keep a state variable.
        rem = self.acclen
        i = 0
        while rem:
            # print("-----------------------------------------------------------")
            # print("Rem is", rem)
            if self.spec_num_start < self.obj.spec_num[0]: #wont be triggered for the first file, since we need to start somewhere
                # we are in a gap between the files
                # print("IN A GAP BETWEEN FILES")
                step = min(self.obj.spec_num[0] - self.spec_num_start, rem)
                rem -= step
                # i+=self.acclen-rem
                self.spec_num_start += step
            else:
                # print("calculating ell: ", self.obj.spec_idx[-1], self.obj.spec_idx[0], self.spec_num_start)
                # print("status of overflow", self.obj._overflowed, self._OVERFLOW_CTR)
                l = (
                    self.obj.spec_idx[-1] - self.spec_num_start + 1
                )  # this file spans this many spectra. not all of them may be present, if l > len(spec_idx). files are size limited.
                # print("dist to end is", l, "rem is", rem)
                if rem >= l:
                    # spillover to next file.

                    rowstart, rowend = get_rows_from_specnum(
                        self.spec_num_start, self.spec_num_start + l, self.obj.spec_idx
                    )
                    # print(
                    #     "From if:, rowstart, rowend",
                    #     rowstart,
                    #     rowend,
                    #     rowend - rowstart,
                    # )
                    specnums = np.append(
                        specnums, self.obj.spec_idx[rowstart:rowend]
                    )
                    rem -= l #we've consumed l spectra, whether or not l were present is a different question. nrows <= l
                    (
                        pol0[i : i + rowend - rowstart],
                        pol1[i : i + rowend - rowstart],
                    ) = self.obj._unpack(rowstart, rowend)
                    i += rowend - rowstart
                    self.spec_num_start += l
                    # print("Reading new file")
                    self.fileidx += 1
                    if len(self.file_paths) == self.fileidx:
                        raise StopIteration("BFI Ran out of files!")
                    self.obj = self.file_loader(
                        self.file_paths[self.fileidx],
                        channels=self.channel_idxs,
                        unpack=False,
                    )
                    # print(
                    #     "Current obj first spec, last spec, and acc spec start",
                    #     self.obj.spec_idx[0],
                    #     self.obj.spec_idx[-1],
                    #     self.spec_num_start,
                    # )
                else:
                    rowstart, rowend = get_rows_from_specnum(
                        self.spec_num_start,
                        self.spec_num_start + rem,
                        self.obj.spec_idx,
                    )
                    # print(
                    #     "From else:, rowstart, rowend",
                    #     rowstart,
                    #     rowend,
                    #     rowend - rowstart,
                    # )
                    specnums = np.append(
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
        # data = {"specnums": specnums}
        t2 = time.time()
        # print("TIME TAKEN FOR RETURNING NEW OBJECT",t2-t1)
        return data
