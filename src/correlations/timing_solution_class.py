import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

def get_all_batches(root_dir):
    """
    Gives start and end times of all batches for which there are timing solutions.

    Parameters
    ----------
    root_dir : str
        Directory to the data

    Returns
    -------
    list
        List of tuples (start, end) for all the batches
    """
    root = Path(root_dir)
    index_file = root / "index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"Could not find {index_file}")

    with open(index_file, "r") as f:
        index = json.load(f)
    
    batches = []
    for batch in index:
        batches.append((batch['start'], batch['end']))
    return batches


class TimingSolution:
    """
    Interface for accessing antenna timing solutions.

    The class loads a database of calibration batches and selects the
    batch containing a requested observation start time. Timing solutions
    and associated metadata for the selected batch can then be accessed
    through the class methods. Timing solutions can then also be interpolated
    as desired for data analysis.

    Parameters
    ----------
    query_unix_start : float or int
        Unix timestamp (seconds) used to identify the calibration batch
        containing the desired observation.
    root : str or pathlib.Path
        Path to the directory containing the timing solution database.
        This directory must contain an ``index.json`` file and the HDF5
        files referenced by the index.

    Attributes
    ----------
    root : pathlib.Path
        Root directory of the timing solution database.
    batch : dict
        Dictionary describing the selected calibration batch.
    batch_name : str
        Name of the selected batch, in form 'batch_{batch_start_unix}'.
    batch_start_unix : float
        Start time of the selected batch in Unix seconds.
    batch_end_unix : float
        End time of the selected batch in Unix seconds.
    ref_ant : str
        Name of the reference antenna for the selected batch.
    non_ref_ants : list of str
        Names of the non-reference antennas whose delays are stored in the
        timing solutions.
    UTC_per_spec : float
        Conversion rate from spectrum number to absolute UTC time
    UTC_offset : float
        Initial UTC time corresponding to spectrum number zero.

    Raises
    ------
    FileNotFoundError
        If the timing solution database or its ``index.json`` file cannot
        be found.
    ValueError
        If ``query_unix_start`` is not contained within any calibration
        batch, or if it is contained within more than one batch.

    Notes
    -----
    Timing solutions are organized into independent, non-overlapping
    batches. The class selects the unique batch whose time range contains
    ``query_unix_start``. 
    Moreover, spectra and absolute UTC are interchangable via UTC_per_spec * s + UTC_offset.
    """

    def __init__(self, query_unix_start, root):

        # root path where all timing solutions stored
        self.root = Path(root)

        # set up batch
        self.batch = self._find_batch(query_unix_start)

        # set up metadata
        self.batch_name = self.batch['batch']
        self.batch_start_unix = self.batch['start']
        self.batch_end_unix = self.batch['end']
        self.ref_ant = self.batch['ref_ant']
        self.non_ref_ants = self.batch['non_ref_ants']
        self.UTC_per_spec = self.batch["UTC_per_spec"]
        self.UTC_offset = self.batch["UTC_offset"]

        # load up timing solution
        self._load_data()

    def _find_batch(self, query_unix_start):
        # set index file name
        index_file = self.root / "index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Could not find {index_file}")

        #set up index of all batches
        with open(index_file, "r") as f:
            index = json.load(f)
            
        # figure out what batch it's asking for, set to property
        matches = []

        for batch in index:
            contained = (query_unix_start >= batch["start"] and query_unix_start <= batch["end"])
            if contained:
                matches.append(batch)

        if len(matches) == 0:
            raise ValueError("Your queried unix start time does not lie in any batches with timing solutions.")
        if len(matches) > 1:
            raise ValueError("Your queried unix start time somehow lies within multiple batches. Check the timing solutions, something is stored wrong (there should be no overlap).")
        return matches[0]


    def _load_data(self):

        filename = (self.root / f"{self.batch_name}.h5")

        if not filename.exists():
            raise FileNotFoundError(f'Could not find {filename}')

        with h5py.File(filename, "r") as f:
            self.taus = f["taus"][:]
            self.spectra = f["spectra"][:]
            self.unix = self.spectra * self.UTC_per_spec + self.UTC_offset


    def spectra_to_unix(self, spectra):
        """
        Converts any desired reference antenna spectrum indices to UNIX timestamps.

        Parameters
        ----------
        spectra : np.ndarray
            Array of reference antenna spectra.

        Returns
        -------
        np.ndarray
            Unix timesamps corresponding to input spectra.
        """
        return spectra * self.UTC_per_spec + self.UTC_offset


    def interpolate_delay(self, interp_unix_start, interp_unix_end, dt, extrapolate = True, method="linear" ):
        """
        Interpolate antenna delay timing solutions for entire requested Unix interval.

        The timing solutions loaded for the selected batch are
        interpolated onto a user-defined time grid. Each antenna delay solution
        is interpolated independently using the specified interpolation method.

        Parameters
        ----------
        interp_unix_start : float
            Starting Unix timestamp for the requested interpolation range.

        interp_unix_end : float
            Ending Unix timestamp for the requested interpolation range.

        dt : float
            Spacing between requested output timestamps in seconds.

        extrapolate : bool, optional
            Whether to allow interpolation outside the time range covered by
            the loaded timing solution. If False, a ValueError is raised when
            the requested range extends beyond the available data.
            Default is True.

        method : str, optional
            Interpolation method to use. Currently only ``"linear"`` is
            supported.
            Default is ``"linear"``.

        Returns
        -------
        unix_interp : np.ndarray
            Requested Unix timestamps at which delay solutions have been
            interpolated.

        taus_interp : np.ndarray
            Interpolated antenna delay solutions.
            Shape is ``(nant - 1, ntime)``, where ``nant - 1`` is the number
            of non-reference antennas and ``ntime`` is the number of requested
            output timestamps.

        Raises
        ------
        ValueError
            If extrapolation is disabled and the requested interpolation range
            lies outside the available timing solution range.

        NotImplementedError
            If an unsupported interpolation method is requested.

        Notes
        -----
        Delay solutions are stored internally as ``self.taus`` and their
        corresponding Unix timestamps as ``self.unix``. Linear interpolation is
        performed independently for each antenna delay stream.
        """

        #check the interpolation times fall inside the batch
        if (interp_unix_start < self.batch_start_unix) or (interp_unix_end > self.batch_end_unix):
            raise ValueError("Interpolation time interval falls outside batch range")

        # generate requested timestamps
        unix_interp = np.arange(interp_unix_start, interp_unix_end, dt)
        unix_data = self.unix 

        # set taus as array
        taus = self.taus

        # check bounds
        if extrapolate == False:
            if unix_interp[0] < unix_data[0] or unix_interp[-1] > unix_data[-1]:
                raise ValueError("You have extrapolation turned off, and requested times outside batch's satellite data range.")

        if method == "linear":

            taus_interp = np.zeros((taus.shape[0], len(unix_interp)))
            print('number of baselines (containing ref ant) in tau data:', taus.shape[0])
            print('number of desired interpolation unix times:', len(unix_interp))
            print('number of data unix times', unix_data.shape)

            # Interpolate each baseline independently
            for bl in range(taus.shape[0]):
                taus_interp[bl, :] = np.interp(unix_interp, unix_data, taus[bl,:])

            return unix_interp, taus_interp

        else:
            raise NotImplementedError(f"Unknown interpolation method {method}")


    
    def interpolate_delay2(self, unix_interp, extrapolate = True, break_batch = False, method="linear" ):
        """
        Interpolate antenna delay timing solutions for a requested Unix time array.

        The timing solutions loaded for the selected batch are
        interpolated onto a user-defined time array. Each antenna delay solution
        is interpolated independently using the specified interpolation method.

        Parameters
        ----------
        unix_interp : np.ndarray
            Unix timestamp array. Must be increasing.

        extrapolate : bool, optional
            Whether to allow interpolation outside the time range covered by
            the loaded timing solution. If False, a ValueError is raised when
            the requested range extends beyond the available data.
            Default is True.

        method : str, optional
            Interpolation method to use. Currently only ``"linear"`` is
            supported.
            Default is ``"linear"``.

        Returns
        -------
    
        taus_interp : np.ndarray
            Interpolated antenna delay solutions.
            Shape is ``(nant - 1, ntime)``, where ``nant - 1`` is the number
            of non-reference antennas and ``ntime`` is the number of requested
            output timestamps.

        Raises
        ------
        ValueError
            If extrapolation is disabled and the requested interpolation range
            lies outside the available timing solution range.

        NotImplementedError
            If an unsupported interpolation method is requested.

        Notes
        -----
        Delay solutions are stored internally as ``self.taus`` and their
        corresponding Unix timestamps as ``self.unix``. Linear interpolation is
        performed independently for each antenna delay stream.
        """ 
        
        #check that the queried unix timestamps are increasing
        if np.any(np.diff(unix_interp) < 0):
            raise ValueError("unix_interp must be increasing.")

        #check the interpolation times fall inside the batch
        if not break_batch:
            if (unix_interp[0] < self.batch_start_unix) or (unix_interp[-1] > self.batch_end_unix):
                raise ValueError("Interpolation time interval falls outside batch range")

        # load unix timestamps that have data
        unix_data = self.unix 

        # set taus as array
        taus = self.taus

        # check bounds
        if extrapolate == False:
            if unix_interp[0] < unix_data[0] or unix_interp[-1] > unix_data[-1]:
                raise ValueError("You have extrapolation turned off, and requested times outside batch's satellite data range.")

        if method == "linear":

            taus_interp = np.zeros((taus.shape[0], len(unix_interp)))
            print('number of baselines (containing ref ant) in tau data:', taus.shape[0])
            print('number of desired interpolation unix times:', len(unix_interp))
            print('number of total data unix times', unix_data.shape)

            # Interpolate each baseline independently
            for bl in range(taus.shape[0]):
                taus_interp[bl, :] = np.interp(unix_interp, unix_data, taus[bl,:])

            return taus_interp

        else:
            raise NotImplementedError(f"Unknown interpolation method {method}")


    def interpolate_existing_delays(interp_unix_start, interp_unix_end, dt, border_window):
        """
        Interpolate timing solutions only around periods with existing data.

        Finds all timing solution samples within the requested interval,
        extends each contiguous data region by ``border_window`` seconds,
        and interpolates delays onto a regular time grid. Regions with no
        timing solution data are not included.

        Parameters
        ----------
        interp_start : float
            Start Unix timestamp of requested range.

        interp_end : float
            End Unix timestamp of requested range.

        dt : float
            Output time resolution in seconds.

        border_window : float
            Time padding applied around the edges of existing data regions.

        Returns
        -------
        unix_interp : np.ndarray
            Unix timestamps where interpolated delays are available.

        taus_interp : np.ndarray
            Interpolated delay solutions corresponding to ``unix_interp``.
        """

        # data samples inside requested interval
        unix_mask = (self.unix >= interp_unix_start) & (self.unix <= interp_unix_end)

        if not np.any(unix_mask):
            raise ValueError("No timing solution data exists within requested interval.")

        unix_existing = self.unix[unix_mask]

        # find gaps between existing data regions
        gaps = np.where(np.diff(unix_existing) > 1.5 * dt)[0]

        # Split into continuous regions
        pass_starts = np.concatenate(([0], gaps + 1))
        pass_ends = np.concatenate((gaps, [len(unix_existing) - 1]))

        unix_all = []
        taus_all = []

        # Iterate over all passes
        for start_idx, end_idx in zip(pass_starts, pass_ends):

            # Actual data limits for this pass
            pass_start = pass_existing[start_idx]
            pass_end = pass_existing[end_idx]

            # Expand by border window (careful to not go outisde requested times)
            interp_pass_start = max(interp_unix_start, pass_start - border_window)
            interp_pass_end = min(interp_end, pass_end + border_window)

            # Interpolate this region using existing method
            unix_i, taus_i = self.interpolate_delay(interp_pass_start, interp_pass_end, dt)

            unix_all.append(unix_i)
            taus_all.append(taus_i)

        # Concatenate separate passes
        unix_interp = np.concatenate(unix_all)
        taus_interp = np.concatenate(taus_all, axis=1)

        return unix_interp, taus_interp


    def get_data_mask(self, query_unix, tolerance):
        """
        Return mask indicating where timing solution samples exist nearby.

        Parameters
        ----------
        query_unix : np.ndarray
            Requested Unix timestamps.

        tolerance : float
            Maximum allowed distance (seconds) between a requested time and
            an available timing solution sample.

        Returns
        -------
        np.ndarray
            Boolean mask. True where a timing solution sample exists within
            the tolerance window.
        """

        idx = np.searchsorted(self.unix, query_unix)

        # clip indices at boundaries
        idx = np.clip(idx, 1, len(self.unix)-1)

        # distance to nearest available sample
        dt = np.minimum(
            np.abs(query_unix - self.unix[idx]),
            np.abs(query_unix - self.unix[idx-1])
        )

        return dt <= tolerance


    def all_blines(self, taus):
        """
        For taus that are only with respect to the reference, creates all baseline combinations.
        By convention the reference antenna has zero delay.

        Parameters
        ----------
        taus : np.ndarray
            All delays, shape (nant-1, ntimes)

        Returns
        -------
        np.ndarray
            All delays for all baselines, shape (nbl, ntimes)

        Notes
        -----
        Beware of convention: ref-nref, and small-large.
        Means that when subtracting, if A<B,  have delayA - delayB = (ref - tauA) - (ref - tauB) = tauB - tauA.
        Therefore, counterintuitively, you subtract smaller index from larger. 
        """
        nant = taus.shape[0] + 1
        ntimes = taus.shape[1]
        nbl = nant*(nant-1)//2

        ant_names = [self.ref_ant] + self.non_ref_ants
        bline_names = []

        taus2 = np.vstack([np.zeros(ntimes), taus])
        taus_all = np.zeros((nbl, ntimes))

        blid = 0
        for i in range(nant):
            taus_i = taus2[i,:]
            for j in range(i+1 ,nant):
                taus_j = taus2[j,:]
                taus_all[blid, :] = taus_j - taus_i
                bline_names.append(ant_names[i] + '-' + ant_names[j])
                blid +=1

        return taus_all, bline_names