import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

class TimingSolution:
    """
    Interface for retrieving antenna delays from calibration batches.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing calibration batches and index.json.

    Notes
    -----
    The database is organized into independent calibration batches.
    A requested time range must lie entirely inside one batch.
    """

    def __init__(self, root):

        self.root = Path(root)

        index_file = self.root / "index.json"

        if not index_file.exists():
            raise FileNotFoundError(
                f"Could not find {index_file}"
            )

        with open(index_file, "r") as f:
            self.index = json.load(f)


    def find_batch(self, start, end):
        """
        Find calibration batch containing requested time range.

        Parameters
        ----------
        start : float
            Starting Unix timestamp.

        end : float
            Ending Unix timestamp.

        satellite : str, optional
            Satellite name.

        Returns
        -------
        dict
            Batch metadata.

        Raises
        ------
        ValueError
            If requested interval spans multiple batches
            or no batch exists.
        """

        matches = []

        for batch in self.index:

            contained = (start >= batch["start"] and end <= batch["end"])

            if contained:
                matches.append(batch)

        if len(matches) == 0:
            raise ValueError("You are asking for times that are not contained in a single batch.")

        if len(matches) > 1:
            raise ValueError("Your requested times somehow completely fit into two or more different batches, check your data there must be an error.")

        return matches[0]

    def find_ref_ant(self, start, end):
        batch = self.find_batch(start, end)
        return batch['ref_ant']

    def find_other_ants(self, start, end):
        batch = self.find_batch(start, end)
        return batch['non_ref_ants']


    def load_batch(self, batch_name):
        """
        Load delay data from HDF5 file.

        Parameters
        ----------
        batch_name : str
            Batch identifier.

        Returns
        -------
        dict
            Dictionary containing spectra,
            unix_time, and delays.
        """

        filename = (self.root / f"{batch_name}.h5")

        if not filename.exists():
            raise FileNotFoundError(filename)

        with h5py.File(filename, "r") as f:
            data = {
                "spectra": f["spectra"][:], 
                "taus": f["taus"][:],
                }

        return data



    def spectra_to_unix(self, batch, data):
        """
        Convert spectrum indices to UNIX timestamps.

        Parameters
        ----------
        batch_name
            Name of batch we want to look at
        data
            Data of batch we want, which contains its spectra

        Returns
        -------
        np.ndarray
            Fractional spectrum positions.
        """

        UTC_per_spec = batch["UTC_per_spec"]
        UTC_offset = batch["UTC_offset"]

        spectra = data["spectra"]
        unix = data["spectra"]*UTC_per_spec + UTC_offset

        return unix


    def interpolate_delay(self, query_unix_start, query_unix_end, time_resolution, data, unix_data, extrapolate = True, method="linear" ):
        """
        Interpolate delays at requested Unix timestamps.

        Parameters
        ----------
        query_unix_start : float
            Starting Unix timestamp.

        query_unix_end : float
            Ending Unix timestamp.

        time_resolution : float
            Output spacing in seconds.

        data : dict
            Dictionary containing delay solutions.
            Requires:
                data["taus"] : ndarray, shape (ntimes, nblines)

        unix : np.ndarray
            Unix timestamps corresponding to the delay solutions.
            Shape (ntimes,)

        method : str
            Interpolation method.

        Returns
        -------
        query_unix : np.ndarray
            Requested Unix timestamps.

        taus_interp : np.ndarray
            Interpolated delays.
            Shape (nblines, nrequested_times)
        """

        taus = data["taus"]

        # Generate requested timestamps
        query_unix = np.arange(query_unix_start, query_unix_end, time_resolution)

        # Check bounds
        if extrapolate == False:
            if query_unix[0] < unix_data[0] or query_unix[-1] > unix_data[-1]:
                raise ValueError("Requested times outside batch range. Don't want to extrapolate.")

        if method == "linear":

            taus_interp = np.zeros((taus.shape[0], len(query_unix)))
            print('number of baselines in tau data:', taus.shape[0])
            print('number of queried unix times:', len(query_unix))
            print('number of data unix times', unix_data.shape)

            # Interpolate each baseline independently
            for bl in range(taus.shape[0]):
                taus_interp[bl, :] = np.interp(query_unix, unix_data, taus[bl,:])

            return query_unix, taus_interp

        else:
            raise NotImplementedError(f"Unknown interpolation method {method}")



    def find_timing_sol(
        self,
        start,
        end,
        time_resolution,
        extrapolate = True,
        method="linear"
    ):
        """
        Retrieve delays over a requested time interval.

        Parameters
        ----------
        start : float
            Starting Unix timestamp.

        end : float
            Ending Unix timestamp.

        time_resolution : float
            Requested output spacing in seconds.

        extrapolate: bool, optional
            Determines whether you can take data that lies before or after all sat passes.
            Basically, are you allowed to guess the evolution when you don't know what happens before or after.

        method : str
            Delay interpolation method.

        Returns
        -------

        delays : np.ndarray
            Interpolated antenna delays.
        """

        # identify correct batch. make sure only lives in single batch
        batch = self.find_batch(start, end)
        batch_name = batch["batch"]
        print('Your data lives in', batch_name)

        # get the data for this batch
        data = self.load_batch(batch_name)

        #get the unix times at which we have data
        unix_times_data = self.spectra_to_unix(batch, data)

        # interpolate the delay
        unix_times_query, taus = self.interpolate_delay(start, end, time_resolution, data, unix_times_data, extrapolate = extrapolate, method=method)

        return unix_times_query, taus