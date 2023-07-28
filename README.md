This README reflects the latest state of the code-base written by Mohan.

This branch (`master`) currently has functionality for dealing with direct data. At the moment all baseband functionality resides in branch `newcode`. I will merge that soon. For a detailed how-to about both direct and baseband data codes, refer to the PDF in "docs" directory.

If you're interested in old code/notes, you can find them under the "legacy" directory (may be hidden in ".legacy").

## Setup

*On Niagara only:* Load modules python 3.8.5 and gcc 9.4.0 with `module load python/3.8.5 gcc/9.4.0`. Note that it is only necessary to load python 3.8.5 if you have not already setup your virtual environment. 

To setup a virtual environment on Niagara:
- ssh into niagara `ssh <username>@niagara.computecanada.ca` (make sure to configure your ssh key access first by logging into your compute-canada account)
- If you haven't already, clone the gh repository `git clone https://github.com/ALBATROS-Experiment/albatros_analysis.git`
- cd into it `cd albatros_analysis`
- Load python 3.8.5 `module load python/3.8.5`
- Make sure it's loaded correctly by running `which python` and `python --version`. Also check pip with `which pip`.
- Create your environment in the /env folder `python -m venv env`
- Activate the environment `source env/bin/activate`
- Install all the packages `pip install -r requirements_py385.txt`

Before running analysis code, you must **build c-libraries** that get called by python correlation scripts. To do this run `python ./correlations/setup_cpu.py`. Before running tests, you must also build the mars2019 testing c-lib (see tests section below for further instruction). 

For everyday use in niagara, the flow should go something like this
- `source env/bin/activate`
- `module load gcc/9.4.0`

To use a compute node `debugjob --clean 1` will give you access for 1 hour. 

## Data Types

There are two types of data:

1) Directly computed auto- and cross spectra from two inputs of a SNAP box.  These data products exist in directories named "data_auto_cross"
   The data are chunked in regular hour-long intervals. The number of rows in a direct data file depends on the accumulation length (acclen). The default acclen has been 393216 for quite some time, and corresponds to 6.44 s. Thus, there are roughly 560 rows (may vary by +/- 1 sometimes) in the direct files.
   There are 5-digit directories corresponding to each day (ctime),
   and 10-digit ctime subdirectories within each.  Within each
   subdirectory, the files are:
   - acc_cnt[12].raw : FFT accumulation counter
   - fpga_temp.raw : FPGA temperature
   - pfb_fft_of[12].raw : PFB FFT overflow counter
   - pi_temp.raw : Raspberry Pi temperature
   - pol00.scio, pol11.scio : autospectra of inputs 0 and 1 (each a 2D array)
   - pol01r.scio, pol01i.scio : real and imaginary parts of cross spectra (each a 2D array)
   - sync_cnt[12].raw : some FPGA-related counter, I forget the details
   - sys_clkcounter[12].raw : some FPGA-related counter, I forget the details
   - time_gps_start.raw, time_gps_stop.raw : GPS time_start is when reading a row of accumulated data (one row of a direct file) from FPGA registers started, time_stop is when reading finished. Therefore, stop-start is not the accumulation time. If you're interested in inferring accumulation time from timestamps, look at mean of diff of either start or stop time. The plotting scripts have the correct implementation. 

2) Baseband data: details to follow


## Code that's included here:

* plot_overnight_new.py : Plot directly computed auto- and
  cross-spectra for a given interval of time (in UTC). The code will automatically find all directories within that time period, concatenate them, average them as per your input, and plot them. Very useful if you want to visualize several hours/days/months of data in a single summary plot. Supports `--help`.

* quick_spectra.py : Plot directly computed auto- and
cross-spectra for a particular direct spectra folder (10-digit) for quick visualization and sanity checks. Supports `--help`.

* SNAPfiletools.py : Miscellaneous functions for handling data and
  timestamps

* timestamp_loginfo.py : Troll the log files to find system
  information for a particular time stamp.

* utc_ls.py : Poorly written script for converting ctimes into
  human-readable timestamps.  The globbing is fragile and sometimes
  needs coaxing in order to work.

## Tests

Before running tests, make sure to build dynamically linked c library used in testing. 

To run tests execute `pytest -rP correaltions/tests` from the project root directory.

## Build the docs

Then, enter `pdoc --html ./` from the project's root directory. To overwrite current documentation already in /html, append the `--force` flag. Pdoc3 should already be installed if you are using a virtual environment (which is advisable), but in case you need to troubleshoort: first make sure that you have pdoc3 installed (not pdoc, which is no longer maintained) with `pip uninstall pdoc && pip install pdoc`. 





