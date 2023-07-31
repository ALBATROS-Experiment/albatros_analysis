*Before you clone this repository, read the LFS section*

This README reflects the latest state of the code base written by Mohan.

This branch (`master`) currently has functionality for dealing with direct data. At the moment, all baseband functionality resides in branch `newcode`. I will merge that soon. For a detailed how-to about both direct and baseband data codes, refer to the PDF in the "docs" directory.

If you're interested in old code/notes, you can find them under the hidden ".legacy" directory.

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

Before running tests, make sure to build the dynamically linked c-library used in testing `python correlations/tests/mars2019/build_albatrostools.py`

To run tests execute `pytest -rP correaltions/tests` from the project root directory. If they do not pass it may be because you haven't fetched the data files used in the tests. 

## Setup

**(1)** Python environment. **(2)** Gnu compiler collection (GCC). **(3)** Build c-libs.

*On Niagara only:* Load modules python 3.8.5 and gcc 9.4.0 with `module load python/3.8.5 gcc/9.4.0`. Note that it is only necessary to load Python 3.8.5 if you have not already set up your virtual environment. 

To set up a virtual environment in Niagara:
- ssh into Niagara `ssh <username>@niagara.computecanada.ca` (make sure to configure your ssh key access first by logging into your compute-canada account)
- If you haven't already, clone the gh repository `git clone https://github.com/ALBATROS-Experiment/albatros_analysis.git`
- cd into it `cd albatros_analysis`
- Load python 3.8.5 `module load python/3.8.5`
- Make sure it's loaded correctly by running `which python` and `python --version`. (Also check pip with `which pip && pip --version`.
- Create your environment in the /env folder `python -m venv env`
- Activate the environment `source env/bin/activate`
- Install all the packages `pip install -r requirements_py385.txt`

Before running analysis code, you must **build c-libraries** that get called by Python correlation scripts. To do this, run `python ./correlations/setup_cpu.py`. (Before running tests, you must also build the mars2019 testing c-lib; see the "tests" section below for further instructions.) 

For everyday use in Niagara, the flow should go something like this
- `source env/bin/activate`
- `module load gcc/9.4.0`

Enter `debugjob --clean 1` to gain access to a compute node for one hour. 




## Large File Storage (LFS)

This repository uses GitHub LFS. To control how the large files are dealt with, you need `git-lfs`. 

Once the binary is installed and included in your path, you need to configure it with `git lfs install --skip-smudge`. 

We *strongly recommend* passing the `--skip-smudge` flag, otherwise you will experience unnecessary delays on the order of minutes (depending on your specs & download speeds) when switching between branches. The `--skip-smudge` flag configures LFS to *not* download large files when cloning and switching branches. Large data files (~500 Mb) are only used for testing code. 

If & when you want to download large, you can do so individually with `git lfs pull --include="<filename>"`.

<img width="716" alt="Screenshot 2023-07-28 at 10 07 50 PM" src="https://github.com/ALBATROS-Experiment/albatros_analysis/assets/21654151/315066f5-3d17-43af-91ef-260941c8864c">


Resources:
- [https://sabicalija.github.io/git-lfs-intro/](https://sabicalija.github.io/git-lfs-intro/)


### Installing git lfs

If you have root access, you can install it with a standard package manager, for example
- Mac: `brew install git-lfs`
- Linux: `sudo apt install git-lfs`

If you are on Niagara (Cent OS 7), we have compiled a binary of git-lfs that you can simply download. 

If you do not have root access, here's what I did to get it working:
- Download correct binary tarball from [pkgs.org](https://pkgs.org) (the one fore CENTOS 7) with `wget https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/g/git-lfs-2.10.0-2.el7.x86_64.rpm -O git-lfs-2.10.0-2.el7.x86_64.rpm`
- Turn it into a cpio file with  `rpm2cpio git-lfs-2.10.0-2.el7.x86_64.rpm git-lfs-2.10.0-2.el7.x86_64.cpio`
- Extract cpio with `cpio -idv < git-lfs-2.10.0-2.el7.x86_64.cpio`
- Copy binary (in bin/ subdirectory of the extracted folder) to where you think is a sensible install location, e.g. `cp bin/git-lfs ~/local/bin`
- Make sure that the install location is permanently in your path by adding this line to your .bashrc `export PATH="~/local/bin:$PATH"`
Resources:
- [https://unix.stackexchange.com/questions/61283/yum-install-in-user-home-for-non-admins](https://unix.stackexchange.com/questions/61283/yum-install-in-user-home-for-non-admins)
- [https://rhel.pkgs.org/7/epel-x86_64/git-lfs-2.10.0-2.el7.x86_64.rpm.html](https://rhel.pkgs.org/7/epel-x86_64/git-lfs-2.10.0-2.el7.x86_64.rpm.html)





## Build the docs

Before building docs, make sure c-libs are properly compiled (`cd corrlations && python setup_cpu.py`) otherwise not all pages will build.

Navigate to the project root's parent directory to build. This is the script I use:

```sh

#!/opt/homebrew/bin/bash

# [Steve] This is what I use to build the docs. 
# Place this file in the parent to the project root and run it from there with bash. 

# uncomment for dev, builds and serves the docs
pdoc --docformat="numpy" --logo="https://upload.wikimedia.org/wikipedia/commons/7/73/Short_tailed_Albatross1.jpg" --math --mermaid albatros_analysis;

# uncomment for prod, builds docs and output to html
#pdoc --docformat="numpy" --logo="https://upload.wikimedia.org/wikipedia/commons/7/73/Short_tailed_Albatross1.jpg" --math --mermaid albatros_analysis -o albatros_analysis/docs;
```

Reffer to the [pdoc api](https://pdoc.dev/docs/pdoc.html).


