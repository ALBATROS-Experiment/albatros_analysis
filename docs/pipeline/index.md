To begin, we need to establish some basic knowledge about the project. There are several design choices that require some context to explain and justify, making a bare understanding of the subject matter indispensable.

The objective of the pipeline is to align the antenna timestreams (with respect to each other) to nanosecond level precision. This matters a lot in inteferometry, since information lives in the relative signal delays and phases. ALBATROS antenna have clocks which are not synchronized to each other, so we have order 1 second offsets (huge). All the data is saved to hard-drive (it's the arctic) so all analysis has to be done off-site and off-line. The antenna also see the whole sky, meaning that we can't just look at a bright source for calibration (annoying). Moreover, for various reasons Mohan told me but I can't quite remember, we can't use an artificial beacon (also annoying). Therefore, we need to use satellites that pass overhead as our point of reference to get our timing solution.

Our antenna ADC samples at 250 MSPS, each antenna channelizes (using a PFB) the incoming timestream into 2048 channels (4096-point FT step). Thus our baseband data has spectra of channel width ~61 kHz and period ~16 microsecs. It is also 1-bit quantized. The natural proxy for time in our system is baseband spectra, i.e. the smallest timestep iteration.

The general flow of the pipeline, with rather self-explanatory script names and data storage files, is best outlined in the following flowchart:

![alt text](../images/flow.jpeg)

Note that the arrows only point towards the right, meaning data that is generated in the script then gets stored in that file. Each script uses almost all previous data (e.g. practically all scripts use antenna coordinates, which are stored in the config files). 

Now we can go through each of the main steps, numbered accordingly in the flowchart.