The aim of this section is to provide a clear, step by step explanation of each component of the pipeline.

Our antenna ADC samples at 250 MSPS, each antenna channelizes (using a PFB) the incoming timestream into 2048 channels (4096-point FT step). Thus our baseband data has spectra of channel width ~61 kHz and period ~16 microsecs. It is also 1-bit quantized. The natural proxy for time in our system is baseband spectra, i.e. the smallest timestep iteration.

The general flow of the pipeline, with rather self-explanatory script names and data storage files, is best outlined in the following flowchart:

![alt text](../images/flow.jpeg)

Note that the arrows only point towards the right, meaning data that is generated in the script then gets stored in that file. Each script uses almost all previous data (e.g. practically all scripts use antenna coordinates, which are stored in the config files). 

Now we can go through each of the main steps, numbered accordingly in the flowchart.