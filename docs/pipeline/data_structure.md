# 1. Data Structure

## 1.1 Batches

The first thing to check is where we have data, and to verify if there are any discontinuities within it. For that, there is a notebook called 'read_batches', which mainly uses two functions from 'baseband_utils', called 'get_present_files' and 'get_simul_files'. Given some inputs of start and end times, alongside antenna paths, these will give you and plot for you all the contiguous 'batches' of data where each antenna has data present. The notebook also has the nice feature of auto-generating config files for each desired batch, alongside job files for future analysis.

Config files serve as the sort of 'home base' for each iteration of data. Each antenna re-starts every 24 hours or so, meaning batches are the natural day-long ish chunk of data to analyze. Moreover, during their re-start, each antenna timestream resets its spectrum number count alignment, which causes the antenna alignments to change. Therefore, it's important we only work within self-consistent 'batches'. If there are any corruptions or accidental restarts during the day, it is natural to simply separate the day into two, which is why we use the term 'batch', as it does not have to coincide with an exact day. Config files contain all the main information that is needed to get started with our analysis, such as antenna coordinates, data paths, names, etc. They also contain the start and end times for each batch, marking the interval where we know data is present and consistent.

## 1.2 Baseband

The data is divided into `.raw` folders of about 50 seconds each, with the rounded starting UTC timestamp of the file as its name. The files contain all the raw data, but also each spectrum's absolute spectrum number in the overall timestream. The main tool we use to parse such antenna timestreams is 'baseband_data_classes.py'. The data is initialized as a so-called Baseband_File_Iterator object, which, in accordance with its name, handily allows us to parse data continuously across files. When you open a timestream with a certain starting time, the Baseband-File-Iterator object, due to the poor UTC timestamping of the file data, will return a certain starting spectrum which very loosely corresponds to the actual time at which the data was recorded. Therefore, there is a discrepancy between system query and actual data measurement time.

We select a single antenna as a reference antenna to which we align all the other ones. The spectra of the reference antenna will not be shifted, instead all the others are aligned with it. By default, and due to their central geographical position, we pick MARS1 (data permitting), or as second choice MARS2. By convention, all alignment is done as ref - nonref. So if the offset is positive, it means the reference spectrum number is ahead of the nonreference spectrum number. 

As an example, let us open two antenna timestreams for some fixed system query, say 11 am. Consider $s_y$ as the opened spectrum number for the reference, $s_x$ as the opened spectrum number for the non-reference, and $Δx$ as the total TRUE spectrum offset between the two timestreams. The difference between $s_y$ and $s_x$ at this point will not correspond to the true spectrum offset. Instead, there will exist some required spectrum shift, $Δs_r$, which must be applied to the non-reference antenna such that it becomes aligned. We can write:
$$
Δx = s_y - s_x + Δs_r
$$
Beware of the sign of $Δs_r$, might be the opposite in the actual code. A visualization of this alignment can be seen in the image below:

![alt text](../images/specnumoffsets.jpeg)

When opening up two files, the initial difference between the two, $s_x - s_y$, is easily determined. What we need the satellites for is finding the required RELATIVE shift $Δs_r$ to alter the net difference and obtain alignment. Note that we have not yet determined the absolute time at which we measure the data. This will come in later. 
