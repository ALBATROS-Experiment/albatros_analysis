# 1. Data Structure

## 1.1 Baseband

Data is divided into `.raw` folders of about 50 seconds each, with the rounded starting UTC timestamp of the file as its name. The files contain all the raw data, but also each spectrum's absolute spectrum number in the overall timestream. The main tool we use to parse such antenna timestreams is the script `baseband_data_classes` (see the relevant [tutorial](/tutorials/BDC/) and [API](/api/correlations/)). Data timestamping is very inaccurate, so when we query the system for data at specific times, it returns a starting spectrum which only loosely corresponds to the actual time at which the data was recorded (order 1 second error). Therefore, there is a discrepancy between system query and actual data measurement time.

Not all frequencies are present in each file; we may make measurements at different frequencies for different days. Moreover, the data collection can be configured to be in either 1-bit or 4-bit, meaning files can change data types as well. The script `baseband_data_classes` lets you check frequencies, data type, and various other metadata corresponding to the raw data.

## 1.2 Batches

When making observations that span more than a single specific file of data, we may want to know the regions where we have data and to check if there are any discontinuities within it. By convention, we call continuous periods of full data coverage (for all antenna we wish to look at) a **batch**. For example, if MARS3 is not in operation and we do not consider it, a batch is a period of time (say of 20 hours) where all other antenna contain uninterrupted data. Each antenna reboots once per day around the same time, so a batch is at very most 24 hours. If there are any other interruptions, it will be shorter. Batches therefore serve as the natural unit of long-term data collection. See the associated [tutorial](/tutorials/batches/) to see how we determine batches.

## 1.3 Configuration Files

Once we determine the batches we wish to analyse, we create a configuration file for each of them, more informally known as **config files**. All relevant information about the batch is found within this `.json` file, and we assume this information is self-consistent within each batch. For example, it contains the batch's start and end times (in UTC), which antennas are looked at, the antenna coordinates, and the antenna data directories. It also specifies how the later analysis should be done, containing the satellite detection coarse accumulation lenth, the visibility accumulation length, and the oversampling factor for fine timing. For more information on config files, alongside information on each parameter it contains, see the relevant [tutorial](/tutorials/config/). 

## 1.4 Alignment Conventions

When aligning antenna timestreams, we select a single antenna as a reference antenna to which we shift all the others. The spectra of the reference antenna will not be changed, instead all the others are aligned with it. By default, and due to its central geographical position and ease of indexing, we pick MARS1 (data permitting), or as second choice MARS2. By convention, all alignment is done as **ref** - **nonref**. Therefore, if the offset is positive, it means the reference spectrum number is ahead of the nonreference spectrum number. 

As an example, let us open two antenna timestreams for some fixed system query, say 11 am. Consider $s_y$ as the opened spectrum number for the reference, $s_x$ as the opened spectrum number for the non-reference, and $Δx$ as the total TRUE spectrum offset between the two timestreams. The difference between $s_y$ and $s_x$ at this point will not correspond to the true spectrum offset. Instead, there will exist some required spectrum shift, $Δs_r$, which must be applied to the non-reference antenna such that it becomes aligned. We can write:
$$
Δx = s_y - s_x + Δs_r
$$
Beware of the sign of $Δs_r$, might be the opposite in the actual code. A visualization of this alignment can be seen in the image below:

![alt text](../images/spectrum_alignment.png)

When opening up two files, the initial difference between the two, $s_x - s_y$, is easily determined. What we need the satellites for is finding the required RELATIVE shift $Δs_r$ to alter the net difference and obtain alignment. Note that we have not yet determined the absolute time at which we measure the data: absolute UTC timing treatment comes in [later](/pipeline/utc/).
