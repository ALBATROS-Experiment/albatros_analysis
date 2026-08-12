# 2 Satellite Detection

There are two primary objectives to satellite detection (known as satdet): to determine when each antenna sees clear satellite signal, and to determine the best spectrum alignment for each antenna. We select a reference antenna (by convention MARS1) and consider all baselines containing it. This makes relative alignment significantly simpler. The script used to generate data in satellite detection is `get_satdet.py`, while `get_consensus_offsets.py` and `get_pulses.py` extract and format usable information from this data.

## 2.1 Detection Process

Using satellite trajectories overhead, we predict when one of them passes overhead and becomes visible to the antenna (orbcomm_utils.get_risen_sats). For each baseline containing the reference antenna, we perform a coarse cross-correlation. If the SNR is high enough, we call a satellite 'pass' a 'pulse', and it counts as a detection. For each such detection, we record the spectrum number offset of the peak of the cxcorr (which is a function of delay, as you may recall).

## 2.2 Finding Spectrum Number Offsets

This part addresses the first of the main objectives of satdet: determining spectrum alignment.
There are plenty of pulses, some of which yield different values of 'specnum offset'?
We must determine a consensus. The relevant script is 'get_consensus_offset.py'

## 2.3 Finding Good Satellite Passes

The second main objective is some reliable measure of where to look for high SNR satellite pulses. We need this so as to not waste our time computing data for pulses that are either very faint or not long enough (our usual length cut is about 120 seconds). We have two primary jsons that keep data, which serve two different purposes.

**/satdet/satdet_XM_refMARSX.json** records all the detections for each chunk of each pulse for each antenna. It's the output of get_satdet.py, so it's messy but exhaustive. It is also the home of a 'summary' section which stores all the consensus specnumoffsets for each antenna. The question it aims to answer is what antenna we see, how we align them, and whether it's worth looking somewhere. Note that it does not actually record whether there is any holes in the antenna data; it is simply recorded as a non-detection. We also do not index the pulses in any way in satdet, since we have not yet determined if they are valuable or will be used.

An example of why this dump might be useful: the image shows satellite SNR with time, for two different batches overlayed. 

![alt text](./images/snrs_times_properly.png)

**/data/pulses.json** records all the pulses we actually use in the later steps. Upsampling is computationally expensive, so we want to use pulse selection sparingly and cleverly. The relevant script is 'get_pulse_list.py', which finds all the times where there is a satellite risen for N continuous chunks at a minimum of X SNR. The script automatically checks the data presence for each antenna, and records which ones have missing data so as to inform the later upsampling scriupts. Due to the unreliability of detections on longer baselines, we select a short baseline to run the script on, with a reasonably high SNR floor. This way, we can gauge which pulses are bright. (TODO: We then verify there is a detection (need not be as high SNR) in the other baselines). This json answers the question: where do you want me to look for a timing solution, and what antenna can I trust to have data there? The computation of pulses.json marks the first time we actually index pulses by their starting time (rounded to nearest integer), since we can now reliably say there are visible, detected, and have sufficient SNR.

An important conceptual point is that when making this pulse list, we don't actually care about the exact times, we are free to round to the nearest integer. It serves as its 'name' for the rest of the pipeline. Using this approximate starting time, the timestream loads up data at this approximate time. We upsample this data and store it, recording the exact starting spectrum of the pulse. This is the value which is fixed forever, and holds the time information of the pulse. Since the clock are unreliable, we find a spectrum-to-UTC time mapping, which then gives us the actual starting time of the pulse. That gives its exact starting time, not its UTC name (which only serves to give a second-level precise idea of where to look for data).