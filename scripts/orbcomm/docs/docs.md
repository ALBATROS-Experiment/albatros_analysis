# Explaining the Pipeline

Hello and welcome to Thomas' super amazing timing solution bonanza. This is a quick (non-exhaustive!) outline of what each step does, what the relevant scripts and functions are, and what conventions are used! This could be added to the github wiki, made into markdown, or somewhere else it makes sense to explain this thing.

## 0. Outline, Conventions, Relevant Theory

To begin, we need to establish some basic knowledge about the project. There are several design choices that require some context to explain and justify, making a bare understanding of the subject matter indispensable.

The objective of the pipeline is to align the antenna timestreams (with respect to each other) to nanosecond level precision. This matters a lot in inteferometry, since information lives in the relative signal delays and phases. ALBATROS antenna have clocks which are not synchronized to each other, so we have order 1 second offsets (huge). All the data is saved to hard-drive (it's the arctic) so all analysis has to be done off-site, off-line, off-the-cuff (okay not really but you see what I mean). The antenna also see the whole sky, meaning that we can't just look at a bright source for calibration (annoying). Moreover, for various reasons Mohan told me but I can't quite remember, we can't use an artificial beacon (also annohing). Therefore, we need to use satellites that pass overhead as our point of reference to get our timing solution.

Our ADC samples at 250 MSPS, each antenna PFBs the incoming timestream into 2048 channels (4096-point FT step). Thus our baseband data has spectra of channel width ~61 kHz and period ~16 microsecs. 

Couple useful things to note:
- delays in time manifest as phase rotations in the frequency domain
- (Wiener-Khinshin) the FT of the autocorrelation aka coherence in delay space is the power spectrum
- PFBs and re-channelization

The way we do this is best outlined in the following flowchart:

IMAGE!!

There are a bunch of conventions: 
- refant stuff (ref-nref ?)
- pass graduates to pulse if we see it


## 1. In the beginning, there were batches


The first thing we care about is checking what data we have, and where there are any discontinuities.
For that, there is a notebook called read_batches, which mainly uses two functions from baseband_utils, called 'get_present_files' and 'get_simul_files'.
Given some inputs of start and end times, alongside antenna paths, these will give you and plot for you all the contiguous 'batches' of data where each antenna has data present.
The notebook also has the nice feature of auto-generating config files for each desired batch. These are obviously customizable.

IMPORTANT: Config files serve as the sort of 'home base' for each iteration of data. 
Each antenna re-starts every 24 hours or so, meaning batches are the natural day-long ish chunk of data to analyze.
Moreover, during their re-start, they change alignment, so it's important we only work within self-consistent 'batches'.
Config files contain all the main information that is needed to get started with our analysis.


## 2. Satellite Detection (known affectionately as 'satdet')

This is the first big analysis, which as you may or may not have guessed, involves detecting satellites. The appropriate, callable module is 'get_satdet.py'. There are two main objectives to satdet: determine when we can actually see what satellite, and determine the best spectrum alignment for each baseline. Naturally, doing this pairwise is a pain for each antenna, so we pick a reference antenna beforehand, so everything can be determined by relative offsets.

### a. Doing the 'det'

Using satellite trajectories overhead, we predict when one of them passes overhead and becomes visible to the antenna (orbcomm_utils.get_risen_sats). For each baseline containing the reference antenna, we perform a coarse cross-correlation. If the SNR is high enough, we call a satellite 'pass' a 'pulse', and it counts as a detection. For each such detection, we record the spectrum number offset of the peak of the cxcorr (which is a function of delay, as you may recall).


### b. Specnumoffsetting

This part addresses the first of the main objectives of satdet: determining spectrum alignment.
There are plenty of pulses, some of which yield different values of 'specnum offset'?
We must determine a consensus. The relevant script is 'get_consensus_offset.py'


### c. Pulses and Data (what who where when why?)

The second main objective is some reliable measure of where to look for high SNR satellite pulses. We need this so as to not waste our time computing data for pulses that are either very faint or not long enough (our usual length cut is about 120 seconds). We have two primary jsons that keep data, which serve two different purposes.

**/satdet/satdet_XM_refMARSX.json** records all the detections for each chunk of each pulse for each antenna. It's the output of get_satdet.py, so it's messy but exhaustive. It is also the home of a 'summary' section which stores all the consensus specnumoffsets for each antenna. The question it aims to answer is what antenna we see, how we align them, and whether it's worth looking somewhere. Note that it does not actually record whether there is any holes in the antenna data; it is simply recorded as a non-detection. We also do not index the pulses in any way in satdet, since we have not yet determined if they are valuable or will be used.

An example of why this dump might be useful: the image shows satellite SNR with time, for two different batches overlayed. 

![alt text](./snrs_times_properly.png)

**/data/pulses.json** records all the pulses we actually use in the later steps. Upsampling is computationally expensive, so we want to use pulse selection sparingly and cleverly. The relevant script is 'get_pulse_list.py', which finds all the times where there is a satellite risen for N continuous chunks at a minimum of X SNR. The script automatically checks the data presence for each antenna, and records which ones have missing data so as to inform the later upsampling scriupts. Due to the unreliability of detections on longer baselines, we select a short baseline to run the script on, with a reasonably high SNR floor. This way, we can gauge which pulses are bright. (TODO: We then verify there is a detection (need not be as high SNR) in the other baselines). This json answers the question: where do you want me to look for a timing solution, and what antenna can I trust to have data there? The computation of pulses.json marks the first time we actually index pulses by their starting time (rounded to nearest integer), since we can now reliably say there are visible, detected, and have sufficient SNR.

An important conceptual point is that when making this pulse list, we don't actually care about the exact times, we are free to round to the nearest integer. It serves as its 'name' for the rest of the pipeline. Using this approximate starting time, the timestream loads up data at this approximate time. We upsample this data and store it, recording the exact starting spectrum of the pulse. This is the value which is fixed forever, and holds the time information of the pulse. Since the clock are unreliable, we find a spectrum-to-UTC time mapping, which then gives us the actual starting time of the pulse. That gives its exact starting time, not its UTC name (which only serves to give a second-level precise idea of where to look for data).


## 3. What UTC time is it Mr. Wolf

So we have aligned our timestreams to the correct spectrum, and have a handy-dandy list of pulses to look at (where we know we will see something).
But do we know what actual time it is for these pulses? We know the clocks suck, so there must be some shift between UTC time and 'file time'. 
To get the most accurate beamform possible (super important for later clock drift corrections!) we need a better estimate of UTC time. 
(can give the quick demo for how it's important)
Let's find it.

The fitting cost surface typically looks something like this:
![alt text](./cost_curve_timing.png)

## 4. Fine Timing

Once we have proper spectrum alignment, and good UTC timing that corresponds to the reference antenna spectra, we can start on a timing solution.
We beamform properly onto the satellie signal to remove its geometric delay.
We then compute the visibilities.
We expect the only remaining delay to be due to the clock.
Funadmentally, any delay in the time domain will manifest as a phase ramp in the visibilities: basic FT concept.
Therefore, to find the correct delays, we need to fit for the phase ramp gradient of the visibilities across frequency.
That's why we use upsampled data: finer sampling in frequency domain.
You may be wondering if we have high enough sampling in time domain. Simple answer: yes. Longer answer: hopefully. 
Let me elaborate, some theory is important to understand why we do what we do.

Naturally, the best scenario is to just fit a phase ramp across each time sample and be done; have gradient with time.
Our system is 0.1 rad phase noise ish, and 138 MHz carrier frequency. 
We want to determine what timing error we will incurr, and see what measures we should take in accordance.
(write the derivation and explain what it means)

This means that we must use a more clever approach; we call it peak tracking.
Notationally, we can split up the total delay, tau(t) = tau_s(t) + tau_0, small scale and net offset.
The correlation function of a narrow-band white noise signal such as ours is a sinc-modulated cosine. 
Near the peak, it looks like a flat cosine.
Delay shifts correspond to left-right shifts in the correlation space (aka delay space!).
If we assume that the clock does not drift by more than pi radians=3.5ns in one time sample, we can do an iterative fit and stay with one peak.
So all we need is a reasonable guess and we can determine tau_s(t). 
We can then align the phases, average over time, and get a much better SNR for a single phase ramp to get tau_0.

In practice, we can't just take the whole pulse, since bad patches of data will completely tank the algorithm; we need to cut first.
The way we do that is with phases: we take the full indexed pulse, then get thermal noise on each timestamp. ('get_thermal_noise()')
Since we expect each to have SOME phase ramp, we can determine how noisy they are by the noise from a rapid fit on each timestamp.
We then select the best ~2mins of clean, high signal visibilities. Why 2 minutes ish? This gets into what we're trying to do.
The two minute thing, although sort of arbitrary, is long enough for good time evolution and high SNR, but also not long enough that it gets bad.
Since phase noise is about 0.1 rads for each timestamp, the time averaged phase noise is /120 seconds so about about 0.01, which is acceptable (write what it yields, plus better with joint fit)

After we have a nice cut of data, we run the following:
- get good guesses (for each baseline)
- plot the cost space (as function of one tau, 1D space) as a sanity check
- get noise, weights for each timestamp
- do that iterative fit to track cost peak
- get tau_s(t), set first value to zero
- use it to align visibility phases, plot them
- average together, get ramps, get noise on them
- fit lines for each of the taus

Then the full timing solution is the sum of both; by convention we let the first value of tau_s(t) be zero such that tau_0 is the net initial offset at the start of the pulse.
IMPORTANT: tau(t) is the delay WITHIN the timestamp, i.e. in the middle.
If you assign the delay value to the first baseband spectrum corresponding to the visibility, you will get a net shift, since the delay is averaged (visibilities) across the integration time.
So in actuality, you need to use your accumulation time to determine what spectrum best corresponds to the offset.


## 5. Putting everything together, and various helper scripts along the way

Add stuff here!