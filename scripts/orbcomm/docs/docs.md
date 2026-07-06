# Explaining the Pipeline

Hello and welcome to Thomas' super amazing timing solution bonanza. This is a quick (non-exhaustive!) outline of what each step does, what the relevant scripts and functions are, and what conventions are used! This could either be added to the github wiki, or added somewhere else so people can understand what we're doing.

## 0. Outline, Conventions, Relevant Theory

To begin, we need to establish some basic knowledge about the project. There are several design choices that require some context to explain and justify, making a bare understanding of the subject matter indispensable.

The objective of the pipeline is to align the antenna timestreams (with respect to each other) to nanosecond level precision. This matters a lot in inteferometry, since information lives in the relative signal delays and phases. ALBATROS antenna have clocks which are not synchronized to each other, so we have order 1 second offsets (huge). All the data is saved to hard-drive (it's the arctic) so all analysis has to be done off-site, off-line, off-the-cuff (okay not really but you see what I mean). The antenna also see the whole sky, meaning that we can't just look at a bright source for calibration (annoying). Moreover, for various reasons Mohan told me but I can't quite remember, we can't use an artificial beacon (also annohing). Therefore, we need to use satellites that pass overhead as our point of reference to get our timing solution.

Our ADC samples at 250 MSPS, each antenna PFBs the incoming timestream into 2048 channels (4096-point FT step). Thus our baseband data has spectra of channel width ~61 kHz and period ~16 microsecs. 

The way we do this is best outlined in the following flowchart:

IMAGE!!

There are a bunch of conventions: 
- refant stuff (ref-nref ?)
- pass graduates to pulse if we see it

There is also some stuff that will easily trip you up. This problem is actually three different timestreams all mixed into one.

EXPLAIN, WITH DIAGRAMS!


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

### a. On Upchannelization

Our baseband data does not have sufficient frequency resolution to sufficiently sample phase ramps across frequency. Therefore, we need to re-channelize our data into finer channels. This data is used throughout the rest of the pipeline. We upchannelize using a re-PFB pipeline written by Mohan, found in '/scripts/xcorr'. The re-PFB script that is mainly used in this pipeline is 'fine_timing.py', but the engine for this is found elsewhere in the same directory.

We usually upchannelize by a factor of 64, meaning channels are about 1 kHz wide. Thus, since METEOR satellites we have between 80 and 100 kHz of bandwidth, we have 80-100 channels of signal. 

We use the recorded pulse times from 'pulses.json', alongside the spectrum alignment to generate the upchannelized data for each pulse. We now record the spectrum number of the first spectrum used in the upchannelized data of the reference antenna (to which, if you recall, all other antenna timestreams are aligned), and keep it in the same 'pulses.json' file. This spectrum number is now anchored as the starting point of the pulse, as the upchannelized data starts at that point. Also notice that we are upchannlizing, meaning that each new spectrum contains 64 old spectra: we sample less often in time. (we have to be careful with spectrum number indexing; it's our only reliable measure of data poition in time, but we can easily get confused).

The data is stored in the '/batchX/data' folder, and is named according to the rounded pulse time found in the pulses.json file. Note that these do not contain much information about the exact time of the pulse, but instead serve more as an indexing tool to facilitate naming and differentiating pulses.

### b. Absolute Alignment (UTC)

Now that we have upchannelized, high SNR, spectrally aligned antenna data, we might think that we are done. Alas, no. To isolate clock delay, we must first remove the satellite's geometric delay. We know the exact satellite positions, so removing geometric delay is a simple exercise in, well, you guessed it, geometry. However, we don't know what actual abolute UTC time the data in our antenna corresponds to. That means any estimate we make of satellite position incurrs some error, and therefore some delay offset. Depending on the geometry of the setup, this may be a large error. For example, consider XXXX

(insert example derivation)

Thankfully, they are all reasonably closely aligned with respect to each other (by spectrum), so we can just find a single net UTC offset that will best fit the entire setup. The question is, how do we localize our data in UTC time?

We fit on unwrapped phases. We expect the only major drift in unwrapped phases (more significant than clock noise) to be the residual drift experienced by the visibilities due to the growth in offset between actual position and beamformed position. The fit is performed with respect to the starting time in the pulses.json file. The fitted time is then the difference with with respect to that time.

Since we fit on unwrapped phases, we must first cut the pulse such that unwrapping issues do not corrupt the fit. Since we are not aligned to UTC time yet, and we are cut by visibility amplitude. Use XXXX to get best continuous 2 minutes. Results saved in '/data/cutting_UTC.json' so as to facilitate repeated runs for the same pulse. 

The fitting cost surface typically looks something like this:
![alt text](./scripts/cost_curve_timing.png)

### c. UTC Mapping

Once we have a fit for each pulse throughout the batch, we want to find a single mapping between UTC time and pulses. Clock drift does not exceed a couple microseconds over the course of a batch, so to extremely good approximation UTC time and spectrum number can be related by a linear relationship. Using the starting fitted starting time and starting spectrum number as our data, we can fit a line and get a best fit mapping to get from spectrum number to UTC time. This is the mapping we use to determine UTC time throughout the entire batch. The fit (with residuals) looks something like this:

(INSERT FIT AND RESIDUALS FIGURE)

## 4. Fine Timing
As a recap, we now have proper spectrum alignment, and good UTC timing that corresponds to the reference antenna spectra.

Once we have , we can start on a timing solution. To remove geometric delays, we beamform onto the satellie souce, using the corrected UTC time to be as accurate as possible. We expect the only remaining source of delays to be clock delay and noise. Then, we compute the visibilities on all baselines. The integration time is usually between 0.5 and 1 seconds (since the clock drift not expected to exceed 1 ns for such timescales ????). We do everything joinly on all baselines, but for the purpose of explanation I will only consider one. Moreover, we don't consider ionospheric effects here.

Fundamentally, any delays between antenna timestreams in the time domain will manifest as linear phase ramps across the visibilities. Therefore, to correct for delays over the course of the pulse, we must determine the gradient of the phase ramp across frequency, for each time sample. Our upsampled data gives us access to more datapoints across frequency. The largest residual delay is of order 1 spectrum, i.e. 20 microseconds. This would cause a phase wrapping frequency of about 50 kHz. Therefore, with our channel resolution of 1 kHz, we do not have to worry about the signal causing the wrapping of phase across channels. When there is clear signal, we have phase noise of about 0.1 radians for our typical integration times, meaning we do not have to worry about unwrapping issues due to noise.

Below is an example of a visibility phase plot, next to its phase noise over time, for a single baseline of X km. Notice that there are some bad parts. To ensure the fits are not compromised, and to ensure that we do not suffer noise that would cause unwrapping problems across phase, we must often cut our visibilities to approximately 2 minutes of the cleanest signal. The way this is done is to get thermal noise on each timestamp using 'get_thermal_noise()', and to cut the pulse to get the contiguous 2 minutes with the lowest thermal noise. The signal channels are taken from the UTC cutting. The cut times and channels are then recorded in 'cutting_finetiming.json', to facilitate re-running the pulse in the future. 

Naturally, the best scenario is to just fit a phase ramp across each time sample and be done; have gradient with time. The way that is done is an ordinary least squares line fit on the unwrapped phase data. The extracted value is the 'group delay'. The fit yields an error of 
$$
    \sigma_g = \frac{\sqrt{12},\sigma_{\phi}}{2\pi B\sqrt{N}},
$$
which gives about 70 ns for our case of 0.1 rad phase noise ish, and 138 MHz carrier frequency. Too large, so can't just do a straight up linear fit for each time, we gain very little information. This means that we must use a more clever approach; we call it peak tracking. 

If the visibility phases can first be aligned relative to a common reference, the data may be averaged coherently in time. Such averaging reduces the phase uncertainty by approximately $\sqrt{b}$, leading to a corresponding improvement in the group-delay precision. The figure below illustrates the phase-alignment procedure applied to the simulated visibilities. After coherent averaging, the group-delay uncertainty is sufficiently reduced to localize the overall delay of the observation to within a small number of carrier peaks.

(PHASE ALIGNMENT PLOT BELOW!)

The delay fitting strategy is therefore to split up the full time dependent group delay into an initial overall delay $\tau_0$, and small scale time-dependent variations $\tau_s(t)$, such that $\tau(t) = \tau_0 + \tau_s(t)$. The $\tau_s(t)$ serve both as the terms which cause the SNR boost, but also encode the actual clock drift with time. The small scale changes are by convention zero for the first visibility spectrum, such that $\tau_0$ represents the delay at the start of the visibility data. In practice, the final timing solution $\tau(t)$ is determined simultaneously using all baselines in the array, which also yields higher confidence in delays. For clarity, however, we the simulation and description are done for a single baseline only. 

The small-scale time-dependent component $\tau_s(t)$ is determined by a nonlinear least-squares fit to each time-sample spectrum. The visibility phases are expressed in complex form as $e^{j2\pi \nu_0 \tau(t)}$, and a corresponding model of the form $e^{i\theta(t)}$ is fitted using the Levenberg–Marquardt algorithm. The resulting cost function is proportional to $1 - C(\tau)$, where $C(\tau)$ denotes the normalized autocorrelation function defined in earlier.

Consequently, coherence maxima in the autocorrelation correspond to minima of the cost function, retaining the same sinc-modulated cosine structure derived previously. These minima define phase-delay solutions, which occur periodically with a spacing set by the carrier frequency. Variations in group delay manifest as a translation of the cost surface along the delay axis, such that the autocorrelation envelope is shifted by the corresponding clock offset between time samples. The fitting procedure therefore identifies the nearest local minimum relative to the initial guess and tracks its evolution over time. By sequentially updating the solution using the most recent estimate, the algorithm remains locked to a consistent branch of the phase-delay ambiguity and follows its temporal evolution throughout the visibility dataset. Under this formulation, the uncertainty in the phase-delay estimate is determined solely by the phase noise and the carrier frequency:
$$
    \sigma_p = \frac{\sigma_\phi}{2\pi\nu_0}.
$$
For our values, this value is on the order of 0.1 ns. 

There are, however, some constraints to peak tracking. First, the cost surface must be well-behaved for the fitter to reliably pick the nearest cost minimum. Near the sinc modulation maximum, the cost function looks like a flat sinusoid, and therefore yields reliable fitting. Therefore, when initializing a fit, we make a rough guess using linear least squares, to place our tracking within the reliable section of the cost surface. This is the reason we do an initial rough guess with about 10 time samples to get guess taus.

Secondly, the fitting must be able to unambiguously resolve $2\pi$ ambiguities. If the clock drifts by half of the carrier wave period, which corresponds to a phase shift of $\pi$, the cost surface does not encode anything about which direction the drift went, and therefore the fitter cannot differentiate between a drift forward by $\pi$ or backward by $\pi$. This can lead to a misinterpreted relationship between phase delay and the underlying group delay, as shown in the left plot below. We do not expect the clock to drift by more than a nanosecond, so this type of ambiguity would be arise as the consequence of a highly unlikely event in our setup. (QUANTIFY CLOCK NOISE)

(AMBIGUITY EXPLANATION PLOT HERE)

To elaborate on the figure above, the first three colors represent well-behaved delay changes across time-samples, which unambiguously map phase delay to group delay evolution. Red represents an ambiguous case. The left plot shows the cost structure for various time-samples, with a shifting envelope corresponding to delay drift. The ambiguous case arises when the drift is $\pi$ radians or more, as the fitter may select either of the equidistant forward or backward minima. The right plot shows what the cost surfaces would yield as phase ramps and therefore group delay. The red envelope could yield two vastly different ramps, as the fitter cannot distinguish between the two phase delay solutions.


Once $\tau(t)$ is determined for the whole pulse, we save it into a h5 file called '/batchX/fine_timing/timing_solution.py'. Since we have several pulses which we may want to evaluate independently, we save each pulse according to the file name it corresponds to. Each dataset then has shape (ntimes, nant-1) to store the solution. Moreover, we save errors on the fit overall, and for each timestamp also. For a good pulse, we expect net group delay errors of about 4 ns.

IMPORTANT: there are two different types of fit here: fitting on the cost carrier peak and fitting on the averaged phase ramp. One will actually give you a carrier peak (perhaps the wrong one) while the other will give you something close, with a better uncertainty on its value. They signify and represent different stuff. 


The final fit, once you do the time average and normal equation fit gives us the red line. The correct carrier peak is right next to it, and the fit uncertainties are right next to it. 

![alt text](./cost_surface.png)

Here is also the full with-time plot.

## 5. Putting everything together, and various helper scripts along the way

Once you've got your timing solution, you may want to do some analysis on it, understand what's going on, read it, love it, care for it, etc. There are several helper scripts to reformat and visualize the data. 

## 6. Basic Signal Processing

Collection of theorems and ideas, alongside derivations and justifications, which may help you understand this pipeline.

- basic ideas of interferometry
- delays in time manifest as phase rotations in the frequency domain
- (Wiener-Khinshin) the FT of the autocorrelation aka coherence in delay space is the power spectrum
- correlation vs least squares connection
- PFBs and re-channelization