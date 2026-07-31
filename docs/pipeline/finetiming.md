# 4. Fine Timing

As a recap, we now have proper spectrum alignment, and good UTC timing that corresponds to the reference antenna spectra.

Once we have , we can start on a timing solution. To remove geometric delays, we beamform onto the satellie souce, using the corrected UTC time to be as accurate as possible. We expect the only remaining source of delays to be clock delay and noise. Then, we compute the visibilities on all baselines. The integration time is usually between 0.5 and 1 seconds since the clock drift not expected to exceed 1 ns for such timescales. We do everything joinly on all baselines, but for the purpose of explanation I will only consider one. Moreover, we don't consider ionospheric effects here.

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

Secondly, the fitting must be able to unambiguously resolve $2\pi$ ambiguities. If the clock drifts by half of the carrier wave period, which corresponds to a phase shift of $\pi$, the cost surface does not encode anything about which direction the drift went, and therefore the fitter cannot differentiate between a drift forward by $\pi$ or backward by $\pi$. This can lead to a misinterpreted relationship between phase delay and the underlying group delay, as shown in the left plot below. We do not expect the clock to drift by more than a nanosecond, so this type of ambiguity would be arise as the consequence of an unlikely event in our setup.

(AMBIGUITY EXPLANATION PLOT HERE)

To elaborate on the figure above, the first three colors represent well-behaved delay changes across time-samples, which unambiguously map phase delay to group delay evolution. Red represents an ambiguous case. The left plot shows the cost structure for various time-samples, with a shifting envelope corresponding to delay drift. The ambiguous case arises when the drift is $\pi$ radians or more, as the fitter may select either of the equidistant forward or backward minima. The right plot shows what the cost surfaces would yield as phase ramps and therefore group delay. The red envelope could yield two vastly different ramps, as the fitter cannot distinguish between the two phase delay solutions.


Once $\tau(t)$ is determined for the whole pulse, we save it into a h5 file called '/batchX/fine_timing/timing_solution.py'. Since we have several pulses which we may want to evaluate independently, we save each pulse according to the file name it corresponds to. Each dataset then has shape (ntimes, nant-1) to store the solution. Moreover, we save errors on the fit overall, and for each timestamp also. For a good pulse, we expect net group delay errors of about 4 ns.

IMPORTANT: there are two different types of fit here: fitting on the cost carrier peak and fitting on the averaged phase ramp. One will actually give you a carrier peak (perhaps the wrong one) while the other will give you something close, with a better uncertainty on its value. They signify and represent different stuff. 


The final fit, once you do the time average and normal equation fit gives us the red line. The correct carrier peak is right next to it, and the fit uncertainties are right next to it. 

![alt text](../images/cost_surface.png)

Here is also the full with-time plot.