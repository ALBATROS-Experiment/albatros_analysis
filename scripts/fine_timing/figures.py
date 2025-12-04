import numpy as np
import matplotlib.pyplot as plt
import cupy as cp


def make_alpha_approximator_plot(xc_fft, N2=100):
    xc_fft = cp.asnumpy(xc_fft)
    fig, ax=plt.subplots(1,2)
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22
        })
    fig.set_size_inches(8,4)
    ax=ax.flatten()
    peak_idx=np.argmax(xc_fft)
    #add SNR? peak info?

    #left plot (no zoom)
    ax[0].plot(xc_fft)
    ax[0].set_title(f'Full FFT Amplitude. Peak: {peak_idx}')
    ax[0].set_xlabel("Sampled Alpha Values ()")
    ax[0].set_ylabel("Amplitude")

    #data information
    #stats_text = f"SNR: {snr:.0f}\nMAD: {noise_amp:.4f}\nOffset: {peak_idx-100000} "
    #ax[0].text(0.02, 0.95, stats_text,transform=ax[0].transAxes,fontsize=12,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    #right plot (with zoom)
    data_chan_zoomed = xc_fft[peak_idx - N2: peak_idx + N2]
    ax[1].plot(data_chan_zoomed)
    ax[1].set_title('Zoomed FFT Amplitude')
    ax[1].set_xlabel("Sampled Alpha Values ()")

    plt.tight_layout()

    return fig


def plot_vis_alpha(xc_vis, old_alpha, new_alpha, v_acclen = 10000):
    fig=plt.figure(figsize=(8,4))
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
            "legend.fontsize": 14
        })

    plt.plot(xc_vis, label='Unwrapped Phase')
    plt.plot(old_alpha, color='green', linestyle='--', alpha = 0.7, label='Initial Alpha')
    plt.plot(new_alpha, color='red', linestyle='--', alpha = 0.7, label='Fitted Alpha')
    plt.legend()
    plt.suptitle(f'Clock Drift on Block Visibility')
    plt.xlabel(f"Visibility Chunk ({v_acclen//1000}k spectra)")
    plt.ylabel("Phase (radians)")

    plt.tight_layout()
    return fig


def plot_vis_all(xc_vis, v_acclen = 10000):
    fig=plt.figure(figsize=(8,4))
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22
        })

    plt.plot(xc_vis, label='Phase')
    plt.legend()
    plt.suptitle(f'Unwrapped Phase of Beamformed Pulse')
    plt.xlabel(f"Vis Chunk ({(v_acclen/1000):.1f}k spectra)")
    plt.ylabel("Phase (radians)")

    plt.tight_layout()
    return fig


def plot_around_guess(xc, 
                      alpha_guess, 
                      alpha_fitted, 
                      chan_b_idx, 
                      niter, 
                      step):
    
    plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.titlesize": 22,
            "legend.fontsize": 16 
        })
    
    n = np.arange(len(xc))
    nvals = 2*niter + 1
    xc = cp.asnumpy(xc)
    test_vals = alpha_guess + (step*np.arange(-niter, niter+1))
    output = np.empty(nvals)
    for i in range(nvals):
        xc_phased = xc * np.exp(1j*2*np.pi*chan_b_idx*n*test_vals[i])
        output[i] = np.abs(np.mean(xc_phased))**2

    fig = plt.figure(figsize=(8, 4))
    plt.plot(test_vals, output)
    plt.xlabel('Clock Drift (ADC/Spectrum)')
    plt.ylabel('Amplitude')
    plt.suptitle('Amplitude against Alpha Parameter')
    plt.axvline(alpha_fitted, color='r', linestyle='--', label=f"Fitted = {alpha_fitted:.3e}")
    plt.axvline(alpha_guess, color='green', linestyle='--', label=f"Guess = {alpha_guess:.3e}")
    plt.legend()
    plt.tight_layout()
    return fig



#def waterfall_phase(data, chanstart, chanend):
    


    
