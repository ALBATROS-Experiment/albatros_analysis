from utils.orbcomm_utils import get_risen_sats, find_pulses, gauss_smooth

def filter_single_sat(risen_sat_count):
    """Get epoch start and end indices corresponding to periods during which only one satellite is risen

    Parameters
    ----------
    risen_sat_count : ndarray of int
        count of risen satellites at each epoch
    """

    ss_epochs = find_pulses(risen_sat_count, cond="==", thresh=1) #signal is e.g. _______|-----|____|---|__ ON when only 1 sat

def find_single_sat_transits(spectra, acctime=None):

    nspec, nchan = spectra.shape
    spectra = 10*np.log10(spectra/np.median(spectra)) # convert to SNR in dB. median inside log same as median outside log since log is strictly monotonously increasing.
    spectra = np.apply_along_axis(smooth,0,spectra)
    transits = {}
    for chan in range(0, nchan):
        transits[chan] = find_pulses(spectra[:,chan], cond=">", thresh=5)



