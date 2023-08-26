import sys
from os import path
import numpy as np
sys.path.insert(0, '/home/mohan/Projects/albatros_analysis/')
from utils.orbcomm_utils import get_risen_sats, find_pulses, gauss_smooth
from matplotlib import pyplot as plt

def filter_single_sat(risen_sat_count):
    """Get epoch start and end indices corresponding to periods during which only one satellite is risen

    Parameters
    ----------
    risen_sat_count : ndarray of int
        count of risen satellites at each epoch
    """

    ss_epochs = find_pulses(
        risen_sat_count, cond="==", thresh=1
    )  # signal is e.g. _______|-----|____|---|__ ON when only 1 sat
