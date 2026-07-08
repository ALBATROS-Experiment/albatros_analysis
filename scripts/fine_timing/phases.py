import os
import sys
sys.path.append(os.path.expanduser('~/albatros_analysis'))
import numpy as np
import cupy as cp
import helper as hp_f
import figures as fgs
import json
import argparse
from src.utils import orbcomm_utils as outils
from src.utils import orbcomm_utils_gpu as outils_g
from src.utils import baseband_utils as butils
from src.correlations import baseband_data_classes as bdc
from scripts.orbcomm import sat_utils as su
from scripts.orbcomm import sat_utils_gpu as sug
from scripts.xcorr import helper as hp_x
import matplotlib.pyplot as plt


