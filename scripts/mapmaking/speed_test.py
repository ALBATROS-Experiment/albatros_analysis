import healpy as hp
import numpy as np
import cupy as cp
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import time
import rfitools
import os
import sys
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

ev1 = cp.cuda.Event()
ev2 = cp.cuda.Event()
delays = np.random.randn(21 , 12582912)
print("delays size", delays.nbytes/1e9)
for i in range(10):
    ev1.record()
    d_delays = cp.asarray(delays, dtype=cp.float32)
    ev2.record()
    ev2.synchronize()
    print("GPU setup time:",cp.cuda.get_elapsed_time(ev1, ev2)/1000,"seconds")
sys.exit()

coords = {
    0: [79.417161473, -90.767238685, 187.9577],   # MARS1
    1: [79.417198047, -90.758739192, 183.0684],   # MARS2
    2: [79.388456412, -91.019202963, 25.1938],    # MARS4
    3: [79.418302573, -90.667395452, 59.6242],    # MARS5
    4: [79.397984238, -90.799842408, 41.6994],    # MARS6
    5: [79.411474117, -90.695266129, 31.6314],    # MARS7
    6: [79.443757694, -90.718202634, 414.9131]    # MARS8
}

antmap = {0: "MARS1", 1: "MARS2", 2: "MARS4", 3: "MARS5", 4: "MARS6", 5: "MARS7", 6: "MARS8"}

# ============================================================
# Baseline Stuff

nant = 7
triu_idx = np.triu_indices(nant, k=1)
nbl = len(triu_idx[0])
print("total baselines:", nbl)

bl_enus = rfitools.get_all_bls(coords,np.arange(len(coords)))
print("baseline ENU shape:", bl_enus.shape)

# ============================================================
# pixel count
NSIDE = 1024
NPIX = hp.nside2npix(NSIDE)
print("HEALPix resolution:", hp.nside2resol(NSIDE, arcmin=True), "arcmin")
print("number of pixels:", NPIX)

# general pixel coordinates
co_dec, ra = hp.pix2ang(NSIDE,np.arange(NPIX))
dec = np.pi/2 - co_dec
src = SkyCoord(ra=ra * u.rad,dec=dec * u.rad,frame='icrs')

# reference antenna
ant0 = EarthLocation.from_geodetic(lat=coords[0][0], lon=coords[0][1], height=coords[0][2])
obstime = Time(178650000 + np.arange(200), format="unix", scale="utc")
frame = AltAz(location=ant0,obstime=obstime)

t1=time.time()
# out1=src[:, None].transform_to(frame)
out1=src[0].transform_to(frame)
t2=time.time()
# print(t2-t1)
# print(out1.shape)

t1=time.time()
# out1=src[:, None].transform_to(frame)
out1=src[1].transform_to(frame)
t2=time.time()
print(t2-t1)
print(out1.shape)


t1=time.time()
with erfa_astrom.set(ErfaAstromInterpolator(300 * u.s)):
    # out2 = src[:, None].transform_to(frame)
    out2 = src[1].transform_to(frame)
t2=time.time()
print(t2-t1)
print(out2.shape)


