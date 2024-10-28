from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from tqdm import tqdm

import iricore as iri
from pymap3d import azel2radec, radec2azel

# This function needs to be run only once to update the index database
# update()

# Lets define constants
dt = datetime(year=2021, month=4, day=11, hour=12)

# First antenna coords in degrees
lat1, lon1 = 0, 0
# Second antenna coords in degrees
lat2, lon2 = 0, 0

# Can be regular or irregular grid; don't recommend going lower than 90 km
# Also, 2000 km is the IRI upper limit. Third parameter is number of points.
# More points - slower, but more accurate calculation. Default is 1000 (~2km/step)
heights = np.linspace(90, 2000, 500)

# If heights are not passed as an array, they can be controlled by
# parameters hbot, htop, npoints which create a regular grid

# iricore has two methods for sTEC calculations: iricore.stec() and iricore.refstec()
# First calculates sTEC along the line of sight, second implements simple refraction
# Both have the same signature, except iricore.refstec() also requires a frequency of
# the observation in MHz

# Test 1
# At very high frequencies both methods should give the same result
el, az = 0, 0
freq = 5000  # MHz
test_tec = iri.stec(el, az, dt, lat1, lon1, heights=heights)
test_tec2 = iri.refstec(el, az, dt, lat1, lon1, freq, heights=heights)
print(f"Test 1:\n   stec: {test_tec:.3f}, refstec: {test_tec2:.3f}, diff: {test_tec2 - test_tec:.3f}")
np.testing.assert_almost_equal(test_tec, test_tec2, decimal=2)

# Each method also has a parameter "return_hist". If True - a ray tracking data is returned.
# This is a dictionary with keys:
# 'edens' - electron density data along the way
# 'lat' - geographic latitude of the ray at specified height
# 'lon' - geographic longitude of the ray at specified height
# 'h' - corresponding height - does not include a starting point
# 'ds' - delta path between specified layers. Important! First value is distance from the
#        ground to the first layer, which can't be used in the integration. Second point is
#        the distance from the first layer to the second and so on

# Test 2
# Visualizing path difference
el, az = 0, 0
freq = 40  # MHz
test_tec, hist = iri.stec(el, az, dt, lat1, lon1, heights=heights, return_hist=True)
test_tec2, hist2 = iri.refstec(el, az, dt, lat1, lon1, freq, heights=heights, return_hist=True)

plt.plot(hist['lat'], hist['h'], label="stec")
plt.plot(hist2['lat'], hist2['h'], label="refstec")
plt.xlabel("Latitude")
plt.ylabel("Height")
plt.title(f"Test 2\nstec: {test_tec:.3f}, refstec: {test_tec2:.3f}, diff: {test_tec2 - test_tec:.3f}")
plt.legend()
plt.show()


# Test 3: delta sTEC difference for two albatros antennas
lat1, lon1 = 79.456944, 90.800833
lat2, lon2 = 79.388333, 91.019167

# Selecting a single coordinate on the sky for both antennas to look at
# Important! In case of raytracing two rays are not guaranteed to end up in the same point
# in the sky

el1, az1 = 80, 20
ra, dec = azel2radec(az1, el1, lat1, lon1, dt)
az2, el2 = radec2azel(ra, dec, lat2, lon2, dt)

freq = 10  # MHz

stec_plain1, hist_p1 = iri.stec(el1, az1, dt, lat1, lon1, heights=heights, return_hist=True)
stec_plain2, hist_p2 = iri.stec(el2, az2, dt, lat2, lon2, heights=heights, return_hist=True)
stec_ref1, hist_r1 = iri.refstec(el1, az1, dt, lat1, lon1, freq, heights=heights, return_hist=True)
stec_ref2, hist_r2 = iri.refstec(el2, az2, dt, lat2, lon2, freq, heights=heights, return_hist=True)

delta_stec_plain = stec_plain2 - stec_plain1
delta_stec_ref = stec_ref2 - stec_ref1
print(f"Test 3:\n   sTEC: {delta_stec_plain:.2e}, sTEC_ref {delta_stec_ref:.2e}, Relative diff: "
      f"{(delta_stec_plain - delta_stec_ref) / delta_stec_plain * 100 :.3f} %")


# Test 4
# Same as test 3, but with time evolution

# dts = [datetime(year=2021, month=4, day=11, hour=i) for i in range(23)]
dts = [datetime(year=2023, month=4, day=11, hour=i) for i in range(23)]
freq = 10
ddstec = np.empty(len(dts))
dstecplain = np.empty(len(dts))

# Less points for speed, but dont make it less than ~300
heights = np.linspace(90, 2000, 300)

for i, dt in enumerate(tqdm(dts)):
    stec_plain1, hist_p1 = iri.stec(el1, az1, dt, lat1, lon1, heights=heights, return_hist=True)
    stec_plain2, hist_p2 = iri.stec(el2, az2, dt, lat2, lon2, heights=heights, return_hist=True)
    stec_ref1, hist_r1 = iri.refstec(el1, az1, dt, lat1, lon1, freq, heights=heights, return_hist=True)
    stec_ref2, hist_r2 = iri.refstec(el2, az2, dt, lat2, lon2, freq, heights=heights, return_hist=True)
    ddstec[i] = stec_ref2 - stec_plain2 - stec_ref1 + stec_plain1
    dstecplain[i] = stec_plain1 - stec_plain2


plt.plot(ddstec)
# plt.plot(dts, ddstec)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%h %m'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.ylabel("Difference, TECU")
plt.xlabel("Hour")
plt.title("Test 4")
plt.show()
