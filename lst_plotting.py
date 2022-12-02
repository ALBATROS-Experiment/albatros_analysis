# Modules
import ephem as e
import skyfield as sk
import datetime as dt
import os
from scio import scio

# Custom scripts
import diagnostics as dg

# TODO: think about moving this to a file maybe?
# Config
nBins = 100


def setLocation(local, obs): # TODO: these will need to be filled in appropriately
    if "mars" in local.lower():
        print("Set location: MARS")
        obs.lat, obs.lon, obs.elevation = "79.35242455716913", "-90.79314803147464", 0
    if "marion" in local.lower():
        print("Set location: Marion Island")
        obs.lat, obs.lon, obs.elevation = "-46.886802", "37.819775", 208.5
    if "uapishka" in local.lower():
        print("Set location: Uapishka")
        obs.lat, obs.lon, obs.elevation = "51.4641932", "-68.2348603", 0

def findType(filename):
    pol = os.path.splitext(os.path.basename(filename))[0]
    if pol[-1] == pol[-2]: return "auto"
    else: return "cross"

# a = e.Observer()
# a.date = dt.datetime.utcfromtimestamp(1669923718)
# b = a.sidereal_time()
# setLocation("marion island", a)