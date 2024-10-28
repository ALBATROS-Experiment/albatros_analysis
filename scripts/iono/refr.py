from dionpy import *
import datetime
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0,'/home/mohan/Projects/')
from albatros_analysis.src.utils import orbcomm_utils as outils
import iricore as iri
T_SPECTRA = 4096/250e6
T_ACCLEN = T_SPECTRA * 393216
t1 = 1627453681 + 150 * T_ACCLEN
t2 = 1627453681 + 180 * T_ACCLEN
pos1 = [51.4646065, -68.2352594, 341.052]  # north antenna
pos2 = [51.46418956, -68.23487849, 338.32526665]  # south antenna
satnorads = [40087, 41187]
niter=int(t2-t1)+2
times=np.arange(0,niter)
delay,altaz1,altaz2=outils.get_sat_delay(pos1, pos2, "/home/mohan/Projects/albatros_analysis/scripts/orbcomm/orbcomm_28July21.txt", t1, niter, satnorads[0],altaz=True)

def get_true_alt(alt_tle, az_tle, freq, frame):
    old_alt=alt_tle.copy()
    for i in range(10):
        print("iter i",i)
        refr=frame.raytrace(old_alt,az_tle,137)[0]
        # print(alt_tle, old_alt+refr)
        eps=alt_tle-(old_alt+refr)
        # print("iter", i, "cur alt", old_alt, "refr at this alt", refr, "error is", eps, "tol at", np.abs(eps/old_alt))
        if(np.all(np.abs(eps/alt_tle)) < 1e-12):
                print('approaching convergence')
                # print('final errors', eps)
                break
        # print(old_alt)
        old_alt+=eps
    return old_alt

# pos1 = (79+24.925/60, -90-46.385/60, 175)
# pos2 = (79+24.925/60, -90-46.385/60, 175)
dt=datetime.datetime.utcfromtimestamp(t1+100)
frame1 = IonFrame(dt, pos1, hbot=100, htop=1000, nlayers=51)
# alt_tle=np.linspace(5,80,101)
# az_tle=np.tile(67,len(alt_tle)).astype("float64")
# az_tle=np.asarray([67.,67.])
alt1_new = get_true_alt(altaz1[:,0],altaz1[:,1],137,frame1)
frame2 = IonFrame(dt, pos1, hbot=100, htop=1000, nlayers=51)
alt2_new = get_true_alt(altaz2[:,0],altaz2[:,1],137,frame2)

# print(alt1_new, alt2_new)

# heights=np.linspace(100,1000,51)
heights=10**np.linspace(2,3,51)


tec1=[]
tec2=[]
for i in range(len(alt1_new)):
    tec1.append(iri.refstec(alt1_new[i],altaz1[i,1], dt, pos1[0], pos1[1], 137, heights=heights, return_hist=False))
for i in range(len(alt2_new)):
    tec2.append(iri.refstec(alt2_new[i],altaz2[i,1], dt, pos1[0], pos1[1], 137, heights=heights, return_hist=False))
tec1=np.asarray(tec1)
tec2=np.asarray(tec2)

plt.title("phase delay @ 137 MHz at b/w 2 antennas")
print(tec2-tec1)
print((tec2-tec1)*8082/137)
plt.plot((tec2-tec1)*8082/137/(2*np.pi*137e6))
plt.xlabel("time (s)")
plt.ylabel("delay (s)")
plt.show()