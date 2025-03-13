import symlib
import numpy as np
from colossus.cosmology import cosmology
from colossus.halo import mass_defs, mass_so

suite = "SymphonyMilkyWay"
scale = symlib.scale_factors(suite)

param = symlib.simulation_parameters(suite)
mp, eps = param["mp"]/param["h100"], param["eps"]*scale/param["h100"]
cosmo = cosmology.setCosmology('', symlib.colossus_parameters(param))
base_dir = "/sdf/home/p/phil1/ZoomIns"

z = 1/scale - 1
t = cosmo.age(z)
print(t)
t_dyn = mass_so.dynamicalTime(z, "vir", "crossing")

def pre_snap_max(h, hist):
    return np.argmax(h["m"][:hist["first_infall_snap"]])

def is_slow_grower(h, hist, t, t_dyn, mp):
    snap_max = np.zeros(hist.shape, dtype=int)
    for i in range(len(snap_max)):
        if hist["first_infall_snap"][i] == 0:
            snap_max[i] = 0
            continue
        snap_max[i] = pre_snap_max(h[i], hist[i])

    dt = t[hist["first_infall_snap"]] - t[snap_max]
    dt_T = dt/t_dyn[snap_max]

    mf_mi = h["m"][:,hist["first_infall_snap"][i]]/hist["mpeak_pre"]
    ok1 = (hist["mpeak"]/mp > 1e4) & (dt_T > 1) & (hist["mpeak"]/mp < 1e6)
    ok2 = mf_mi > 0.5
    ok1[0] = False

    return ok1 & ok2

